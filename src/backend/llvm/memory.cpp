#include "backend/llvm/emitter.h"
#include "backend/codegen_switches.h"
#include "cpu/cpu.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Alignment.h>

#include <cstdlib>

namespace dolllvm {

using namespace llvm;

// Spike: DOLRECOMP_MEM2_BASE moves the second direct-memory window. For GameCube builds 0xA0000000
// (the locked L1 cache at 0xE0000000 after normalization) lets the runtime map GX's scratchpad there.
static u32 mem2Base() {
  static const u32 base = [] {
    const char *v = std::getenv("DOLRECOMP_MEM2_BASE");
    return v ? static_cast<u32>(std::strtoul(v, nullptr, 0)) : static_cast<u32>(WII_MEM2_BASE);
  }();
  return base;
}

// DOLRECOMP_UNMAPPED_READS=<first>-<last>, both hex and inclusive.
bool FunctionEmitter::unmappedReads(u32 *first, u32 *last) const {
  static const struct Range {
    bool valid = false;
    u32 first = 0, last = 0;
    Range() {
      const char *text = std::getenv("DOLRECOMP_UNMAPPED_READS");
      if (!text || !text[0])
        return;
      char *end = nullptr;
      const unsigned long a = std::strtoul(text, &end, 16);
      if (end == text || *end != '-')
        return;
      const char *second = end + 1;
      const unsigned long b = std::strtoul(second, &end, 16);
      if (end == second || *end || b < a + 8ul || b > 0xFFFFFFFFul)
        return;
      first = static_cast<u32>(a);
      last = static_cast<u32>(b);
      valid = true;
    }
  } range;
  if (!range.valid || modern_runtime_ || native_abi_)
    return false;
  *first = range.first;
  *last = range.last;
  return true;
}

Value *FunctionEmitter::normalizeAddress(Value *address) {
  return builder_.CreateAnd(address, builder_.getInt32(~0x40000000u));
}

Value *FunctionEmitter::provenMemoryPointer(const DolIRInstruction &instruction,
                                            Value *address, u32 width,
                                            Value **offset) {
  if (!fixed_memory_layout_ ||
      instruction.address_domain != DOLIR_ADDRESS_MEM1 ||
      instruction.address_lower != instruction.address_upper)
    return nullptr;
  auto *constant = dyn_cast<ConstantInt>(address);
  if (!constant || constant->getZExtValue() != instruction.address_lower)
    return nullptr;
  const u32 normalized = instruction.address_lower & ~0x40000000u;
  if (expected_ram_size_ < width || normalized < GC_RAM_BASE ||
      normalized - GC_RAM_BASE > expected_ram_size_ - width)
    return nullptr;
  *offset = builder_.getInt32(normalized - GC_RAM_BASE);
  function_->addFnAttr("dolrecomp-native-memory", "mem1");
  return builder_.CreateInBoundsGEP(Type::getInt8Ty(context_), ram_, *offset);
}

Value *FunctionEmitter::rangeCheck(Value *normalized, u32 base, Value *size,
                                   u32 width) {
  Value *offset = builder_.CreateSub(normalized, builder_.getInt32(base));
  Value *largeEnough = builder_.CreateICmpUGE(size, builder_.getInt32(width));
  Value *last = builder_.CreateSub(size, builder_.getInt32(width));
  return builder_.CreateAnd(largeEnough, builder_.CreateICmpULE(offset, last));
}

Value *FunctionEmitter::endianLoad(Value *pointer, Type *resultType,
                                   u32 width) {
  Type *integerType = IntegerType::get(context_, width * 8u);
  LoadInst *load = builder_.CreateLoad(integerType, pointer, "native.load");
  load->setAlignment(Align(1));
  Value *loaded = load;
  loaded = bswap(loaded);
  if (resultType != integerType)
    loaded = builder_.CreateZExtOrTrunc(loaded, resultType);
  return loaded;
}

void FunctionEmitter::traceAccess(Value *address, u32 width, bool write) {
  static const bool enabled =
      dolrecomp_codegen_switch_is("DOLRECOMP_MEM_TRACE", "1");
  if (!enabled || modern_runtime_)
    return;
  Type *i32 = Type::getInt32Ty(context_);
  FunctionCallee trace = module_.getOrInsertFunction(
      "fz_trace", FunctionType::get(Type::getVoidTy(context_), {i32, i32}, false));
  builder_.CreateCall(trace,
                      {address, builder_.getInt32(width | (write ? 0x100u : 0u))});
}

void FunctionEmitter::trackMem2(Value *offset, u32 width, bool write) {
  static const bool enabled =
      dolrecomp_codegen_switch_is("DOLRECOMP_MEM2_TRACK", "1");
  if (!enabled || modern_runtime_)
    return;
  Type *i8 = Type::getInt8Ty(context_);
  Type *i32 = Type::getInt32Ty(context_);
  Type *pointer = PointerType::getUnqual(context_);
  Value *controlAt = builder_.CreateInBoundsGEP(
      i8, mem2_, builder_.CreateSub(mem2_size_, builder_.getInt32(8)));
  LoadInst *control = builder_.CreateLoad(pointer, controlAt, "mem2.track");
  control->setAlignment(Align(1));
  BasicBlock *record = BasicBlock::Create(context_, "mem2_track", function_);
  BasicBlock *done =
      BasicBlock::Create(context_, "mem2_track_done", function_);
  builder_.CreateCondBr(builder_.CreateIsNotNull(control), record, done,
                        MDBuilder(context_).createBranchWeights(1, 2000));
  // In line: the record's two bitmaps, a bit a byte of the window's first 16 KiB, written at +24 and read at
  // +24 + 2048. An access inside one byte of a bitmap whose bytes are all written (a store), or all written or
  // read (a load), is recorded already: no call. Anything else calls the record's function.
  builder_.SetInsertPoint(record);
  Type *i64 = Type::getInt64Ty(context_);
  BasicBlock *bits = BasicBlock::Create(context_, "mem2_track_bits", function_);
  BasicBlock *slow = BasicBlock::Create(context_, "mem2_track_call", function_);
  Value *shift = builder_.CreateAnd(offset, builder_.getInt32(7));
  Value *fits = builder_.CreateAnd(
      builder_.CreateICmpULE(builder_.CreateAdd(shift, builder_.getInt32(width)),
                             builder_.getInt32(8)),
      builder_.CreateICmpULT(offset, builder_.getInt32(0x4000)));
  builder_.CreateCondBr(fits, bits, slow);
  builder_.SetInsertPoint(bits);
  Value *index = builder_.CreateZExt(builder_.CreateLShr(offset, builder_.getInt32(3)), i64);
  Value *mask = builder_.CreateShl(builder_.getInt32((1u << width) - 1u), shift);
  Value *written = builder_.CreateZExt(
      builder_.CreateLoad(i8, builder_.CreateInBoundsGEP(
                                  i8, control, builder_.CreateAdd(index, builder_.getInt64(24)))),
      i32);
  Value *seen = written;
  if (!write)
    seen = builder_.CreateOr(
        seen, builder_.CreateZExt(
                  builder_.CreateLoad(i8, builder_.CreateInBoundsGEP(
                                              i8, control,
                                              builder_.CreateAdd(index, builder_.getInt64(24 + 2048)))),
                  i32));
  Value *need = builder_.CreateAnd(mask, builder_.CreateNot(seen));
  builder_.CreateCondBr(builder_.CreateIsNull(need), done, slow,
                        MDBuilder(context_).createBranchWeights(2000, 1));
  builder_.SetInsertPoint(slow);
  Value *function = builder_.CreateLoad(pointer, control, "mem2.track.fn");
  auto *functionType = FunctionType::get(Type::getVoidTy(context_),
                                         {pointer, pointer, i32, i32}, false);
  CallInst *call = builder_.CreateCall(
      functionType, function,
      {control, mem2_, offset,
       builder_.getInt32(width | (write ? 0x100u : 0u))});
  tagRuntimeCall(call, RegisterWrites::None);
  builder_.CreateBr(done);
  builder_.SetInsertPoint(done);
}

Value *FunctionEmitter::emitGuestLoad(const DolIRInstruction &instruction,
                                      Value *address, Type *resultType,
                                      u32 width, bool sign) {
  traceAccess(address, width, false);
  countAccess(address, false);
  Value *directOffset = nullptr;
  if (Value *pointer =
          provenMemoryPointer(instruction, address, width, &directOffset)) {
    Value *loaded = endianLoad(pointer, resultType, width);
    if (sign && width * 8u < resultType->getIntegerBitWidth()) {
      Value *narrow =
          builder_.CreateTrunc(loaded, IntegerType::get(context_, width * 8u));
      return builder_.CreateSExt(narrow, resultType);
    }
    return loaded;
  }
  if (!lazy_fprf_)
    materializeFPRF();
  const Tracking tracking = saveTracking();
  Value *normalized = normalizeAddress(address);
  Value *mem1 = rangeCheck(normalized, GC_RAM_BASE, ram_size_, width);
  BasicBlock *mem1Block = BasicBlock::Create(context_, "load_mem1", function_);
  BasicBlock *checkMem2 =
      BasicBlock::Create(context_, "load_check_mem2", function_);
  BasicBlock *mem2Block = BasicBlock::Create(context_, "load_mem2", function_);
  BasicBlock *slowBlock = BasicBlock::Create(context_, "load_slow", function_);
  BasicBlock *join = BasicBlock::Create(context_, "load_join", function_);
  builder_.CreateCondBr(mem1, mem1Block, checkMem2,
                        MDBuilder(context_).createBranchWeights(2000, 1));

  builder_.SetInsertPoint(mem1Block);
  Value *mem1Ptr = builder_.CreateGEP(
      Type::getInt8Ty(context_), ram_biased_,
      builder_.CreateZExt(normalized, Type::getInt64Ty(context_)));
  Value *mem1Value = endianLoad(mem1Ptr, resultType, width);
  builder_.CreateBr(join);

  builder_.SetInsertPoint(checkMem2);
  Value *inMem2 = builder_.CreateAnd(
      builder_.CreateIsNotNull(mem2_),
      rangeCheck(normalized, mem2Base(), mem2_size_, width));
  builder_.CreateCondBr(inMem2, mem2Block, slowBlock,
                        MDBuilder(context_).createBranchWeights(2000, 1));

  builder_.SetInsertPoint(mem2Block);
  Value *mem2Offset =
      builder_.CreateSub(normalized, builder_.getInt32(mem2Base()));
  trackMem2(mem2Offset, width, false);
  Value *mem2Ptr =
      builder_.CreateInBoundsGEP(Type::getInt8Ty(context_), mem2_, mem2Offset);
  Value *mem2Value = endianLoad(mem2Ptr, resultType, width);
  BasicBlock *mem2End = builder_.GetInsertBlock();
  builder_.CreateBr(join);

  builder_.SetInsertPoint(slowBlock);
  // DOLRECOMP_UNMAPPED_READS=<first>-<last> (hex guest addresses): a title's
  // statement that no translation ever maps this range. The host then answers
  // a read there with zero and nothing else the guest can see (Dolphin
  // without full MMU emulation: no exception, no register), so the read is
  // answered here without the call. Reads only; a store still goes to the
  // host.
  BasicBlock *unmappedBlock = nullptr;
  u32 unmappedFirst = 0, unmappedLast = 0;
  if (unmappedReads(&unmappedFirst, &unmappedLast)) {
    unmappedBlock = BasicBlock::Create(context_, "load_unmapped", function_);
    BasicBlock *hostBlock =
        BasicBlock::Create(context_, "load_host", function_);
    Value *inside = builder_.CreateICmpULE(
        builder_.CreateSub(address, builder_.getInt32(unmappedFirst)),
        builder_.getInt32(unmappedLast - unmappedFirst - (width - 1u)));
    builder_.CreateCondBr(inside, unmappedBlock, hostBlock);
    builder_.SetInsertPoint(unmappedBlock);
    builder_.CreateBr(join);
    builder_.SetInsertPoint(hostBlock);
  }
  Value *slow64 = externalRead(address, width);
  Value *slowValue = builder_.CreateZExtOrTrunc(slow64, resultType);
  BasicBlock *slowEnd = builder_.GetInsertBlock();
  builder_.CreateBr(join);

  builder_.SetInsertPoint(join);
  if (lazy_fprf_ || sync_changed_)
    restoreTracking(tracking);
  PHINode *phi = builder_.CreatePHI(resultType, unmappedBlock ? 4 : 3);
  phi->addIncoming(mem1Value, mem1Block);
  phi->addIncoming(mem2Value, mem2End);
  phi->addIncoming(slowValue, slowEnd);
  if (unmappedBlock)
    phi->addIncoming(Constant::getNullValue(resultType), unmappedBlock);
  if (sign && width * 8u < resultType->getIntegerBitWidth()) {
    Value *narrow =
        builder_.CreateTrunc(phi, IntegerType::get(context_, width * 8u));
    return builder_.CreateSExt(narrow, resultType);
  }
  return phi;
}

void FunctionEmitter::clearReservation(Value *address) {
  if (dolrecomp_codegen_switch_is("DOLRECOMP_RESERVATION", "stwcx-only"))
    return;
  Value *valid = builder_.CreateLoad(Type::getInt1Ty(context_),
                                     state_[DOLIR_STATE_RESERVE_VALID]);
  Value *reserved = builder_.CreateLoad(Type::getInt32Ty(context_),
                                        state_[DOLIR_STATE_RESERVE_ADDR]);
  Value *differentLine = builder_.CreateICmpNE(
      builder_.CreateAnd(builder_.CreateXor(reserved, address),
                         builder_.getInt32(~31u)),
      builder_.getInt32(0));
  builder_.CreateStore(builder_.CreateAnd(valid, differentLine),
                       state_[DOLIR_STATE_RESERVE_VALID]);
  markWritten(DOLIR_STATE_RESERVE_VALID);
}

void FunctionEmitter::journal(Value *offset, u32 width) {
  if (!write_journal_)
    return;
  Type *ptr = PointerType::getUnqual(context_);
  GlobalVariable *journal = cast<GlobalVariable>(
      module_.getOrInsertGlobal("g_mem_write_journal", ptr));
  GlobalVariable *user = cast<GlobalVariable>(
      module_.getOrInsertGlobal("g_mem_write_journal_user", ptr));
  Value *fn = builder_.CreateLoad(ptr, journal);
  BasicBlock *call = BasicBlock::Create(context_, "journal", function_);
  BasicBlock *done = BasicBlock::Create(context_, "journal_done", function_);
  builder_.CreateCondBr(builder_.CreateIsNotNull(fn), call, done,
                        MDBuilder(context_).createBranchWeights(2000, 1));
  builder_.SetInsertPoint(call);
  auto *functionType = FunctionType::get(
      Type::getVoidTy(context_),
      {Type::getInt32Ty(context_), Type::getInt32Ty(context_), ptr}, false);
  builder_.CreateCall(
      functionType, fn,
      {offset, builder_.getInt32(width), builder_.CreateLoad(ptr, user)});
  builder_.CreateBr(done);
  builder_.SetInsertPoint(done);
}

void FunctionEmitter::endianStore(Value *pointer, Value *value, u32 width) {
  Type *integerType = IntegerType::get(context_, width * 8u);
  Value *narrowed = value;
  if (value->getType() != integerType)
    narrowed = builder_.CreateZExtOrTrunc(value, integerType);
  StoreInst *store = builder_.CreateStore(bswap(narrowed), pointer);
  store->setAlignment(Align(1));
}

void FunctionEmitter::emitGuestStore(const DolIRInstruction &instruction,
                                     Value *address, Value *value, u32 width) {
  traceAccess(address, width, true);
  countAccess(address, true);
  Value *directOffset = nullptr;
  if (Value *pointer =
          provenMemoryPointer(instruction, address, width, &directOffset)) {
    clearReservation(address);
    journal(directOffset, width);
    endianStore(pointer, value, width);
    return;
  }
  if (!lazy_fprf_)
    materializeFPRF();
  const Tracking tracking = saveTracking();
  clearReservation(address);
  Value *normalized = normalizeAddress(address);
  BasicBlock *mem1Block = BasicBlock::Create(context_, "store_mem1", function_);
  BasicBlock *checkMem2 =
      BasicBlock::Create(context_, "store_check_mem2", function_);
  BasicBlock *mem2Block = BasicBlock::Create(context_, "store_mem2", function_);
  BasicBlock *slowBlock = BasicBlock::Create(context_, "store_slow", function_);
  BasicBlock *join = BasicBlock::Create(context_, "store_join", function_);
  builder_.CreateCondBr(rangeCheck(normalized, GC_RAM_BASE, ram_size_, width),
                        mem1Block, checkMem2,
                        MDBuilder(context_).createBranchWeights(2000, 1));

  builder_.SetInsertPoint(mem1Block);
  if (write_journal_)
    journal(builder_.CreateSub(normalized, builder_.getInt32(GC_RAM_BASE)),
            width);
  Value *mem1Ptr = builder_.CreateGEP(
      Type::getInt8Ty(context_), ram_biased_,
      builder_.CreateZExt(normalized, Type::getInt64Ty(context_)));
  endianStore(mem1Ptr, value, width);
  builder_.CreateBr(join);

  builder_.SetInsertPoint(checkMem2);
  Value *inMem2 = builder_.CreateAnd(
      builder_.CreateIsNotNull(mem2_),
      rangeCheck(normalized, mem2Base(), mem2_size_, width));
  builder_.CreateCondBr(inMem2, mem2Block, slowBlock,
                        MDBuilder(context_).createBranchWeights(2000, 1));

  builder_.SetInsertPoint(mem2Block);
  Value *mem2Offset =
      builder_.CreateSub(normalized, builder_.getInt32(mem2Base()));
  trackMem2(mem2Offset, width, true);
  Value *mem2Ptr =
      builder_.CreateInBoundsGEP(Type::getInt8Ty(context_), mem2_, mem2Offset);
  endianStore(mem2Ptr, value, width);
  builder_.CreateBr(join);

  builder_.SetInsertPoint(slowBlock);
  if (inline_gather_pipe_)
    emitGatherPipeStore(address, value, width);
  else
    externalWrite(address, value, width);
  builder_.CreateBr(join);
  builder_.SetInsertPoint(join);
  if (lazy_fprf_ || sync_changed_)
    restoreTracking(tracking);
}

// DOLRECOMP_GATHER_PIPE=inline: a store that is in neither memory window and
// whose address is in the write-gather pipe's page (0xCC008000) appends the
// big-endian value to the host's pipe buffer and advances the host's pipe
// pointer here, the way Dolphin's JITs do, instead of calling the hardware
// write hook. The host is called only when 32 bytes or more are in the
// buffer; that call is where it copies the burst to the FIFO, brings its
// clock up to date (the cycles run so far are settled first) and sets the
// cycle budget again.
//
// The module defines dolrecomp_gather_pipe = {address of the host's pipe
// pointer, start of the pipe buffer, flush function}; a host hands the three
// over through the module's staticrecomp_set_gather_pipe. While the first is
// null (a host that knows nothing of this, or one that wants to see every
// write) the store takes the hook as before.
//
// The flush function must not change guest registers, like the other calls
// generated code makes in place.
void FunctionEmitter::emitGatherPipeStore(Value *address, Value *value,
                                          u32 width) {
  Type *i8 = Type::getInt8Ty(context_);
  Type *i64 = Type::getInt64Ty(context_);
  Type *pointer = PointerType::getUnqual(context_);
  StructType *pipeType = StructType::get(context_, {pointer, pointer, pointer});
  auto *pipe = cast<GlobalVariable>(
      module_.getOrInsertGlobal("dolrecomp_gather_pipe", pipeType));
  pipe->setVisibility(GlobalValue::HiddenVisibility);
  pipe->setDSOLocal(true);

  BasicBlock *pagePipe = BasicBlock::Create(context_, "pipe_page", function_);
  BasicBlock *write = BasicBlock::Create(context_, "pipe_write", function_);
  BasicBlock *flush = BasicBlock::Create(context_, "pipe_flush", function_);
  BasicBlock *hook = BasicBlock::Create(context_, "pipe_hook", function_);
  BasicBlock *done = BasicBlock::Create(context_, "pipe_done", function_);
  MDBuilder weights(context_);

  Value *inPage = builder_.CreateICmpEQ(
      builder_.CreateAnd(address, builder_.getInt32(0xFFFFF000u)),
      builder_.getInt32(0xCC008000u));
  builder_.CreateCondBr(inPage, pagePipe, hook);

  builder_.SetInsertPoint(pagePipe);
  Value *location = builder_.CreateLoad(
      pointer, builder_.CreateStructGEP(pipeType, pipe, 0), "pipe.location");
  builder_.CreateCondBr(builder_.CreateIsNotNull(location), write, hook,
                        weights.createBranchWeights(2000, 1));

  builder_.SetInsertPoint(write);
  Value *position = builder_.CreateLoad(pointer, location, "pipe.position");
  endianStore(position, value, width);
  Value *next = builder_.CreateGEP(i8, position, builder_.getInt64(width));
  builder_.CreateStore(next, location);
  Value *start = builder_.CreateLoad(
      pointer, builder_.CreateStructGEP(pipeType, pipe, 1), "pipe.start");
  Value *count = builder_.CreateSub(builder_.CreatePtrToInt(next, i64),
                                    builder_.CreatePtrToInt(start, i64));
  builder_.CreateCondBr(builder_.CreateICmpUGE(count, builder_.getInt64(32)),
                        flush, done, weights.createBranchWeights(1, 8));

  builder_.SetInsertPoint(flush);
  settleCycles();
  Value *function = builder_.CreateLoad(
      pointer, builder_.CreateStructGEP(pipeType, pipe, 2), "pipe.flush");
  tagRuntimeCall(
      builder_.CreateCall(
          FunctionType::get(Type::getVoidTy(context_), {pointer}, false),
          function, {ctx_}),
      RegisterWrites::None);
  builder_.CreateBr(done);

  builder_.SetInsertPoint(hook);
  externalWrite(address, value, width);
  builder_.CreateBr(done);

  builder_.SetInsertPoint(done);
}

} // namespace dolllvm
