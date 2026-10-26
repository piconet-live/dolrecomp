#include "backend/llvm/emitter.h"

#include <vector>

#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include "cpu/cpu.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

namespace dolllvm {

using namespace llvm;

AllocaInst *FunctionEmitter::temporary(Type *valueType, StringRef name) {
  IRBuilder<> allocations(entry_->getTerminator());
  return allocations.CreateAlloca(valueType, nullptr, name);
}

Value *FunctionEmitter::stateValue(DolIRStateSlot slot) {
  if (slot == DOLIR_STATE_FPSCR)
    materializeFPRF();
  if (known_state_[slot])
    return known_state_[slot];
  return builder_.CreateLoad(type(dolir_state_type(slot)), state_[slot]);
}

void FunctionEmitter::syncState(DolIRStateSlot slot) {
  storeContext(slot, stateValue(slot));
}

void FunctionEmitter::reloadState(DolIRStateSlot slot) {
  known_state_[slot] = nullptr;
  unsynced_.reset(slot);
  if (slot == DOLIR_STATE_FPSCR)
    clearPendingFPRF();
  if (slotInMemory(slot))
    return;
  builder_.CreateStore(loadContext(slot), state_[slot]);
}

void FunctionEmitter::reloadModernState() {
  Type *i64 = Type::getInt64Ty(context_);
  ArrayType *valuesType = ArrayType::get(i64, DOLIR_STATE_COUNT);
  ArrayType *maskType =
      ArrayType::get(i64, DOLIR_STATE_MASK_WORDS);
  const std::string name = source_.name + std::string(".used");
  GlobalVariable *mask = module_.getGlobalVariable(name, true);
  if (!mask) {
    SmallVector<Constant *, DOLIR_STATE_MASK_WORDS> words;
    for (u32 word = 0; word < DOLIR_STATE_MASK_WORDS; word++) {
      u64 bits = 0;
      for (u32 bit = 0; bit < 64; bit++) {
        const u32 slot = word * 64u + bit;
        if (slot < DOLIR_STATE_COUNT && used_[slot])
          bits |= 1ull << bit;
      }
      words.push_back(builder_.getInt64(bits));
    }
    mask = new GlobalVariable(module_, maskType, true,
                              GlobalValue::PrivateLinkage,
                              ConstantArray::get(maskType, words), name);
  }
  Type *pointer = PointerType::getUnqual(context_);
  FunctionCallee reload = module_.getOrInsertFunction(
      "moderngekko_reload_state",
      FunctionType::get(Type::getVoidTy(context_),
                        {pointer, pointer, pointer}, false));
  if (auto *function = dyn_cast<Function>(reload.getCallee())) {
    function->addFnAttr(Attribute::Cold);
    function->addFnAttr(Attribute::NoUnwind);
  }
  Value *values = builder_.CreateStructGEP(chainType(), chain_, 8);
  CallInst *call = builder_.CreateCall(reload, {state_interface_, values, mask});
  call->addFnAttr(Attribute::Cold);
  call->addFnAttr(Attribute::NoUnwind);
  for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
    if (!used_[slot])
      continue;
    auto stateSlot = static_cast<DolIRStateSlot>(slot);
    if (slotInMemory(stateSlot))
      continue;
    Value *value = nullptr;
    if (stateSlot == DOLIR_STATE_PC || stateSlot == DOLIR_STATE_TIMEBASE ||
        stateSlot == DOLIR_STATE_PROGRAM_EXCEPTION ||
        stateSlot == DOLIR_STATE_DOWNCOUNT) {
      value = loadContext(stateSlot);
    } else {
      value = builder_.CreateLoad(
          i64, builder_.CreateInBoundsGEP(
                   valuesType, values,
                   {builder_.getInt64(0), builder_.getInt64(slot)}));
      Type *target = type(dolir_state_type(stateSlot));
      value = target->isDoubleTy() ? builder_.CreateBitCast(value, target)
                                  : builder_.CreateZExtOrTrunc(value, target);
    }
    builder_.CreateStore(value, state_[slot]);
  }
}

void FunctionEmitter::reloadUsedState() {
  invalidateFPRepresentations();
  fp_available_checked_ = false;
  clearPendingFPRF();
  known_state_.fill(nullptr);
  psq_direct_proven_ = false;
  psq_indexed_proven_ = false;
  if (modern_runtime_) {
    reloadModernState();
    return;
  }
  for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
    if (used_[slot])
      reloadState(static_cast<DolIRStateSlot>(slot));
  }
}

void FunctionEmitter::continueAfterRuntimeBoundary(StringRef prefix) {
  Value *exception =
      loadOffset(Type::getInt32Ty(context_), offsetof(CPUState, exception));
  BasicBlock *resume =
      BasicBlock::Create(context_, prefix + "_resume", function_);
  BasicBlock *failed =
      BasicBlock::Create(context_, prefix + "_exit", function_);
  builder_.CreateCondBr(builder_.CreateICmpEQ(exception, builder_.getInt32(0)),
                        resume, failed);
  builder_.SetInsertPoint(failed);
  returnFromBody();
  builder_.SetInsertPoint(resume);
  reloadUsedState();
}

// The host's rounding mode and flush-to-zero state follow FPSCR[RN] and
// FPSCR[NI]: when those bits differ between `before` and `after` (or `before`
// is not known), MXCSR is loaded from a table of the eight combinations, as
// Dolphin's Jit64 does. The runtime helper this replaces reads MXCSR, edits
// the same bits and writes it back.
void FunctionEmitter::emitHostFPMode(Value *before, Value *after) {
  Type *i32 = Type::getInt32Ty(context_);
  ArrayType *tableType = ArrayType::get(i32, 8);
  GlobalVariable *table = module_.getGlobalVariable("dolrecomp.mxcsr", true);
  if (!table) {
    // index: bit 2 = NI (flush to zero, denormals are zero), bits 1:0 = RN
    // (nearest, toward zero, up, down); all exceptions masked.
    static const u32 rounding[4] = {0u << 13, 3u << 13, 2u << 13, 1u << 13};
    std::vector<Constant *> values;
    for (u32 index = 0; index < 8; index++)
      values.push_back(builder_.getInt32(0x1F80u | rounding[index & 3u] |
                                         ((index & 4u) ? 0x8040u : 0u)));
    table = new GlobalVariable(module_, tableType, true,
                               GlobalValue::PrivateLinkage,
                               ConstantArray::get(tableType, values),
                               "dolrecomp.mxcsr");
  }
  Value *control = builder_.CreateAnd(after, builder_.getInt32(7));
  BasicBlock *update = BasicBlock::Create(context_, "fp_mode_update", function_);
  BasicBlock *done = BasicBlock::Create(context_, "fp_mode_done", function_);
  Value *changed =
      before ? builder_.CreateICmpNE(
                   builder_.CreateAnd(builder_.CreateXor(before, after),
                                      builder_.getInt32(7)),
                   builder_.getInt32(0))
             : static_cast<Value *>(builder_.getTrue());
  builder_.CreateCondBr(changed, update, done);
  builder_.SetInsertPoint(update);
  countHere(current_pc_, 2, 0);
  builder_.CreateCall(
      Intrinsic::getDeclaration(&module_, Intrinsic::x86_sse_ldmxcsr),
      {builder_.CreateInBoundsGEP(
          tableType, table,
          {builder_.getInt64(0),
           builder_.CreateZExt(control, Type::getInt64Ty(context_))})});
  builder_.CreateBr(done);
  builder_.SetInsertPoint(done);
}

void FunctionEmitter::emitFPSCRUpdated() {
  if (inline_fpscr_) {
    // ppc_fpscr_updated: VX and FEX are summaries of the bits just written.
    Type *i32 = Type::getInt32Ty(context_);
    Value *fpscr = builder_.CreateLoad(i32, state_[DOLIR_STATE_FPSCR]);
    Value *anyInvalid = builder_.CreateICmpNE(
        builder_.CreateAnd(fpscr, builder_.getInt32(0x01F80700u)),
        builder_.getInt32(0));
    fpscr = builder_.CreateOr(
        builder_.CreateAnd(fpscr, builder_.getInt32(~0x20000000u)),
        builder_.CreateSelect(anyInvalid, builder_.getInt32(0x20000000u),
                              builder_.getInt32(0)));
    Value *enabled = builder_.CreateICmpNE(
        builder_.CreateAnd(
            builder_.CreateAnd(builder_.CreateLShr(fpscr, 22), fpscr),
            builder_.getInt32(0xF8u)),
        builder_.getInt32(0));
    fpscr = builder_.CreateOr(
        builder_.CreateAnd(fpscr, builder_.getInt32(~0x40000000u)),
        builder_.CreateSelect(enabled, builder_.getInt32(0x40000000u),
                              builder_.getInt32(0)));
    builder_.CreateStore(fpscr, state_[DOLIR_STATE_FPSCR]);
    known_state_[DOLIR_STATE_FPSCR] = nullptr;
    emitHostFPMode(fpscr_before_write_, fpscr);
    fpscr_before_write_ = nullptr;
    fp_denormal_safe_.fill(false);
    return;
  }
  syncState(DOLIR_STATE_FPSCR);
  auto callee = module_.getOrInsertFunction(
      "ppc_fpscr_control_updated",
      FunctionType::get(Type::getVoidTy(context_),
                        {PointerType::getUnqual(context_)}, false));
  builder_.CreateCall(callee, {ctx_});
  reloadState(DOLIR_STATE_FPSCR);
  fp_denormal_safe_.fill(false);
}

void FunctionEmitter::emitFPSCRBit(u64 descriptor) {
  const u32 bitMask = 0x80000000u >> (descriptor & 0x1Fu);
  if (inline_fpscr_ && (bitMask & 7u)) {
    // RN or NI: no exception bit and no summary bit changes.
    Value *before = stateValue(DOLIR_STATE_FPSCR);
    Value *after = ((descriptor >> 8) & 1u)
                       ? builder_.CreateOr(before, builder_.getInt32(bitMask))
                       : builder_.CreateAnd(before,
                                            builder_.getInt32(~bitMask));
    builder_.CreateStore(after, state_[DOLIR_STATE_FPSCR]);
    known_state_[DOLIR_STATE_FPSCR] = nullptr;
    emitHostFPMode(before, after);
    fp_denormal_safe_.fill(false);
    return;
  }
  syncState(DOLIR_STATE_FPSCR);
  const char *name =
      ((descriptor >> 8) & 1u) ? "ppc_mtfsb1_op" : "ppc_mtfsb0_op";
  auto callee = module_.getOrInsertFunction(
      name, FunctionType::get(
                Type::getVoidTy(context_),
                {PointerType::getUnqual(context_), Type::getInt8Ty(context_)},
                false));
  builder_.CreateCall(callee, {ctx_, builder_.getInt8(descriptor & 0xFFu)});
  reloadState(DOLIR_STATE_FPSCR);
  fp_denormal_safe_.fill(false);
}

void FunctionEmitter::emitProgramException(const DolIRInstruction &inst) {
  BasicBlock *taken = BasicBlock::Create(context_, "trap_taken", function_);
  BasicBlock *resume = BasicBlock::Create(context_, "trap_resume", function_);
  builder_.CreateCondBr(operand(inst, 0), taken, resume);
  builder_.SetInsertPoint(taken);
  if (modern_runtime_) {
    Value *exceptions = builder_.CreateLoad(Type::getInt32Ty(context_),
                                            state_[DOLIR_STATE_EXCEPTION]);
    Value *updated =
        builder_.CreateOr(exceptions, builder_.getInt32(0x80));
    builder_.CreateStore(updated, state_[DOLIR_STATE_EXCEPTION]);
    noteStateWrite(DOLIR_STATE_EXCEPTION, updated);
    Value *cause = builder_.getInt32(inst.immediate);
    builder_.CreateStore(cause, state_[DOLIR_STATE_SRR1]);
    noteStateWrite(DOLIR_STATE_SRR1, cause);
    sideExit(inst.guest_pc, 1);
    builder_.SetInsertPoint(resume);
    return;
  }
  materialize(inst.guest_pc);
  auto callee = module_.getOrInsertFunction(
      "ppc_program_exception",
      FunctionType::get(Type::getVoidTy(context_),
                        {PointerType::getUnqual(context_),
                         Type::getInt32Ty(context_),
                         Type::getInt32Ty(context_)},
                        false));
  builder_.CreateCall(callee, {ctx_, builder_.getInt32(inst.immediate),
                               builder_.getInt32(inst.guest_pc)});
  returnFromBody();
  builder_.SetInsertPoint(resume);
}

Value *FunctionEmitter::emitSPRRead(const DolIRInstruction &inst) {
  materialize(inst.guest_pc);
  auto callee = module_.getOrInsertFunction(
      "ppc_mfspr", FunctionType::get(Type::getInt32Ty(context_),
                                     {PointerType::getUnqual(context_),
                                      Type::getInt16Ty(context_),
                                      Type::getInt32Ty(context_)},
                                     false));
  Value *result =
      builder_.CreateCall(callee, {ctx_, builder_.getInt16(inst.immediate),
                                   builder_.getInt32(inst.guest_pc)});
  continueAfterRuntimeBoundary("mfspr");
  return result;
}

void FunctionEmitter::emitSPRWrite(const DolIRInstruction &inst) {
  materialize(inst.guest_pc);
  auto callee = module_.getOrInsertFunction(
      "ppc_mtspr",
      FunctionType::get(Type::getVoidTy(context_),
                        {PointerType::getUnqual(context_),
                         Type::getInt16Ty(context_), Type::getInt32Ty(context_),
                         Type::getInt32Ty(context_)},
                        false));
  builder_.CreateCall(callee,
                      {ctx_, builder_.getInt16(inst.immediate),
                       operand(inst, 0), builder_.getInt32(inst.guest_pc)});
  continueAfterRuntimeBoundary("mtspr");
}

void FunctionEmitter::emitLSWX(const DolIRInstruction &inst) {
  materialize(inst.guest_pc);
  auto callee = module_.getOrInsertFunction(
      "ppc_lswx",
      FunctionType::get(Type::getVoidTy(context_),
                        {PointerType::getUnqual(context_),
                         Type::getInt8Ty(context_), Type::getInt8Ty(context_),
                         Type::getInt8Ty(context_), Type::getInt32Ty(context_)},
                        false));
  builder_.CreateCall(callee, {ctx_, builder_.getInt8(inst.immediate & 0xFFu),
                               builder_.getInt8((inst.immediate >> 8) & 0xFFu),
                               builder_.getInt8((inst.immediate >> 16) & 0xFFu),
                               builder_.getInt32(inst.guest_pc)});
  continueAfterRuntimeBoundary("lswx");
}

Value *FunctionEmitter::emitRuntimeBoundary(const DolIRInstruction &inst) {
  if (modern_runtime_ && inst.aux == DOLIR_HELPER_CACHE_CONTROL) {
    emitCacheControl(inst);
    return nullptr;
  }

  materialize(inst.guest_pc);
  Type *ptr = PointerType::getUnqual(context_);
  Value *result = nullptr;
  StringRef prefix;
  if (inst.aux == DOLIR_HELPER_DCBZ_L) {
    prefix = "dcbz_l";
    auto callee = module_.getOrInsertFunction(
        "ppc_dcbz_l", FunctionType::get(Type::getVoidTy(context_),
                                        {ptr, Type::getInt32Ty(context_),
                                         Type::getInt32Ty(context_)},
                                        false));
    builder_.CreateCall(
        callee, {ctx_, operand(inst, 0), builder_.getInt32(inst.guest_pc)});
  } else if (inst.aux == DOLIR_HELPER_ECIWX) {
    prefix = "eciwx";
    auto callee = module_.getOrInsertFunction(
        "ppc_eciwx", FunctionType::get(Type::getInt32Ty(context_),
                                       {ptr, Type::getInt32Ty(context_),
                                        Type::getInt32Ty(context_)},
                                       false));
    result = builder_.CreateCall(
        callee, {ctx_, operand(inst, 0), builder_.getInt32(inst.guest_pc)});
  } else if (inst.aux == DOLIR_HELPER_ECOWX) {
    prefix = "ecowx";
    auto callee = module_.getOrInsertFunction(
        "ppc_ecowx", FunctionType::get(Type::getVoidTy(context_),
                                       {ptr, Type::getInt32Ty(context_),
                                        Type::getInt32Ty(context_),
                                        Type::getInt32Ty(context_)},
                                       false));
    builder_.CreateCall(callee, {ctx_, operand(inst, 0), operand(inst, 1),
                                 builder_.getInt32(inst.guest_pc)});
  } else if (inst.aux == DOLIR_HELPER_TLBIE) {
    prefix = "tlbie";
    auto callee = module_.getOrInsertFunction(
        "ppc_tlbie", FunctionType::get(Type::getVoidTy(context_),
                                       {ptr, Type::getInt32Ty(context_),
                                        Type::getInt32Ty(context_)},
                                       false));
    builder_.CreateCall(
        callee, {ctx_, operand(inst, 0), builder_.getInt32(inst.guest_pc)});
  } else {
    prefix = "cache";
    auto callee = module_.getOrInsertFunction(
        "ppc_cache_control", FunctionType::get(Type::getVoidTy(context_),
                                               {ptr, Type::getInt8Ty(context_),
                                                Type::getInt32Ty(context_),
                                                Type::getInt32Ty(context_)},
                                               false));
    builder_.CreateCall(callee,
                        {ctx_, builder_.getInt8(inst.immediate),
                         operand(inst, 0), builder_.getInt32(inst.guest_pc)});
  }
  continueAfterRuntimeBoundary(prefix);
  return result;
}

} // namespace dolllvm
