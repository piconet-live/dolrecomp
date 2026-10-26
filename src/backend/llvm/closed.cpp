#include "backend/llvm/emitter.h"
#include "backend/codegen_switches.h"
#include "cpu/cpu.h"

#include <cstdio>
#include <cstdlib>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>

// Closed bodies (DOLRECOMP_CLOSED_ROOTS); the protocol is described at
// FunctionEmitter::closed_ in emitter.h.

namespace dolllvm {

using namespace llvm;

namespace {
// Bit 63 of the guard word a closed body returns: the body escaped, CPUState
// is complete and the PC says where execution continues.
constexpr u64 kEscaped = u64{1} << 63;
} // namespace

bool FunctionEmitter::closedCapable() const {
  return abi_range_ &&
         (abi_range_->abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED) != 0 &&
         !modern_runtime_ && !native_abi_ &&
         register_counters_ && split_entry_loads_ && sync_changed_ &&
         source_.block_count != 0 &&
         source_.blocks[0].guest_address == source_.guest_start;
}

bool FunctionEmitter::wantsClosedBody() const {
  return closedCapable() && !imported_;
}

// DOLRECOMP_INLINE_CLOSED=1: a callee imported into this chunk's module
// (DOLRECOMP_INLINE_CALLEES) that is in the closed set is given as its closed
// body, which a closed caller's call with the closed protocol can then be
// inlined from. Without the switch an imported callee is given as its
// ordinary body only, which no closed body calls.
bool FunctionEmitter::wantsImportedClosedBody() const {
  static const bool enabled =
      dolrecomp_codegen_switch_is("DOLRECOMP_INLINE_CLOSED", "1");
  return enabled && imported_ && closedCapable();
}

// A direct call or branch to `target` can use the closed protocol.
bool FunctionEmitter::closedTarget(const DolLLVMFunctionRange *range,
                                   u32 target) const {
  return closed_ && range && target == range->start &&
         (range->abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED) != 0;
}

FunctionType *FunctionEmitter::closedFunctionType() {
  Type *i64 = Type::getInt64Ty(context_);
  return FunctionType::get(StructType::get(context_, {i64, i64}),
                           {PointerType::getUnqual(context_), i64, i64, i64},
                           false);
}

FunctionCallee FunctionEmitter::closedBody(u32 start) {
  char name[64];
  snprintf(name, sizeof(name), "func_%08X_closed", start);
  FunctionCallee callee =
      module_.getOrInsertFunction(symbolName(name), closedFunctionType());
  if (auto *function = dyn_cast<Function>(callee.getCallee())) {
    function->setCallingConv(bodyCallingConvention());
    function->setVisibility(GlobalValue::HiddenVisibility);
    function->setDSOLocal(true);
  }
  return callee;
}

Value *FunctionEmitter::closedEscaped(Value *result) {
  return builder_.CreateICmpSLT(builder_.CreateExtractValue(result, 1),
                                builder_.getInt64(0));
}

Value *FunctionEmitter::withEscape(Value *result) {
  return builder_.CreateInsertValue(
      result,
      builder_.CreateOr(builder_.CreateExtractValue(result, 1),
                        builder_.getInt64(kEscaped)),
      1);
}

void FunctionEmitter::passResultUp(Value *result) {
  if (!register_counters_) {
    builder_.CreateRetVoid();
    return;
  }
  builder_.CreateRet(closed_ ? withEscape(result) : result);
}

// In the ordinary body of a closed function, on the path of an entry at the
// first instruction and before any register is loaded: hand over to the
// closed body unless the host has a call hook installed (with one, a function
// may be intercepted, and intercepted functions must be left through an exit,
// so the closed bodies are not used at all).
bool FunctionEmitter::closedBuild() {
  static const bool set = [] {
    const char *roots = std::getenv("DOLRECOMP_CLOSED_ROOTS");
    return roots && *roots;
  }();
  return set;
}

GlobalVariable *FunctionEmitter::closedPermission() {
  char name[64];
  snprintf(name, sizeof(name), "dolrecomp_closed_ok_%08X",
           source_.guest_start);
  GlobalVariable *flag = module_.getGlobalVariable(symbolName(name));
  if (!flag)
    flag = new GlobalVariable(module_, Type::getInt8Ty(context_), false,
                              GlobalValue::ExternalLinkage,
                              imported_ ? nullptr : builder_.getInt8(2),
                              symbolName(name));
  flag->setDSOLocal(true);
  return flag;
}

void FunctionEmitter::emitClosedHandover(Value *hostCall) {
  BasicBlock *handover =
      BasicBlock::Create(context_, "closed_handover", function_);
  BasicBlock *returned =
      BasicBlock::Create(context_, "closed_returned", function_);
  BasicBlock *escaped =
      BasicBlock::Create(context_, "closed_escaped", function_);
  BasicBlock *ordinary =
      BasicBlock::Create(context_, "start_ordinary", function_);
  // Permitted by the host (1), or the host has said nothing (2) and has no
  // call hook installed.
  Value *permission =
      builder_.CreateLoad(Type::getInt8Ty(context_), closedPermission());
  Value *allowed = builder_.CreateOr(
      builder_.CreateICmpEQ(permission, builder_.getInt8(1)),
      builder_.CreateAnd(builder_.CreateICmpEQ(permission, builder_.getInt8(2)),
                         builder_.CreateIsNull(hostCall)));
  builder_.CreateCondBr(allowed, handover, ordinary,
                        MDBuilder(context_).createBranchWeights(2000, 1));

  builder_.SetInsertPoint(handover);
  CallInst *call = builder_.CreateCall(
      closedBody(source_.guest_start),
      {ctx_, control_pc_, initial_cycles_, initial_guard_});
  call->setCallingConv(bodyCallingConvention());
  builder_.CreateCondBr(closedEscaped(call), escaped, returned,
                        MDBuilder(context_).createBranchWeights(1, 2000));

  // A normal return: the closed body has checked that the guest returned to
  // the return address but has not stored it.
  builder_.SetInsertPoint(returned);
  storeContext(DOLIR_STATE_PC, return_pc_);
  builder_.CreateRet(call);

  // An escape: the PC is where the run loop continues. The flag is the
  // closed protocol's; ordinary callers do not know it.
  builder_.SetInsertPoint(escaped);
  builder_.CreateRet(builder_.CreateInsertValue(
      call,
      builder_.CreateAnd(builder_.CreateExtractValue(call, 1),
                         builder_.getInt64(~kEscaped)),
      1));

  builder_.SetInsertPoint(ordinary);
}

// blr and bctr in a closed body, the state already written back. A return to
// the caller's return address is the normal return. A bctr to an instruction
// of this function that starts a block (a jump table) continues there, with
// every register read back from CPUState first: the jump then looks like an
// ordinary body's resume entry, and no value has to be kept for it across the
// function. Anything else is an escape to that address.
void FunctionEmitter::emitClosedIndirect(Value *target, bool throughCTR) {
  BasicBlock *normal = BasicBlock::Create(context_, "closed_return", function_);
  BasicBlock *other = BasicBlock::Create(context_, "closed_jump", function_);
  builder_.CreateCondBr(builder_.CreateICmpEQ(target, return_pc_), normal,
                        other);
  builder_.SetInsertPoint(normal);
  builder_.CreateRet(counterResult());

  builder_.SetInsertPoint(other);
  if (!throughCTR) {
    storeContext(DOLIR_STATE_PC, target);
    builder_.CreateRet(withEscape(counterResult()));
    return;
  }
  if (!closed_jump_block_) {
    // One switch for the whole function.
    closed_jump_block_ =
        BasicBlock::Create(context_, "closed_jump_table", function_);
    BasicBlock *leave =
        BasicBlock::Create(context_, "closed_jump_leave", function_);
    IRBuilderBase::InsertPoint saved = builder_.saveIP();
    builder_.SetInsertPoint(closed_jump_block_);
    closed_jump_target_ =
        builder_.CreatePHI(Type::getInt32Ty(context_), 0, "jump_target");
    const Tracking tracking = saveTracking();
    reloadUsedState();
    restoreTracking(tracking);
    auto *dispatch = builder_.CreateSwitch(closed_jump_target_, leave);
    for (u32 i = 0; i < source_.block_count; i++) {
      if (!region_leaders_[i] ||
          source_.blocks[i].terminator.kind == DOLIR_TERM_FALLBACK)
        continue;
      dispatch->addCase(builder_.getInt32(source_.blocks[i].guest_address),
                        blocks_[i]);
    }
    builder_.SetInsertPoint(leave);
    storeContext(DOLIR_STATE_PC, closed_jump_target_);
    builder_.CreateRet(withEscape(counterResult()));
    builder_.restoreIP(saved);
  }
  closed_jump_target_->addIncoming(target, builder_.GetInsertBlock());
  builder_.CreateBr(closed_jump_block_);
}

} // namespace dolllvm
