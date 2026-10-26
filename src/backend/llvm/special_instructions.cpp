#include "backend/llvm/emitter.h"
#include "cpu/cpu.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

namespace dolllvm {

using namespace llvm;

void FunctionEmitter::emitStateWrite(const DolIRInstruction &inst) {
  auto slot = static_cast<DolIRStateSlot>(inst.aux);
  Value *value = operand(inst, 0);
  if (slot != DOLIR_STATE_MSR) {
    if (slot == DOLIR_STATE_FPSCR && inline_fpscr_)
      fpscr_before_write_ = builder_.CreateLoad(Type::getInt32Ty(context_),
                                                state_[inst.aux]);
    builder_.CreateStore(value, state_[inst.aux]);
    if (track_singles_ && slot >= DOLIR_STATE_FPR0 &&
        slot <= DOLIR_STATE_PS1_31) {
      const u32 reg = slot - (slot <= DOLIR_STATE_FPR31 ? DOLIR_STATE_FPR0
                                                        : DOLIR_STATE_PS1_0);
      const FPRepresentation rep = fp_rep_[reg];
      const bool exact = fp_exact_single_[reg];
      const bool safe = fp_denormal_safe_[reg];
      const bool canonical = fp_single_canonical_[reg];
      noteStateWrite(slot, value);
      noteTrackedLaneWrite(slot, value, rep, exact, safe, canonical);
      return;
    }
    noteStateWrite(slot, value);
    noteFloatLaneWrite(slot, value);
    return;
  }

  Value *old =
      builder_.CreateLoad(Type::getInt32Ty(context_), state_[inst.aux]);
  builder_.CreateStore(value, state_[inst.aux]);
  noteStateWrite(slot, value);
  // DOLRECOMP_CLOSED_DEFER: a closed body goes on; an interrupt that is
  // pending is delivered when control next reaches the run loop.
  if (closed_ && closed_defer_)
    return;
  Value *enabled = builder_.CreateAnd(builder_.CreateNot(old), value);
  enabled = builder_.CreateICmpNE(
      builder_.CreateAnd(enabled, builder_.getInt32(0x8000)),
      builder_.getInt32(0));
  BasicBlock *exit = BasicBlock::Create(context_, "msr_ee_exit", function_);
  BasicBlock *resume = BasicBlock::Create(context_, "msr_ee_resume", function_);
  builder_.CreateCondBr(enabled, exit, resume);
  builder_.SetInsertPoint(exit);
  sideExit(inst.guest_pc + 4u);
  builder_.SetInsertPoint(resume);
}

void FunctionEmitter::emitStoreConditional(const DolIRInstruction &inst) {
  materialize(inst.guest_pc);
  Type *ptr = PointerType::getUnqual(context_);
  auto callee = module_.getOrInsertFunction(
      "ppc_stwcx_op", FunctionType::get(Type::getVoidTy(context_),
                                        {ptr, Type::getInt8Ty(context_),
                                         Type::getInt32Ty(context_),
                                         Type::getInt32Ty(context_)},
                                        false));
  builder_.CreateCall(callee,
                      {ctx_, builder_.getInt8(inst.immediate & 0xFFu),
                       operand(inst, 0), builder_.getInt32(inst.guest_pc)});
  Value *exception =
      loadOffset(Type::getInt32Ty(context_), offsetof(CPUState, exception));
  BasicBlock *resume = BasicBlock::Create(context_, "stwcx_resume", function_);
  BasicBlock *failed = BasicBlock::Create(context_, "stwcx_exit", function_);
  builder_.CreateCondBr(builder_.CreateICmpEQ(exception, builder_.getInt32(0)),
                        resume, failed);
  builder_.SetInsertPoint(failed);
  returnFromBody();
  builder_.SetInsertPoint(resume);
  for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
    if (used_[slot])
      reloadState(static_cast<DolIRStateSlot>(slot));
  }
}

Value *FunctionEmitter::emitFPAvailable(u32 pc) {
  if (fp_available_checked_)
    return ConstantInt::getTrue(context_);
  Value *msr =
      builder_.CreateLoad(Type::getInt32Ty(context_), state_[DOLIR_STATE_MSR]);
  Value *enabled = builder_.CreateICmpNE(
      builder_.CreateAnd(msr, builder_.getInt32(1u << 13)),
      builder_.getInt32(0));
  BasicBlock *fast = builder_.GetInsertBlock();
  BasicBlock *good = BasicBlock::Create(context_, "fp_ok", function_);
  BasicBlock *cold = BasicBlock::Create(context_, "fp_check", function_);
  builder_.CreateCondBr(enabled, good, cold);
  builder_.SetInsertPoint(cold);
  if (modern_runtime_) {
    Value *exceptions = builder_.CreateLoad(Type::getInt32Ty(context_),
                                            state_[DOLIR_STATE_EXCEPTION]);
    builder_.CreateStore(builder_.CreateOr(exceptions, builder_.getInt32(0x40)),
                         state_[DOLIR_STATE_EXCEPTION]);
    sideExit(pc, 1);
    builder_.SetInsertPoint(good);
    fp_available_checked_ = true;
    return ConstantInt::getTrue(context_);
  }
  materialize(pc);
  auto callee = module_.getOrInsertFunction(
      "ppc_fp_available", FunctionType::get(Type::getInt1Ty(context_),
                                            {PointerType::getUnqual(context_),
                                             Type::getInt32Ty(context_)},
                                            false));
  CallInst *available =
      builder_.CreateCall(callee, {ctx_, builder_.getInt32(pc)});
  tagRuntimeCall(available, RegisterWrites::None);
  BasicBlock *reload = BasicBlock::Create(context_, "fp_reload", function_);
  BasicBlock *bad = BasicBlock::Create(context_, "fp_exit", function_);
  builder_.CreateCondBr(available, reload, bad);
  builder_.SetInsertPoint(bad);
  returnFromBody();
  builder_.SetInsertPoint(reload);
  // A successful check changes no guest register.
  for (u32 slot = 0; slot < DOLIR_STATE_COUNT && !sync_changed_; slot++) {
    if (used_[slot]) {
      auto stateSlot = static_cast<DolIRStateSlot>(slot);
      Value *reloaded = loadContext(stateSlot);
      builder_.CreateStore(reloaded, state_[slot]);
    }
  }
  builder_.CreateBr(good);
  builder_.SetInsertPoint(good);
  PHINode *checked = builder_.CreatePHI(Type::getInt1Ty(context_), 2);
  checked->addIncoming(ConstantInt::getTrue(context_), fast);
  checked->addIncoming(available, reload);
  fp_available_checked_ = true;
  return checked;
}

} // namespace dolllvm
