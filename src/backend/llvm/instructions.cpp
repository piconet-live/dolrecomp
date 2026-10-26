#include "backend/llvm/emitter.h"
#include "backend/llvm/native_abi.h"

#include <algorithm>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

namespace dolllvm {

using namespace llvm;

bool FunctionEmitter::emitRegion(u32 index, raw_ostream &diagnostics) {
  resetFPRepresentations();
  applySingleSeed(index);
  known_state_.fill(nullptr);
  psq_direct_proven_ = false;
  psq_indexed_proven_ = false;
  fp_available_checked_ = false;
  clearPendingFPRF();
  unsynced_.reset();
  // What the runs that branch here have left unwritten (carryUnsynced).
  if (!lazy_allowed_.empty())
    unsynced_ = lazy_allowed_[index];
  lazy_edge_.fill(nullptr);
  deferred_cycles_ = 0;
  fpscr_before_write_ = nullptr;
  builder_.SetInsertPoint(blocks_[index]);
  GlobalVariable *runCount =
      countHere(source_.blocks[index].guest_address, closed_ ? 1u : 0u, 0);
  for (u32 current = index; current < source_.block_count; current++) {
    const DolIRBlock &block = source_.blocks[current];
    if (current != index && region_leaders_[current]) {
      setCountDetail(runCount, closed_ ? 1u : 0u, current - index);
      materializeFPRF();
      if (!carryUnsynced(&current, 1))
        storeUnsynced(true);
      flushDeferredCycles();
      recordSingleEdge();
      builder_.CreateBr(blocks_[current]);
      return true;
    }
    const bool useService =
        modern_runtime_ && native_abi_ && needsInterpreter(block);
    if (loop_headers_[current])
      emitBudgetGuard(block.guest_address, true);
    chargeCycles(block.cycle_cost);
    if (useService) {
      emitInstructionService(block.guest_address);
    } else {
      values_.assign(source_.value_count, nullptr);
      for (u32 i = 0; i < block.instruction_count; i++) {
        if (!emitInstruction(block.instructions[i], diagnostics))
          return false;
      }
    }
    if (block.terminator.kind == DOLIR_TERM_FALLTHROUGH) {
      u32 next = block.terminator.targets[0];
      if (next != DOLIR_NO_BLOCK && next == current + 1u &&
          next < source_.block_count && !region_leaders_[next])
        continue;
    }
    setCountDetail(runCount, closed_ ? 1u : 0u, current - index + 1u);
    materializeFPRF();
    {
      const DolIRTerminator &term = block.terminator;
      bool carried = false;
      if (!term.linked && (term.kind == DOLIR_TERM_FALLTHROUGH ||
                           term.kind == DOLIR_TERM_BRANCH))
        carried = carryUnsynced(&term.targets[0], 1);
      else if (!term.linked && term.kind == DOLIR_TERM_COND_BRANCH)
        carried = carryUnsynced(term.targets, 2);
      if (!carried)
        storeUnsynced(true);
    }
    flushDeferredCycles();
    // The closing branch of a plain terminator is created in this block
    // (calls and exits get blocks of their own, which provide no facts).
    if (block.terminator.kind == DOLIR_TERM_FALLTHROUGH ||
        block.terminator.kind == DOLIR_TERM_BRANCH ||
        block.terminator.kind == DOLIR_TERM_COND_BRANCH)
      recordSingleEdge();
    return emitTerminator(block.terminator, diagnostics);
  }
  diagnostics << "dolllvm: unterminated native region at 0x"
              << format_hex_no_prefix(source_.blocks[index].guest_address, 8)
              << "\n";
  return false;
}

Value *FunctionEmitter::operand(const DolIRInstruction &inst, u32 index) {
  return values_[inst.operands[index]];
}

Value *FunctionEmitter::castValue(DolIROp op, Type *resultType, Value *value) {
  switch (op) {
  case DOLIR_OP_TRUNC:
    return builder_.CreateTrunc(value, resultType);
  case DOLIR_OP_ZEXT:
    return builder_.CreateZExt(value, resultType);
  case DOLIR_OP_SEXT:
    return builder_.CreateSExt(value, resultType);
  case DOLIR_OP_BITCAST:
    return builder_.CreateBitCast(value, resultType);
  case DOLIR_OP_FPTRUNC:
    return builder_.CreateFPTrunc(value, resultType);
  case DOLIR_OP_FPEXT:
    return builder_.CreateFPExt(value, resultType);
  default:
    return nullptr;
  }
}

Value *FunctionEmitter::bswap(Value *value) {
  auto *integer = cast<IntegerType>(value->getType());
  if (integer->getBitWidth() == 8)
    return value;
  Function *intrinsic =
      Intrinsic::getDeclaration(&module_, Intrinsic::bswap, {value->getType()});
  return builder_.CreateCall(intrinsic, {value});
}

bool FunctionEmitter::emitInstruction(const DolIRInstruction &inst,
                                      raw_ostream &diagnostics) {
  current_pc_ = inst.guest_pc;
  Value *result = nullptr;
  Type *resultType = type(inst.type);
  switch (inst.op) {
  case DOLIR_OP_CONSTANT:
    if (inst.type == DOLIR_TYPE_F32)
      result = ConstantFP::get(
          context_, APFloat(APFloat::IEEEsingle(), APInt(32, inst.immediate)));
    else if (inst.type == DOLIR_TYPE_F64)
      result = ConstantFP::get(
          context_, APFloat(APFloat::IEEEdouble(), APInt(64, inst.immediate)));
    else
      result = ConstantInt::get(resultType, inst.immediate);
    break;
  case DOLIR_OP_STATE_READ:
    result = stateValue(static_cast<DolIRStateSlot>(inst.aux));
    if (inst.aux >= DOLIR_STATE_FPR0 && inst.aux <= DOLIR_STATE_FPR31 &&
        ps0Single(inst.aux - DOLIR_STATE_FPR0))
      noteSingleValue(result);
    if (track_singles_ && inst.aux >= DOLIR_STATE_FPR0 &&
        inst.aux <= DOLIR_STATE_PS1_31) {
      const bool ps0 = inst.aux <= DOLIR_STATE_FPR31;
      const u32 reg = inst.aux - (ps0 ? DOLIR_STATE_FPR0 : DOLIR_STATE_PS1_0);
      if (ps0 ? laneView(reg)
              : fp_rep_[reg] == FPRepresentation::PairF32 &&
                    fp_exact_single_[reg])
        noteTwin(result,
                 builder_.CreateExtractElement(pairF32(reg), ps0 ? 0u : 1u),
                 fp_denormal_safe_[reg], fp_single_canonical_[reg]);
    }
    break;
  case DOLIR_OP_STATE_WRITE:
    emitStateWrite(inst);
    break;
  case DOLIR_OP_ADD:
    result = builder_.CreateAdd(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_SUB:
    result = builder_.CreateSub(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_MUL:
    result = builder_.CreateMul(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_UDIV:
    result = builder_.CreateUDiv(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_SDIV:
    result = builder_.CreateSDiv(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_AND:
    result = builder_.CreateAnd(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_OR:
    result = builder_.CreateOr(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_XOR:
    result = builder_.CreateXor(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_NOT:
    result = builder_.CreateNot(operand(inst, 0));
    break;
  case DOLIR_OP_SHL:
    result = builder_.CreateShl(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_LSHR:
    result = builder_.CreateLShr(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_ASHR:
    result = builder_.CreateAShr(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_ROTL: {
    Function *intrinsic =
        Intrinsic::getDeclaration(&module_, Intrinsic::fshl, {resultType});
    result = builder_.CreateCall(
        intrinsic, {operand(inst, 0), operand(inst, 0), operand(inst, 1)});
    break;
  }
  case DOLIR_OP_CLZ: {
    Function *intrinsic =
        Intrinsic::getDeclaration(&module_, Intrinsic::ctlz, {resultType});
    result = builder_.CreateCall(
        intrinsic, {operand(inst, 0), ConstantInt::getFalse(context_)});
    break;
  }
  case DOLIR_OP_BSWAP:
    result = bswap(operand(inst, 0));
    break;
  case DOLIR_OP_TRUNC:
  case DOLIR_OP_ZEXT:
  case DOLIR_OP_SEXT:
  case DOLIR_OP_BITCAST:
  case DOLIR_OP_FPTRUNC:
  case DOLIR_OP_FPEXT:
    result = castValue(inst.op, resultType, operand(inst, 0));
    if (inst.op == DOLIR_OP_FPEXT && operand(inst, 0)->getType()->isFloatTy()) {
      noteSingleValue(result);
      // An lfs load: widened under the host mode that single arithmetic on
      // it will run under; its bits are the loaded ones (not canonical).
      if (track_singles_)
        noteTwin(result, operand(inst, 0), true, false);
    }
    break;
  case DOLIR_OP_ICMP_EQ:
    result = builder_.CreateICmpEQ(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_ICMP_NE:
    result = builder_.CreateICmpNE(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_ICMP_ULT:
    result = builder_.CreateICmpULT(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_ICMP_ULE:
    result = builder_.CreateICmpULE(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_ICMP_SLT:
    result = builder_.CreateICmpSLT(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_ICMP_SLE:
    result = builder_.CreateICmpSLE(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_FCMP_OEQ:
    result = builder_.CreateFCmpOEQ(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_FCMP_OLT:
    result = builder_.CreateFCmpOLT(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_FCMP_OGE:
    result = builder_.CreateFCmpOGE(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_SELECT:
    result = builder_.CreateSelect(operand(inst, 0), operand(inst, 1),
                                   operand(inst, 2));
    if (track_singles_) {
      const SingleTwin *left = twinOf(operand(inst, 1));
      const SingleTwin *right = twinOf(operand(inst, 2));
      if (left && right)
        noteTwin(result,
                 builder_.CreateSelect(operand(inst, 0), left->single,
                                       right->single),
                 left->safe && right->safe,
                 left->canonical && right->canonical);
    }
    break;
  case DOLIR_OP_FADD:
    result = builder_.CreateFAdd(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_FSUB:
    result = builder_.CreateFSub(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_FMUL:
    result = builder_.CreateFMul(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_FDIV:
    result = builder_.CreateFDiv(operand(inst, 0), operand(inst, 1));
    break;
  case DOLIR_OP_FNEG:
    result = builder_.CreateFNeg(operand(inst, 0));
    if (std::find(single_values_.begin(), single_values_.end(),
                  operand(inst, 0)) != single_values_.end())
      noteSingleValue(result);
    if (const SingleTwin *twin = twinOf(operand(inst, 0)))
      noteTwin(result, builder_.CreateFNeg(twin->single), twin->safe,
               twin->canonical);
    break;
  case DOLIR_OP_FABS: {
    Function *intrinsic =
        Intrinsic::getDeclaration(&module_, Intrinsic::fabs, {resultType});
    result = builder_.CreateCall(intrinsic, {operand(inst, 0)});
    if (std::find(single_values_.begin(), single_values_.end(),
                  operand(inst, 0)) != single_values_.end())
      noteSingleValue(result);
    if (const SingleTwin *twin = twinOf(operand(inst, 0))) {
      Function *single = Intrinsic::getDeclaration(
          &module_, Intrinsic::fabs, {twin->single->getType()});
      noteTwin(result, builder_.CreateCall(single, {twin->single}), twin->safe,
               twin->canonical);
    }
    break;
  }
  case DOLIR_OP_VECTOR_BUILD: {
    result = PoisonValue::get(resultType);
    result =
        builder_.CreateInsertElement(result, operand(inst, 0), uint64_t{0});
    result = builder_.CreateInsertElement(result, operand(inst, 1), 1u);
    break;
  }
  case DOLIR_OP_VECTOR_EXTRACT:
    result = builder_.CreateExtractElement(operand(inst, 0), inst.aux);
    break;
  case DOLIR_OP_VECTOR_SHUFFLE:
    result = builder_.CreateShuffleVector(
        operand(inst, 0), operand(inst, 1),
        {static_cast<int>(inst.aux & 0xFFu),
         static_cast<int>((inst.aux >> 8) & 0xFFu)});
    break;
  case DOLIR_OP_GUEST_LOAD:
    result = emitGuestLoad(inst, operand(inst, 0), resultType, inst.aux & 0xffu,
                           (inst.aux & 0x100u) != 0);
    break;
  case DOLIR_OP_GUEST_STORE:
    emitGuestStore(inst, operand(inst, 0), operand(inst, 1), inst.aux & 0xffu);
    break;
  case DOLIR_OP_HELPER_CALL:
    if (inst.aux == DOLIR_HELPER_FP_AVAILABLE)
      result = emitFPAvailable(inst.guest_pc);
    else if (inst.aux == DOLIR_HELPER_MEMORY_FENCE)
      builder_.CreateFence(AtomicOrdering::SequentiallyConsistent);
    else if (inst.aux == DOLIR_HELPER_EXACT_FLOAT)
      emitExactFloat(inst.immediate);
    else if (inst.aux == DOLIR_HELPER_EXACT_PAIRED)
      emitExactPaired(inst.immediate);
    else if (inst.aux == DOLIR_HELPER_PSQ_LOAD ||
             inst.aux == DOLIR_HELPER_PSQ_STORE)
      result = emitPSQ(inst);
    else if (inst.aux == DOLIR_HELPER_STORE_CONDITIONAL)
      emitStoreConditional(inst);
    else if (inst.aux == DOLIR_HELPER_FPSCR_UPDATED)
      emitFPSCRUpdated();
    else if (inst.aux == DOLIR_HELPER_FPSCR_BIT)
      emitFPSCRBit(inst.immediate);
    else if (inst.aux == DOLIR_HELPER_PROGRAM_EXCEPTION)
      emitProgramException(inst);
    else if (inst.aux == DOLIR_HELPER_SPR_READ)
      result = emitSPRRead(inst);
    else if (inst.aux == DOLIR_HELPER_SPR_WRITE)
      emitSPRWrite(inst);
    else if (inst.aux == DOLIR_HELPER_TIMEBASE_READ)
      result = emitTimebaseRead();
    else if (inst.aux == DOLIR_HELPER_TIMEBASE_WRITE)
      emitTimebaseWrite(inst);
    else if (inst.aux == DOLIR_HELPER_LSWX)
      emitLSWX(inst);
    else if (inst.aux == DOLIR_HELPER_DCBZ_L ||
             inst.aux == DOLIR_HELPER_ECIWX || inst.aux == DOLIR_HELPER_ECOWX ||
             inst.aux == DOLIR_HELPER_TLBIE ||
             inst.aux == DOLIR_HELPER_CACHE_CONTROL)
      result = emitRuntimeBoundary(inst);
    else {
      diagnostics << "dolllvm: unsupported helper " << inst.aux << " at 0x"
                  << format_hex_no_prefix(inst.guest_pc, 8) << "\n";
      return false;
    }
    break;
  default:
    diagnostics << "dolllvm: unsupported DolIR op " << unsigned(inst.op)
                << " at 0x" << format_hex_no_prefix(inst.guest_pc, 8) << "\n";
    return false;
  }
  if (inst.result)
    values_[inst.result] = result;
  return inst.type == DOLIR_TYPE_VOID || result != nullptr;
}
} // namespace dolllvm
