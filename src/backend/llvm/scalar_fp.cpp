#include "backend/llvm/emitter.h"

#include "backend/llvm/fp_fixups.h"

#include <algorithm>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

// Scalar floating point emitted inline (DOLRECOMP_SCALAR_FP=inline).
//
// The default path hands every scalar FP instruction to a runtime helper that
// works on CPUState: the registers involved are stored, the helper is called
// and the results are loaded back. That keeps the full FPSCR exception record
// (FX, VX*, ZX, XX, FI, FR) but costs a call and several memory round trips
// for what is one host instruction.
//
// This path does what Dolphin's Jit64 does with its default settings
// (AccurateNaNs off, float exceptions off), which is also what the paired
// single path of this backend already does:
//
//   * the operation runs in double precision on the host, under the host
//     rounding mode and flush-to-zero state the runtime keeps in step with
//     FPSCR[RN] and FPSCR[NI];
//   * a single-precision result is that double rounded to single (cvtsd2ss),
//     and is written to both halves of the register;
//   * fmuls and the fmadds family round the multiplier's mantissa to 25 bits
//     first (Jit64's Force25BitPrecision), with no special case for NaN or
//     subnormal multipliers;
//   * the fmadds family applies Jit64's error-free correction for the
//     double-rounding tie (fmadds_eft);
//   * FPRF is set from the result, FPCC from a compare, and no other FPSCR bit
//     changes. NaN operands and invalid operations give the host's NaN, not
//     PowerPC's.
//
// The helper path remains for every other target and for fres.

namespace dolllvm {

using namespace llvm;

Value *FunctionEmitter::scalarF64(u32 reg) {
  return stateValue(static_cast<DolIRStateSlot>(DOLIR_STATE_FPR0 + reg));
}

bool FunctionEmitter::ps0Single(u32 reg) const {
  return fp_exact_single_[reg] || fp_ps0_single_[reg];
}

bool FunctionEmitter::singleSafe(u32 reg) const {
  if (track_singles_)
    return laneView(reg) && fp_denormal_safe_[reg];
  return float_single_math_ && fp_rep_[reg] == FPRepresentation::PairF32 &&
         fp_exact_single_[reg] && fp_denormal_safe_[reg];
}

Value *FunctionEmitter::scalarF32(u32 reg) {
  return builder_.CreateExtractElement(pairF32(reg), uint64_t{0}, "fp.single");
}

// lfs writes both halves of a register with one single widened to double.
// Once both halves are seen to come from the same single, the register gets
// a single view of it. The widening ran under the host's denormals-are-zero
// state and single arithmetic on the view runs under it too, so they agree;
// the bits of the view itself are the loaded ones, which is why it is not
// marked canonical.
void FunctionEmitter::noteFloatLaneWrite(DolIRStateSlot slot, Value *value) {
  if (!float_single_math_)
    return;
  const bool ps0 = slot >= DOLIR_STATE_FPR0 && slot <= DOLIR_STATE_FPR31;
  const bool ps1 = slot >= DOLIR_STATE_PS1_0 && slot <= DOLIR_STATE_PS1_31;
  if (!ps0 && !ps1)
    return;
  const u32 reg = slot - (ps0 ? DOLIR_STATE_FPR0 : DOLIR_STATE_PS1_0);
  auto *widened = dyn_cast<FPExtInst>(value);
  Value *single =
      widened && widened->getOperand(0)->getType()->isFloatTy()
          ? widened->getOperand(0)
          : nullptr;
  (ps0 ? fp_ps0_float_ : fp_ps1_float_)[reg] = single;
  if (!single || fp_ps0_float_[reg] != fp_ps1_float_[reg])
    return;
  Type *pairType = FixedVectorType::get(Type::getFloatTy(context_), 2);
  Value *pair = builder_.CreateInsertElement(PoisonValue::get(pairType), single,
                                             uint64_t{0});
  pair = builder_.CreateInsertElement(pair, single, 1u);
  builder_.CreateStore(pair, pair_f32_[reg]);
  fp_rep_[reg] = FPRepresentation::PairF32;
  fp_exact_single_[reg] = true;
  fp_denormal_safe_[reg] = true;
  fp_single_canonical_[reg] = false;
  fp_value_class_[reg] = FPValueClass::Unknown;
}

void FunctionEmitter::noteSingleValue(Value *value) {
  if (value)
    single_values_.push_back(value);
}

Value *FunctionEmitter::constrainedScalar(unsigned intrinsic, Value *left,
                                          Value *right) {
  return builder_.CreateConstrainedFPBinOp(
      static_cast<Intrinsic::ID>(intrinsic), left, right, nullptr, "fp.math",
      nullptr, RoundingMode::Dynamic, fp::ebIgnore);
}

Value *FunctionEmitter::roundScalarToSingle(Value *value) {
  return builder_.CreateConstrainedFPCast(
      Intrinsic::experimental_constrained_fptrunc, value,
      Type::getFloatTy(context_), nullptr, "fp.round", nullptr,
      RoundingMode::Dynamic, fp::ebIgnore);
}

// Jit64's Force25BitPrecision: (bits & ~0x7FFFFFF) + (bits & 0x8000000).
Value *FunctionEmitter::forceScalarMultiplierPrecision(Value *value) {
  Type *i64 = Type::getInt64Ty(context_);
  Value *bits = builder_.CreateBitCast(value, i64);
  Value *rounded = builder_.CreateAdd(
      builder_.CreateAnd(bits, builder_.getInt64(0xFFFFFFFFF8000000ull)),
      builder_.CreateAnd(bits, builder_.getInt64(0x0000000008000000ull)));
  return builder_.CreateBitCast(rounded, value->getType(), "fp.force25");
}

Value *FunctionEmitter::correctScalarFMARounding(Value *result, Value *left,
                                                 Value *multiplier,
                                                 Value *addend) {
  Value *bits = builder_.CreateBitCast(result, Type::getInt64Ty(context_));
  Value *tie = builder_.CreateICmpEQ(
      builder_.CreateAnd(bits, builder_.getInt64(0x000000001FFFFFFFull)),
      builder_.getInt64(0x0000000010000000ull));
  BasicBlock *regular = builder_.GetInsertBlock();
  BasicBlock *rare = BasicBlock::Create(context_, "fma_tie_fixup", function_);
  BasicBlock *join = BasicBlock::Create(context_, "fma_tie_join", function_);
  builder_.CreateCondBr(tie, rare, join,
                        MDBuilder(context_).createBranchWeights(1, 1000000));

  builder_.SetInsertPoint(rare);
  Function *fixup = getScalarFMARoundingFixup(module_);
  CallInst *fixed = builder_.CreateCall(
      fixup, {result, left, multiplier, addend}, "fma.corrected");
  fixed->setCallingConv(fixup->getCallingConv());
  builder_.CreateBr(join);

  builder_.SetInsertPoint(join);
  PHINode *selected = builder_.CreatePHI(result->getType(), 2, "fma.exact");
  selected->addIncoming(result, regular);
  selected->addIncoming(fixed, rare);
  return selected;
}

void FunctionEmitter::writeScalarSingle(u32 reg, Value *single) {
  Type *pairType = FixedVectorType::get(Type::getFloatTy(context_), 2);
  Value *pair = builder_.CreateInsertElement(PoisonValue::get(pairType), single,
                                             uint64_t{0});
  pair = builder_.CreateInsertElement(pair, single, 1u);
  writePairF32(reg, pair, true, FPValueClass::Unknown);
  updateFPRF(single);
}

void FunctionEmitter::writeScalarDouble(u32 reg, Value *value) {
  auto slot = static_cast<DolIRStateSlot>(DOLIR_STATE_FPR0 + reg);
  builder_.CreateStore(value, state_[slot]);
  noteStateWrite(slot, value);
  updateFPRF(value);
}

bool FunctionEmitter::emitInlineScalarFloat(DolIRExactFloat op, u32 d, u32 a,
                                            u32 b, u32 c, u32 crfd) {
  Type *f64 = Type::getDoubleTy(context_);
  Type *i32 = Type::getInt32Ty(context_);
  Type *i64 = Type::getInt64Ty(context_);
  auto multiplier = [&](u32 reg) {
    Value *value = scalarF64(reg);
    return ps0Single(reg) ? value : forceScalarMultiplierPrecision(value);
  };

  const bool singleResult =
      (op >= DOLIR_EXACT_FADDS && op <= DOLIR_EXACT_FDIVS) ||
      (op >= DOLIR_EXACT_FMADDS && op <= DOLIR_EXACT_FNMSUBS);
  if (singleResult)
    stat_double_++;
  // Single precision throughout when every input has a single view.
  if (op >= DOLIR_EXACT_FADDS && op <= DOLIR_EXACT_FDIVS) {
    const u32 right = op == DOLIR_EXACT_FMULS ? c : b;
    if (singleSafe(a) && singleSafe(right)) {
      stat_double_--;
      stat_single_++;
      const unsigned intrinsic =
          op == DOLIR_EXACT_FADDS   ? Intrinsic::experimental_constrained_fadd
          : op == DOLIR_EXACT_FSUBS ? Intrinsic::experimental_constrained_fsub
          : op == DOLIR_EXACT_FMULS ? Intrinsic::experimental_constrained_fmul
                                    : Intrinsic::experimental_constrained_fdiv;
      writeScalarSingle(
          d, constrainedScalar(intrinsic, scalarF32(a), scalarF32(right)));
      return true;
    }
  }
  if (op >= DOLIR_EXACT_FMADDS && op <= DOLIR_EXACT_FNMSUBS && singleSafe(a) &&
      singleSafe(b) && singleSafe(c)) {
    Type *f32 = Type::getFloatTy(context_);
    stat_double_--;
    stat_single_++;
    Value *addend = scalarF32(b);
    if (op == DOLIR_EXACT_FMSUBS || op == DOLIR_EXACT_FNMSUBS)
      addend = builder_.CreateFNeg(addend, "fma.subtract");
    uses_fma_ = true;
    Function *fma = Intrinsic::getDeclaration(
        &module_, Intrinsic::experimental_constrained_fma, {f32});
    Value *result = builder_.CreateConstrainedFPCall(
        fma, {scalarF32(a), scalarF32(c), addend}, "fma.single",
        RoundingMode::Dynamic, fp::ebIgnore);
    const bool negative =
        op == DOLIR_EXACT_FNMADDS || op == DOLIR_EXACT_FNMSUBS;
    Value *fused = result;
    if (negative)
      result = builder_.CreateBitCast(
          builder_.CreateXor(builder_.CreateBitCast(result, i32),
                             builder_.getInt32(0x80000000u)),
          f32, "fma.negative");
    if (track_singles_) {
      // The single fused multiply-add equals the double one with the tie
      // correction, narrowed, only for a normal result. Zero, a denormal,
      // infinity or a NaN takes the double sequence, on the register's
      // doubles. That sequence negates before it narrows, which is the same
      // as negating the single only when rounding is to nearest or toward
      // zero (FPSCR[RN] bit 1 clear).
      Value *unusual = singleFMAUnusual(fused);
      if (negative)
        unusual = builder_.CreateOr(
            unusual,
            builder_.CreateICmpNE(
                builder_.CreateAnd(
                    builder_.CreateLoad(i32, state_[DOLIR_STATE_FPSCR]),
                    builder_.getInt32(2)),
                builder_.getInt32(0)));
      BasicBlock *fastEnd = builder_.GetInsertBlock();
      BasicBlock *rare =
          BasicBlock::Create(context_, "fma_single_unusual", function_);
      BasicBlock *join =
          BasicBlock::Create(context_, "fma_single_join", function_);
      builder_.CreateCondBr(unusual, rare, join,
                            MDBuilder(context_).createBranchWeights(1, 256));
      builder_.SetInsertPoint(rare);
      Value *left = scalarF64(a);
      Value *factor = scalarF64(c);
      Value *wideAddend = scalarF64(b);
      if (op == DOLIR_EXACT_FMSUBS || op == DOLIR_EXACT_FNMSUBS)
        wideAddend = builder_.CreateFNeg(wideAddend, "fma.subtract");
      Function *wideFMA = Intrinsic::getDeclaration(
          &module_, Intrinsic::experimental_constrained_fma, {f64});
      Value *wide = builder_.CreateConstrainedFPCall(
          wideFMA, {left, factor, wideAddend}, "fma", RoundingMode::Dynamic,
          fp::ebIgnore);
      wide = correctScalarFMARounding(wide, left, factor, wideAddend);
      if (negative)
        wide = builder_.CreateBitCast(
            builder_.CreateXor(builder_.CreateBitCast(wide, i64),
                               builder_.getInt64(0x8000000000000000ull)),
            f64, "fma.negative");
      Value *slow = roundScalarToSingle(wide);
      BasicBlock *rareEnd = builder_.GetInsertBlock();
      builder_.CreateBr(join);
      builder_.SetInsertPoint(join);
      PHINode *selected = builder_.CreatePHI(f32, 2, "fma.single.exact");
      selected->addIncoming(result, fastEnd);
      selected->addIncoming(slow, rareEnd);
      result = selected;
    }
    writeScalarSingle(d, result);
    return true;
  }
  // Not under DOLRECOMP_SINGLE_FP=track: a result that is an exact denormal
  // under flush-to-zero is zero in the register and not in the view.
  if (op == DOLIR_EXACT_FRSP && !track_singles_ && singleSafe(b) &&
      fp_single_canonical_[b]) {
    writeScalarSingle(d, scalarF32(b));
    return true;
  }

  switch (op) {
  case DOLIR_EXACT_FADDS:
    writeScalarSingle(d, roundScalarToSingle(constrainedScalar(
                             Intrinsic::experimental_constrained_fadd,
                             scalarF64(a), scalarF64(b))));
    return true;
  case DOLIR_EXACT_FSUBS:
    writeScalarSingle(d, roundScalarToSingle(constrainedScalar(
                             Intrinsic::experimental_constrained_fsub,
                             scalarF64(a), scalarF64(b))));
    return true;
  case DOLIR_EXACT_FMULS:
    writeScalarSingle(d, roundScalarToSingle(constrainedScalar(
                             Intrinsic::experimental_constrained_fmul,
                             scalarF64(a), multiplier(c))));
    return true;
  case DOLIR_EXACT_FDIVS:
    writeScalarSingle(d, roundScalarToSingle(constrainedScalar(
                             Intrinsic::experimental_constrained_fdiv,
                             scalarF64(a), scalarF64(b))));
    return true;
  case DOLIR_EXACT_FADD:
    writeScalarDouble(d, constrainedScalar(
                             Intrinsic::experimental_constrained_fadd,
                             scalarF64(a), scalarF64(b)));
    return true;
  case DOLIR_EXACT_FSUB:
    writeScalarDouble(d, constrainedScalar(
                             Intrinsic::experimental_constrained_fsub,
                             scalarF64(a), scalarF64(b)));
    return true;
  case DOLIR_EXACT_FMUL:
    writeScalarDouble(d, constrainedScalar(
                             Intrinsic::experimental_constrained_fmul,
                             scalarF64(a), scalarF64(c)));
    return true;
  case DOLIR_EXACT_FDIV:
    writeScalarDouble(d, constrainedScalar(
                             Intrinsic::experimental_constrained_fdiv,
                             scalarF64(a), scalarF64(b)));
    return true;
  case DOLIR_EXACT_FRSP:
    writeScalarSingle(d, roundScalarToSingle(scalarF64(b)));
    return true;
  case DOLIR_EXACT_FMADD:
  case DOLIR_EXACT_FMSUB:
  case DOLIR_EXACT_FNMADD:
  case DOLIR_EXACT_FNMSUB:
  case DOLIR_EXACT_FMADDS:
  case DOLIR_EXACT_FMSUBS:
  case DOLIR_EXACT_FNMADDS:
  case DOLIR_EXACT_FNMSUBS: {
    const bool single = op >= DOLIR_EXACT_FMADDS;
    const bool subtract = op == DOLIR_EXACT_FMSUB || op == DOLIR_EXACT_FNMSUB ||
                          op == DOLIR_EXACT_FMSUBS || op == DOLIR_EXACT_FNMSUBS;
    const bool negative = op == DOLIR_EXACT_FNMADD ||
                          op == DOLIR_EXACT_FNMSUB ||
                          op == DOLIR_EXACT_FNMADDS ||
                          op == DOLIR_EXACT_FNMSUBS;
    Value *left = scalarF64(a);
    Value *factor = single ? multiplier(c) : scalarF64(c);
    Value *addend = scalarF64(b);
    if (subtract)
      addend = builder_.CreateFNeg(addend, "fma.subtract");
    uses_fma_ = true;
    Function *fma = Intrinsic::getDeclaration(
        &module_, Intrinsic::experimental_constrained_fma, {f64});
    Value *result = builder_.CreateConstrainedFPCall(
        fma, {left, factor, addend}, "fma", RoundingMode::Dynamic,
        fp::ebIgnore);
    if (single)
      result = correctScalarFMARounding(result, left, factor, addend);
    // PowerPC rounds before the final negation, and so does Jit64: it flips
    // the sign with a separate xorpd, NaNs included.
    if (negative)
      result = builder_.CreateBitCast(
          builder_.CreateXor(builder_.CreateBitCast(result, i64),
                             builder_.getInt64(0x8000000000000000ull)),
          f64, "fma.negative");
    if (single)
      writeScalarSingle(d, roundScalarToSingle(result));
    else
      writeScalarDouble(d, result);
    return true;
  }
  case DOLIR_EXACT_FCMPU:
  case DOLIR_EXACT_FCMPO: {
    Value *left = scalarF64(a);
    Value *right = scalarF64(b);
    Value *compare = builder_.CreateSelect(
        builder_.CreateFCmpOLT(left, right), builder_.getInt32(8),
        builder_.CreateSelect(
            builder_.CreateFCmpOGT(left, right), builder_.getInt32(4),
            builder_.CreateSelect(builder_.CreateFCmpOEQ(left, right),
                                  builder_.getInt32(2), builder_.getInt32(1))),
        "fcmp");
    auto crSlot = static_cast<DolIRStateSlot>(DOLIR_STATE_CR0 + crfd);
    builder_.CreateStore(compare, state_[crSlot]);
    noteStateWrite(crSlot, compare);
    updateFPCC(compare);
    return true;
  }
  case DOLIR_EXACT_FCTIW:
  case DOLIR_EXACT_FCTIWZ: {
    if (!Triple(module_.getTargetTriple()).isX86())
      return false;
    // Jit64: minsd against 2^31-1, then cvt(t)pd2dq. The host's "integer
    // indefinite" 0x80000000 is PowerPC's answer for NaN and for values below
    // -2^31; the minimum supplies 0x7fffffff above the range. The upper word
    // is 0xfff80000 for every input.
    Value *value = scalarF64(b);
    Value *limit = ConstantFP::get(f64, 2147483647.0);
    Value *clamped = builder_.CreateSelect(
        builder_.CreateFCmpOLT(limit, value), limit, value, "fctiw.clamped");
    Type *pairType = FixedVectorType::get(f64, 2);
    Value *lane = builder_.CreateInsertElement(
        ConstantAggregateZero::get(pairType), clamped, uint64_t{0});
    Function *convert = Intrinsic::getDeclaration(
        &module_, op == DOLIR_EXACT_FCTIWZ ? Intrinsic::x86_sse2_cvttsd2si
                                           : Intrinsic::x86_sse2_cvtsd2si);
    Value *integer = builder_.CreateCall(convert, {lane}, "fctiw");
    Value *bits = builder_.CreateOr(builder_.CreateZExt(integer, i64),
                                    builder_.getInt64(0xFFF8000000000000ull));
    auto slot = static_cast<DolIRStateSlot>(DOLIR_STATE_FPR0 + d);
    Value *stored = builder_.CreateBitCast(bits, f64);
    builder_.CreateStore(stored, state_[slot]);
    noteStateWrite(slot, stored);
    return true;
  }
  case DOLIR_EXACT_FRSQRTE: {
    // The table lookup has no FPSCR or CPU state of its own; Jit64 calls the
    // same function (Common::ApproximateReciprocalSquareRoot) for the cases
    // its inline table code does not cover.
    FunctionCallee estimate = module_.getOrInsertFunction(
        "ppc_approx_rsqrt", FunctionType::get(f64, {f64}, false));
    if (auto *function = dyn_cast<Function>(estimate.getCallee())) {
      function->addFnAttr(Attribute::NoUnwind);
      function->addFnAttr(Attribute::WillReturn);
      function->setMemoryEffects(MemoryEffects::none());
    }
    writeScalarDouble(d, builder_.CreateCall(estimate, {scalarF64(b)}));
    return true;
  }
  default:
    return false;
  }
}

} // namespace dolllvm
