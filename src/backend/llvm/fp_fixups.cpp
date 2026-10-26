#include "backend/llvm/fp_fixups.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

#include "common/types.h"

namespace dolllvm {

using namespace llvm;

static void markColdFixup(Function *function) {
  function->setCallingConv(CallingConv::PreserveAll);
  function->addFnAttr(Attribute::Cold);
  function->addFnAttr(Attribute::NoInline);
  function->addFnAttr(Attribute::NoUnwind);
  function->addFnAttr(Attribute::WillReturn);
  // Mach-O section specifiers are "__SEGMENT,__section" and its writer rejects
  // anything else outright, so a bare ".text.unlikely.*" name aborts emission on
  // Darwin targets ("invalid section specifier"). ELF and COFF both accept the
  // name, and the Cold plus NoInline attributes already carry the placement hint
  // on every target, so dropping only the name on Mach-O costs nothing.
  if (!Triple(function->getParent()->getTargetTriple()).isOSBinFormatMachO())
    function->setSection(".text.unlikely." + function->getName().str());
}

Function *getPairNaNFixup(Module &module, Type *pairType) {
  bool single = pairType->getScalarType()->isFloatTy();
  const char *name = single ? "fix_pair_nan_f32" : "fix_pair_nan_f64";
  if (Function *existing = module.getFunction(name))
    return existing;

  LLVMContext &context = module.getContext();
  Function *fixup = Function::Create(
      FunctionType::get(pairType, {pairType, pairType, pairType}, false),
      GlobalValue::InternalLinkage, name, module);
  markColdFixup(fixup);

  auto argument = fixup->arg_begin();
  Value *result = &*argument++;
  Value *left = &*argument++;
  Value *right = &*argument;
  BasicBlock *entry = BasicBlock::Create(context, "entry", fixup);
  IRBuilder<> builder(entry);
  unsigned width = pairType->getScalarSizeInBits();
  Type *integerPair = FixedVectorType::get(IntegerType::get(context, width), 2);
  auto splat = [integerPair](u64 value) {
    return ConstantVector::getSplat(
        ElementCount::getFixed(2),
        ConstantInt::get(integerPair->getScalarType(), value));
  };
  u64 quietBit = single ? 0x00400000u : 0x0008000000000000ull;
  u64 defaultNaN = single ? 0x7FC00000u : 0x7FF8000000000000ull;
  Value *leftNaN = builder.CreateFCmpUNO(left, left);
  Value *rightNaN = builder.CreateFCmpUNO(right, right);
  Value *resultNaN = builder.CreateFCmpUNO(result, result);
  Value *quietLeft = builder.CreateBitCast(
      builder.CreateOr(builder.CreateBitCast(left, integerPair),
                       splat(quietBit)),
      pairType);
  Value *quietRight = builder.CreateBitCast(
      builder.CreateOr(builder.CreateBitCast(right, integerPair),
                       splat(quietBit)),
      pairType);
  Value *fallback = builder.CreateSelect(
      resultNaN, builder.CreateBitCast(splat(defaultNaN), pairType), result);
  builder.CreateRet(builder.CreateSelect(
      leftNaN, quietLeft,
      builder.CreateSelect(rightNaN, quietRight, fallback)));
  return fixup;
}

Function *getFPRFUnusualFixup(Module &module, bool wide) {
  const char *name =
      wide ? "classify_fprf_unusual_f64" : "classify_fprf_unusual";
  if (Function *existing = module.getFunction(name))
    return existing;

  LLVMContext &context = module.getContext();
  Type *i32 = Type::getInt32Ty(context);
  Type *bitsType = wide ? Type::getInt64Ty(context) : i32;
  Function *fixup =
      Function::Create(FunctionType::get(i32, {bitsType}, false),
                       GlobalValue::InternalLinkage, name, module);
  markColdFixup(fixup);

  Value *bits = fixup->getArg(0);
  BasicBlock *entry = BasicBlock::Create(context, "entry", fixup);
  IRBuilder<> builder(entry);
  auto constant = [&](u64 value) { return ConstantInt::get(bitsType, value); };
  const u64 signBit = wide ? 0x8000000000000000ull : 0x80000000u;
  const u64 exponentMask = wide ? 0x7FF0000000000000ull : 0x7F800000u;
  const u64 fractionMask = wide ? 0x000FFFFFFFFFFFFFull : 0x007FFFFFu;
  Value *sign = builder.CreateICmpNE(builder.CreateAnd(bits, constant(signBit)),
                                     constant(0));
  Value *exponent = builder.CreateAnd(bits, constant(exponentMask));
  Value *hasFraction = builder.CreateICmpNE(
      builder.CreateAnd(bits, constant(fractionMask)), constant(0));
  Value *infinity = builder.CreateSelect(sign, builder.getInt32(0x09u),
                                         builder.getInt32(0x05u));
  Value *specialClass =
      builder.CreateSelect(hasFraction, builder.getInt32(0x11u), infinity);
  Value *zero = builder.CreateSelect(sign, builder.getInt32(0x12u),
                                     builder.getInt32(0x02u));
  Value *subnormal = builder.CreateSelect(sign, builder.getInt32(0x18u),
                                          builder.getInt32(0x14u));
  Value *zeroClass = builder.CreateSelect(hasFraction, subnormal, zero);
  builder.CreateRet(builder.CreateSelect(
      builder.CreateICmpEQ(exponent, constant(exponentMask)), specialClass,
      zeroClass));
  return fixup;
}

// The tie case of a single-precision fused multiply-add computed in double:
// rounding the double result to single would round twice. Dolphin's Jit64
// (fmadds_eft, 2606) recovers the sign of the exact error with an error-free
// transformation and moves the double one unit away from the tie. This is that
// routine; the caller has already found (result & 0x1fffffff) == 0x10000000.
Function *getScalarFMARoundingFixup(Module &module) {
  constexpr const char *name = "fix_scalar_fma_rounding";
  if (Function *existing = module.getFunction(name))
    return existing;

  LLVMContext &context = module.getContext();
  Type *f64 = Type::getDoubleTy(context);
  Type *i64 = Type::getInt64Ty(context);
  Function *fixup = Function::Create(
      FunctionType::get(f64, {f64, f64, f64, f64}, false),
      GlobalValue::InternalLinkage, name, module);
  markColdFixup(fixup);
  auto argument = fixup->arg_begin();
  Value *result = &*argument++;
  Value *left = &*argument++;
  Value *multiplier = &*argument++;
  Value *addend = &*argument;
  BasicBlock *entry = BasicBlock::Create(context, "entry", fixup);
  IRBuilder<> builder(entry);
  Function *fma = Intrinsic::getDeclaration(
      &module, Intrinsic::experimental_constrained_fma, {f64});
  auto binary = [&](Intrinsic::ID id, Value *a, Value *b) {
    return builder.CreateConstrainedFPBinOp(
        id, a, b, nullptr, "", nullptr, RoundingMode::Dynamic, fp::ebIgnore);
  };
  // s = a + b with a = left * multiplier, b = addend.
  Value *aPrime =
      binary(Intrinsic::experimental_constrained_fsub, result, addend);
  Value *bPrime =
      binary(Intrinsic::experimental_constrained_fsub, result, aPrime);
  Value *deltaA = builder.CreateConstrainedFPCall(
      fma, {left, multiplier, builder.CreateFNeg(aPrime)}, "",
      RoundingMode::Dynamic, fp::ebIgnore);
  Value *negativeDeltaB =
      binary(Intrinsic::experimental_constrained_fsub, bPrime, addend);
  Value *error =
      binary(Intrinsic::experimental_constrained_fsub, deltaA, negativeDeltaB);
  // ucomisd/je in the Jit64 routine: equal to zero or unordered leaves the
  // result alone.
  Value *unchanged =
      builder.CreateFCmpUEQ(error, ConstantFP::get(f64, 0.0));
  Value *bits = builder.CreateBitCast(result, i64);
  Value *direction = builder.CreateOr(
      builder.CreateAShr(
          builder.CreateXor(builder.CreateBitCast(error, i64), bits), 63),
      ConstantInt::get(i64, 1));
  builder.CreateRet(builder.CreateBitCast(
      builder.CreateSelect(unchanged, bits, builder.CreateAdd(bits, direction)),
      f64));
  return fixup;
}

Function *getPairFMANaNFixup(Module &module, Type *pairType) {
  bool single = pairType->getScalarType()->isFloatTy();
  const char *name = single ? "fix_pair_fma_nan_f32" : "fix_pair_fma_nan_f64";
  if (Function *existing = module.getFunction(name))
    return existing;

  LLVMContext &context = module.getContext();
  Function *fixup = Function::Create(
      FunctionType::get(pairType, {pairType, pairType, pairType, pairType},
                        false),
      GlobalValue::InternalLinkage, name, module);
  markColdFixup(fixup);
  auto argument = fixup->arg_begin();
  Value *result = &*argument++;
  Value *left = &*argument++;
  Value *addend = &*argument++;
  Value *multiplier = &*argument;
  BasicBlock *entry = BasicBlock::Create(context, "entry", fixup);
  IRBuilder<> builder(entry);
  unsigned width = pairType->getScalarSizeInBits();
  Type *integerPair = FixedVectorType::get(IntegerType::get(context, width), 2);
  auto splat = [integerPair](u64 value) {
    return ConstantVector::getSplat(
        ElementCount::getFixed(2),
        ConstantInt::get(integerPair->getScalarType(), value));
  };
  u64 quietBit = single ? 0x00400000u : 0x0008000000000000ull;
  u64 defaultNaN = single ? 0x7FC00000u : 0x7FF8000000000000ull;
  auto quiet = [&](Value *value) {
    return builder.CreateBitCast(
        builder.CreateOr(builder.CreateBitCast(value, integerPair),
                         splat(quietBit)),
        pairType);
  };
  Value *fallback = builder.CreateSelect(
      builder.CreateFCmpUNO(result, result),
      builder.CreateBitCast(splat(defaultNaN), pairType), result);
  Value *selected =
      builder.CreateSelect(builder.CreateFCmpUNO(multiplier, multiplier),
                           quiet(multiplier), fallback);
  selected = builder.CreateSelect(builder.CreateFCmpUNO(addend, addend),
                                  quiet(addend), selected);
  selected = builder.CreateSelect(builder.CreateFCmpUNO(left, left),
                                  quiet(left), selected);
  builder.CreateRet(selected);
  return fixup;
}

Function *getPairFMARoundingFixup(Module &module, bool jit64) {
  const char *name =
      jit64 ? "fix_pair_fma_rounding_jit64" : "fix_pair_fma_rounding";
  if (Function *existing = module.getFunction(name))
    return existing;

  LLVMContext &context = module.getContext();
  Type *pairType = FixedVectorType::get(Type::getDoubleTy(context), 2);
  Type *integerPair = FixedVectorType::get(Type::getInt64Ty(context), 2);
  Function *fixup = Function::Create(
      FunctionType::get(pairType, {pairType, pairType, pairType, pairType},
                        false),
      GlobalValue::InternalLinkage, name, module);
  markColdFixup(fixup);
  auto argument = fixup->arg_begin();
  Value *result = &*argument++;
  Value *left = &*argument++;
  Value *multiplier = &*argument++;
  Value *addend = &*argument;
  BasicBlock *entry = BasicBlock::Create(context, "entry", fixup);
  IRBuilder<> builder(entry);
  Function *fma = Intrinsic::getDeclaration(
      &module, Intrinsic::experimental_constrained_fma, {pairType});
  auto binary = [&](Intrinsic::ID id, Value *a, Value *b) {
    return builder.CreateConstrainedFPBinOp(
        id, a, b, nullptr, "", nullptr, RoundingMode::Dynamic, fp::ebIgnore);
  };
  Value *aPrime =
      binary(Intrinsic::experimental_constrained_fsub, addend, result);
  Value *bPrime =
      binary(Intrinsic::experimental_constrained_fadd, result, aPrime);
  Value *deltaA = builder.CreateConstrainedFPCall(
      fma, {left, multiplier, aPrime}, "", RoundingMode::Dynamic, fp::ebIgnore);
  Value *deltaB =
      binary(Intrinsic::experimental_constrained_fsub, addend, bPrime);
  Value *error =
      binary(Intrinsic::experimental_constrained_fadd, deltaA, deltaB);
  Value *bits = builder.CreateBitCast(result, integerPair);
  auto splat = [integerPair](u64 value) {
    return ConstantVector::getSplat(
        ElementCount::getFixed(2),
        ConstantInt::get(integerPair->getScalarType(), value));
  };
  Value *tie = builder.CreateICmpEQ(
      builder.CreateAnd(bits, splat(0x000000001FFFFFFFull)),
      splat(0x0000000010000000ull));
  Value *zero = ConstantAggregateZero::get(pairType);
  Value *corrected;
  Value *changes;
  if (jit64) {
    // Jit64's ps_madd_eft: cmppd-eq for "no error" (so an unordered error
    // still moves the result), and the direction from the two sign bits.
    changes = builder.CreateAnd(
        tie, builder.CreateNot(builder.CreateFCmpOEQ(error, zero)));
    Value *direction = builder.CreateOr(
        builder.CreateAShr(
            builder.CreateXor(builder.CreateBitCast(error, integerPair), bits),
            splat(63)),
        splat(1));
    corrected = builder.CreateAdd(bits, direction);
  } else {
    Value *nonzero = builder.CreateFCmpONE(error, zero);
    Value *sameDirection =
        builder.CreateICmpEQ(builder.CreateFCmpOGT(error, zero),
                             builder.CreateFCmpOGT(result, zero));
    Value *one = splat(1);
    Value *delta = builder.CreateSelect(
        sameDirection, one, ConstantExpr::getNeg(cast<Constant>(one)));
    corrected = builder.CreateAdd(bits, delta);
    changes = builder.CreateAnd(tie, nonzero);
  }
  builder.CreateRet(builder.CreateBitCast(
      builder.CreateSelect(changes, corrected, bits), pairType));
  return fixup;
}

void markFMAFixups(Module &module) {
  for (const char *name :
       {"fix_pair_fma_rounding", "fix_pair_fma_rounding_jit64",
        "fix_scalar_fma_rounding"})
    if (Function *fixup = module.getFunction(name))
      fixup->addFnAttr(kHostFMAAttribute);
}

} // namespace dolllvm
