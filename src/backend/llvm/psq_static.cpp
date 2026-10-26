#include "backend/llvm/emitter.h"
#include "backend/llvm/psq_convert.h"
#include "cpu/cpu.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>

#include <utility>

namespace dolllvm {

using namespace llvm;

static u32 psqTypeWidth(u32 type) {
  return type == 0 ? 4 : (type == 5 || type == 7 ? 2 : 1);
}

// Jit64's float psq_l: the two singles go through cvtps2pd as they are, so a
// denormal is zero while the host treats denormals as zero (FPSCR[NI]) and a
// signalling NaN is quieted. The single view of the register is that double
// converted back, which is what a later psq_st of it stores under Jit64.
std::pair<Value *, Value *> FunctionEmitter::hostPSQFloats(Value *bits0,
                                                           Value *bits1) {
  Type *f32 = Type::getFloatTy(context_);
  Type *singles = FixedVectorType::get(f32, 2);
  Type *doubles = FixedVectorType::get(Type::getDoubleTy(context_), 2);
  Value *raw = builder_.CreateInsertElement(
      PoisonValue::get(singles), builder_.CreateBitCast(bits0, f32),
      uint64_t{0});
  raw = builder_.CreateInsertElement(
      raw,
      bits1 ? builder_.CreateBitCast(bits1, f32)
            : static_cast<Value *>(ConstantFP::get(f32, 1.0)),
      1u);
  Value *wide = builder_.CreateFPExt(raw, doubles, "psq.wide");
  return {roundPairToSingle(wide), wide};
}

Value *FunctionEmitter::emitKnownPSQ(const DolIRInstruction &inst, u32 type,
                                     s32 scale, Value *guard) {
  u32 reg = inst.immediate & 0xFFu;
  bool w = ((inst.immediate >> 8) & 1u) != 0;
  u32 gqr = (inst.immediate >> 9) & 7u;
  bool indexed = ((inst.immediate >> 12) & 1u) != 0;
  bool load = inst.aux == DOLIR_HELPER_PSQ_LOAD;
  u32 width = psqTypeWidth(type);
  Value *address = operand(inst, 0);
  Type *ptr = PointerType::getUnqual(context_);
  auto callee = module_.getOrInsertFunction(
      load ? "ppc_psq_load" : "ppc_psq_store",
      FunctionType::get(Type::getInt1Ty(context_),
                        {ptr, Type::getInt8Ty(context_),
                         Type::getInt32Ty(context_), Type::getInt1Ty(context_),
                         Type::getInt8Ty(context_), Type::getInt1Ty(context_),
                         Type::getInt32Ty(context_)},
                        false));
  Value *enabled = psqEnabled(indexed);
  if (type == 0) {
    enabled = builder_.CreateAnd(
        enabled,
        builder_.CreateICmpEQ(builder_.CreateAnd(address, builder_.getInt32(3)),
                              builder_.getInt32(0)));
  }
  if (guard)
    enabled = builder_.CreateAnd(enabled, guard);

  auto incomingRepresentations = fp_rep_;
  auto incomingExactSingles = fp_exact_single_;
  auto incomingDenormalSafety = fp_denormal_safe_;
  auto incomingTrackViews = fp_track_view_;
  auto incomingValueClasses = fp_value_class_;
  auto incomingKnownState = known_state_;
  bool incomingFPAvailable = fp_available_checked_;
  BasicBlock *fast = BasicBlock::Create(context_, "psq_known", function_);
  BasicBlock *slow =
      BasicBlock::Create(context_, "psq_known_helper", function_);
  BasicBlock *join = BasicBlock::Create(context_, "psq_known_join", function_);
  builder_.CreateCondBr(enabled, fast, slow,
                        MDBuilder(context_).createBranchWeights(1000, 1));

  builder_.SetInsertPoint(fast);
  if (load && type == 0 && jit64_paired_fp_) {
    Type *i32 = Type::getInt32Ty(context_);
    Value *bits0 = nullptr;
    Value *bits1 = nullptr;
    if (w) {
      bits0 = emitGuestLoad(address, i32, 4, false);
    } else {
      // Both singles in one eight-byte access: one range check, not two.
      Value *both =
          emitGuestLoad(address, Type::getInt64Ty(context_), 8, false);
      bits0 = builder_.CreateTrunc(builder_.CreateLShr(both, 32), i32);
      bits1 = builder_.CreateTrunc(both, i32);
    }
    auto converted = hostPSQFloats(bits0, bits1);
    builder_.CreateStore(converted.first, pair_f32_[reg]);
    builder_.CreateStore(
        builder_.CreateExtractElement(converted.second, uint64_t{0}),
        state_[DOLIR_STATE_FPR0 + reg]);
    builder_.CreateStore(builder_.CreateExtractElement(converted.second, 1u),
                         state_[DOLIR_STATE_PS1_0 + reg]);
  } else if (load) {
    auto loadLane = [&](Value *laneAddress) {
      if (type == 0) {
        Value *bits =
            emitGuestLoad(laneAddress, Type::getInt32Ty(context_), 4, false);
        return std::pair<Value *, Value *>(
            builder_.CreateBitCast(bits, Type::getFloatTy(context_)),
            extendPSQFloat(builder_, module_, bits));
      }
      Type *integerType = IntegerType::get(context_, width * 8);
      Value *integer = emitGuestLoad(laneAddress, integerType, width, false);
      Value *single = dequantizePSQInteger(
          builder_, integer, builder_.getInt32(static_cast<u32>(scale)),
          type == 6 || type == 7);
      return std::pair<Value *, Value *>(
          single, builder_.CreateFPExt(single, Type::getDoubleTy(context_)));
    };
    auto lane0 = loadLane(address);
    std::pair<Value *, Value *> lane1(
        ConstantFP::get(Type::getFloatTy(context_), 1.0),
        ConstantFP::get(Type::getDoubleTy(context_), 1.0));
    if (!w)
      lane1 = loadLane(builder_.CreateAdd(address, builder_.getInt32(width)));
    Type *pairType = FixedVectorType::get(Type::getFloatTy(context_), 2);
    Value *pair = PoisonValue::get(pairType);
    pair = builder_.CreateInsertElement(pair, lane0.first, uint64_t{0});
    pair = builder_.CreateInsertElement(pair, lane1.first, 1u);
    builder_.CreateStore(pair, pair_f32_[reg]);
    builder_.CreateStore(lane0.second, state_[DOLIR_STATE_FPR0 + reg]);
    builder_.CreateStore(lane1.second, state_[DOLIR_STATE_PS1_0 + reg]);
  } else {
    Value *pair = fp_rep_[reg] == FPRepresentation::PairF32 &&
                          (type == 0 || fp_denormal_safe_[reg]) &&
                          storeViewAllowed(reg)
                      ? pairF32(reg)
                      : pairF64(reg);
    auto storeLane = [&](Value *laneAddress, u32 lane) {
      Value *value = builder_.CreateExtractElement(pair, lane);
      // Jit64 stores what cvtpd2ps gives; the default path also turns a
      // denormal single into zero.
      Value *bits =
          type == 0 && jit64_paired_fp_
              ? builder_.CreateBitCast(value->getType()->isFloatTy()
                                           ? value
                                           : roundScalarToSingle(value),
                                       Type::getInt32Ty(context_))
          : type == 0
              ? truncatePSQFloat(builder_, value)
              : quantizePSQFloat(builder_, value,
                                 builder_.getInt32(static_cast<u32>(scale)),
                                 type == 6   ? -128
                                 : type == 7 ? -32768
                                             : 0,
                                 type == 4   ? 255
                                 : type == 5 ? 65535
                                 : type == 6 ? 127
                                             : 32767);
      emitGuestStore(laneAddress, bits, width);
    };
    if (type == 0 && jit64_paired_fp_ && !w) {
      // Both singles in one eight-byte access, as in the load.
      Type *i32 = Type::getInt32Ty(context_);
      Type *i64 = Type::getInt64Ty(context_);
      Value *singles = pair->getType()->getScalarType()->isFloatTy()
                           ? pair
                           : roundPairToSingle(pair);
      Value *lanes = builder_.CreateBitCast(singles,
                                            FixedVectorType::get(i32, 2));
      Value *first = builder_.CreateZExt(
          builder_.CreateExtractElement(lanes, uint64_t{0}), i64);
      Value *second =
          builder_.CreateZExt(builder_.CreateExtractElement(lanes, 1u), i64);
      emitGuestStore(address,
                     builder_.CreateOr(builder_.CreateShl(first, 32), second),
                     8);
    } else {
      storeLane(address, 0);
      if (!w)
        storeLane(builder_.CreateAdd(address, builder_.getInt32(width)), 1);
    }
  }
  builder_.CreateBr(join);
  BasicBlock *fastEnd = builder_.GetInsertBlock();

  builder_.SetInsertPoint(slow);
  Value *success = nullptr;
  BasicBlock *resume = nullptr;
  if (modern_runtime_) {
    sideExit(inst.guest_pc, 2);
  } else {
    materialize(inst.guest_pc);
    CallInst *helper = builder_.CreateCall(
        callee, {ctx_, builder_.getInt8(reg), address, builder_.getInt1(w),
                 builder_.getInt8(gqr), builder_.getInt1(indexed),
                 builder_.getInt32(inst.guest_pc)});
    tagRuntimeCall(helper, load ? RegisterWrites::FloatingPoint
                                : RegisterWrites::None);
    success = helper;
    resume = BasicBlock::Create(context_, "psq_known_resume", function_);
    BasicBlock *failed =
        BasicBlock::Create(context_, "psq_known_exit", function_);
    builder_.CreateCondBr(success, resume, failed);
    builder_.SetInsertPoint(failed);
    returnFromBody();
    builder_.SetInsertPoint(resume);
    if (!sync_changed_) {
      reloadUsedState();
    } else if (load) {
      reloadState(static_cast<DolIRStateSlot>(DOLIR_STATE_FPR0 + reg));
      reloadState(static_cast<DolIRStateSlot>(DOLIR_STATE_PS1_0 + reg));
      fp_rep_[reg] = FPRepresentation::Raw;
    }
    if (load)
      builder_.CreateStore(roundPairToSingle(pairF64(reg)), pair_f32_[reg]);
    builder_.CreateBr(join);
  }

  builder_.SetInsertPoint(join);
  Value *result = ConstantInt::getTrue(context_);
  if (!modern_runtime_) {
    PHINode *merged = builder_.CreatePHI(Type::getInt1Ty(context_), 2);
    merged->addIncoming(result, fastEnd);
    merged->addIncoming(success, resume);
    result = merged;
  }
  fp_rep_ = incomingRepresentations;
  fp_exact_single_ = incomingExactSingles;
  fp_denormal_safe_ = incomingDenormalSafety;
  fp_track_view_ = incomingTrackViews;
  fp_value_class_ = incomingValueClasses;
  known_state_ = incomingKnownState;
  fp_available_checked_ = incomingFPAvailable;
  if (load) {
    markWritten(static_cast<DolIRStateSlot>(DOLIR_STATE_FPR0 + reg));
    markWritten(static_cast<DolIRStateSlot>(DOLIR_STATE_PS1_0 + reg));
    fp_rep_[reg] = FPRepresentation::PairF32;
    fp_exact_single_[reg] = true;
    fp_track_view_[reg] = false;
    fp_denormal_safe_[reg] = type != 0 || jit64_paired_fp_;
    fp_single_canonical_[reg] = type != 0 || jit64_paired_fp_;
    fp_ps0_float_[reg] = nullptr;
    fp_ps1_float_[reg] = nullptr;
    fp_value_class_[reg] =
        type == 0 ? FPValueClass::Unknown : FPValueClass::NormalOrZero;
  }
  psq_indexed_proven_ = true;
  if (!indexed)
    psq_direct_proven_ = true;
  return result;
}

} // namespace dolllvm
