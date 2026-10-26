#include "backend/llvm/emitter.h"
#include "backend/llvm/psq_convert.h"
#include "cpu/cpu.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>

#include <array>
#include <cstdlib>
#include <utility>

namespace dolllvm {

using namespace llvm;

static bool supportedPSQType(u32 type) {
  return type == 0 || type == 4 || type == 5 || type == 6 || type == 7;
}

// DOLRECOMP_PSQ_GQR="0:0,2:00040004": the value a title keeps in a GQR. A
// psq_l/psq_st through that GQR is emitted for exactly that type and scale
// behind a test of the register; any other value takes the runtime helper.
// Without it, a GQR the function does not set itself costs every access an
// inline dispatch over all five types.
static bool expectedGQR(u32 index, u32 *value) {
  static const std::array<std::pair<bool, u32>, 8> expected = [] {
    std::array<std::pair<bool, u32>, 8> parsed{};
    const char *text = std::getenv("DOLRECOMP_PSQ_GQR");
    for (const char *p = text ? text : ""; *p;) {
      char *end = nullptr;
      const unsigned long gqr = std::strtoul(p, &end, 10);
      if (end == p || *end != ':' || gqr > 7)
        break;
      p = end + 1;
      const unsigned long bits = std::strtoul(p, &end, 16);
      if (end == p)
        break;
      parsed[gqr] = {true, static_cast<u32>(bits)};
      p = end;
      while (*p == ',' || *p == ' ')
        ++p;
    }
    return parsed;
  }();
  if (!expected[index].first)
    return false;
  *value = expected[index].second;
  return true;
}

Value *FunctionEmitter::emitPSQ(const DolIRInstruction &inst) {
  if (!lazy_fprf_)
    materializeFPRF();
  // The helper paths below go through CPUState and come back; a pending FPRF
  // is written there for the helper and is still pending afterwards.
  struct KeepPending {
    FunctionEmitter &emitter;
    llvm::Value *fprf;
    llvm::Value *fpcc;
    ~KeepPending() {
      if (emitter.lazy_fprf_) {
        emitter.pending_fprf_ = fprf;
        emitter.pending_fpcc_ = fpcc;
      }
    }
  } keepPending{*this, pending_fprf_, pending_fpcc_};
  u32 reg = inst.immediate & 0xFFu;
  bool w = ((inst.immediate >> 8) & 1u) != 0;
  u32 gqr = (inst.immediate >> 9) & 7u;
  bool indexed = ((inst.immediate >> 12) & 1u) != 0;
  bool load = inst.aux == DOLIR_HELPER_PSQ_LOAD;
  Type *ptr = PointerType::getUnqual(context_);
  auto callee = module_.getOrInsertFunction(
      load ? "ppc_psq_load" : "ppc_psq_store",
      FunctionType::get(Type::getInt1Ty(context_),
                        {ptr, Type::getInt8Ty(context_),
                         Type::getInt32Ty(context_), Type::getInt1Ty(context_),
                         Type::getInt8Ty(context_), Type::getInt1Ty(context_),
                         Type::getInt32Ty(context_)},
                        false));
  Value *address = operand(inst, 0);
  Value *typeValue = gqrType(gqr, load);
  Value *scaleValue = gqrScale(gqr, load);
  auto *knownType = dyn_cast<ConstantInt>(typeValue);
  auto *knownScale = dyn_cast<ConstantInt>(scaleValue);
  if (knownType && knownScale && supportedPSQType(knownType->getZExtValue()))
    return emitKnownPSQ(inst, static_cast<u32>(knownType->getZExtValue()),
                        static_cast<s32>(knownScale->getSExtValue()));

  u32 expected = 0;
  if (!modern_runtime_ && expectedGQR(gqr, &expected)) {
    const u32 field = load ? expected >> 16 : expected & 0xFFFFu;
    const u32 type = field & 7u;
    const s32 scale = static_cast<s32>(((field >> 8) & 0x3Fu) << 26) >> 26;
    if (supportedPSQType(type)) {
      // The register as it is now, not the copy taken at function entry: a
      // callee may have changed it.
      Value *current = builder_.CreateLoad(
          Type::getInt32Ty(context_), state_[DOLIR_STATE_GQR0 + gqr]);
      return emitKnownPSQ(
          inst, type, scale,
          builder_.CreateICmpEQ(current, builder_.getInt32(expected)));
    }
  }

  if (load)
    return emitPSQLoad(inst, address, typeValue, scaleValue);

  auto incomingRepresentations = fp_rep_;
  auto incomingExactSingles = fp_exact_single_;
  auto incomingDenormalSafety = fp_denormal_safe_;
  auto incomingTrackViews = fp_track_view_;
  auto incomingValueClasses = fp_value_class_;
  auto incomingKnownState = known_state_;
  bool incomingFPAvailable = fp_available_checked_;
  Value *scale = scaleValue;
  Value *enabled = psqEnabled(indexed);

  BasicBlock *dispatch =
      BasicBlock::Create(context_, "psq_store_dispatch", function_);
  BasicBlock *slow =
      BasicBlock::Create(context_, "psq_store_helper", function_);
  BasicBlock *join = BasicBlock::Create(context_, "psq_store_join", function_);
  Value *supported = builder_.CreateOr(
      builder_.CreateOr(builder_.CreateICmpEQ(typeValue, builder_.getInt32(0)),
                        builder_.CreateICmpEQ(typeValue, builder_.getInt32(4))),
      builder_.CreateOr(
          builder_.CreateICmpEQ(typeValue, builder_.getInt32(5)),
          builder_.CreateOr(
              builder_.CreateICmpEQ(typeValue, builder_.getInt32(6)),
              builder_.CreateICmpEQ(typeValue, builder_.getInt32(7)))));
  Value *alignmentValid = builder_.CreateOr(
      builder_.CreateICmpNE(typeValue, builder_.getInt32(0)),
      builder_.CreateICmpEQ(builder_.CreateAnd(address, builder_.getInt32(3)),
                            builder_.getInt32(0)));
  builder_.CreateCondBr(
      builder_.CreateAnd(builder_.CreateAnd(enabled, supported),
                         alignmentValid),
      dispatch, slow);

  BasicBlock *f32Block =
      BasicBlock::Create(context_, "psq_store_f32", function_);
  BasicBlock *u8Block = BasicBlock::Create(context_, "psq_store_u8", function_);
  BasicBlock *u16Block =
      BasicBlock::Create(context_, "psq_store_u16", function_);
  BasicBlock *s8Block = BasicBlock::Create(context_, "psq_store_s8", function_);
  BasicBlock *s16Block =
      BasicBlock::Create(context_, "psq_store_s16", function_);
  builder_.SetInsertPoint(dispatch);
  SwitchInst *typeSwitch = builder_.CreateSwitch(typeValue, slow, 5);
  typeSwitch->addCase(builder_.getInt32(0), f32Block);
  typeSwitch->addCase(builder_.getInt32(4), u8Block);
  typeSwitch->addCase(builder_.getInt32(5), u16Block);
  typeSwitch->addCase(builder_.getInt32(6), s8Block);
  typeSwitch->addCase(builder_.getInt32(7), s16Block);

  SmallVector<BasicBlock *, 5> fastEnds;
  auto emitStores = [&](BasicBlock *block, u32 width, int minValue,
                        int maxValue, bool unquantized) {
    builder_.SetInsertPoint(block);
    Value *pair = fp_rep_[reg] == FPRepresentation::PairF32 &&
                          (unquantized || fp_denormal_safe_[reg]) &&
                          storeViewAllowed(reg)
                      ? pairF32(reg)
                      : pairF64(reg);
    Value *lane0 = builder_.CreateExtractElement(pair, uint64_t{0});
    Value *lane0Bits = unquantized ? truncatePSQFloat(builder_, lane0)
                                   : quantizePSQFloat(builder_, lane0, scale,
                                                      minValue, maxValue);
    emitGuestStore(address, lane0Bits, width);
    if (!w) {
      Value *lane1 = builder_.CreateExtractElement(pair, 1u);
      Value *lane1Bits = unquantized ? truncatePSQFloat(builder_, lane1)
                                     : quantizePSQFloat(builder_, lane1, scale,
                                                        minValue, maxValue);
      emitGuestStore(builder_.CreateAdd(address, builder_.getInt32(width)),
                     lane1Bits, width);
    }
    builder_.CreateBr(join);
    fastEnds.push_back(builder_.GetInsertBlock());
  };
  emitStores(f32Block, 4, 0, 0, true);
  emitStores(u8Block, 1, 0, 255, false);
  emitStores(u16Block, 2, 0, 65535, false);
  emitStores(s8Block, 1, -128, 127, false);
  emitStores(s16Block, 2, -32768, 32767, false);

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
    tagRuntimeCall(helper, RegisterWrites::None);
    success = helper;
    resume = BasicBlock::Create(context_, "psq_store_helper_resume", function_);
    BasicBlock *failed =
        BasicBlock::Create(context_, "psq_store_exit", function_);
    builder_.CreateCondBr(success, resume, failed);
    builder_.SetInsertPoint(failed);
    returnFromBody();
    builder_.SetInsertPoint(resume);
    // The store helper changes no guest register.
    if (!sync_changed_)
      reloadUsedState();
    builder_.CreateBr(join);
  }

  builder_.SetInsertPoint(join);
  Value *result = ConstantInt::getTrue(context_);
  if (!modern_runtime_) {
    PHINode *merged =
        builder_.CreatePHI(Type::getInt1Ty(context_), fastEnds.size() + 1);
    for (BasicBlock *fastEnd : fastEnds)
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
  psq_indexed_proven_ = true;
  if (!indexed)
    psq_direct_proven_ = true;
  return result;
}

} // namespace dolllvm
