#include "backend/llvm/emitter.h"
#include "cpu/cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

namespace dolllvm {

using namespace llvm;

BasicBlock *FunctionEmitter::directDestination(const DolIRTerminator &term,
                                               u32 slot) {
  if (term.targets[slot] != DOLIR_NO_BLOCK) {
    if (source_.blocks[term.targets[slot]].terminator.kind ==
        DOLIR_TERM_FALLBACK)
      return fallbackEdge(term.target_addresses[slot]);
    return blocks_[term.targets[slot]];
  }
  return externalDestination(term, slot);
}

const DolLLVMFunctionRange *FunctionEmitter::rangeFor(u32 address) const {
  for (u32 i = 0; i < range_count_; i++)
    if (address >= ranges_[i].start && address < ranges_[i].end)
      return &ranges_[i];
  return nullptr;
}

BasicBlock *FunctionEmitter::externalDestination(const DolIRTerminator &term,
                                                 u32 slot) {
  u32 target = term.target_addresses[slot];
  const DolLLVMFunctionRange *range = rangeFor(target);
  if (!range)
    return nullptr;
  const bool nativeTarget =
      (range->abi_flags & DOLLLVM_FUNCTION_ABI_NATIVE) != 0;
  if (native_abi_ && !nativeTarget)
    return nullptr;
  if (!native_abi_ && nativeTarget && cold_escapes_)
    return nullptr;
  if (nativeTarget) {
    for (u32 state = 0; state < DOLIR_STATE_COUNT; state++) {
      auto stateSlot = static_cast<DolIRStateSlot>(state);
      if (stateInput(range, stateSlot) && !state_[state])
        return nullptr;
    }
  }
  BasicBlock *callBlock = BasicBlock::Create(
      context_, term.linked ? "direct_call" : "direct_tail", function_);
  IRBuilderBase::InsertPoint saved = builder_.saveIP();
  builder_.SetInsertPoint(callBlock);
  emitBudgetGuard(target);
  char name[64];
  snprintf(name, sizeof(name), "func_%08X_budget", range->start);
  const std::string targetName = symbolName(name);
  if (!nativeTarget)
    syncDirtyState();
  Value *callDepth = nullptr;
  if (modern_runtime_ && nativeTarget) {
    callDepth = builder_.CreateLoad(Type::getInt64Ty(context_), guard_steps_);
    BasicBlock *invoke = BasicBlock::Create(context_, "call_depth_ok", function_);
    BasicBlock *yield = BasicBlock::Create(context_, "call_depth_exit", function_);
    builder_.CreateCondBr(
        builder_.CreateICmpULT(callDepth, builder_.getInt64(64)), invoke,
        yield);
    builder_.SetInsertPoint(yield);
    sideExit(target);
    builder_.SetInsertPoint(invoke);
    builder_.CreateStore(builder_.CreateAdd(callDepth, builder_.getInt64(1)),
                         guard_steps_);
  }
  const bool cyclesInResult = nativeTarget && nativeCyclesInResult(range);
  if (!cyclesInResult)
    flushCallCounters(true);
  // From a closed body to another closed function's first instruction: the
  // closed protocol. Any other target gets the ordinary one.
  const bool closedCall = !nativeTarget && closedTarget(range, target);
  FunctionCallee callee =
      closedCall
          ? closedBody(range->start)
          : module_.getOrInsertFunction(targetName, bodyFunctionType(range));
  if (auto *calleeFunction = dyn_cast<Function>(callee.getCallee())) {
    calleeFunction->setCallingConv(bodyCallingConvention());
    calleeFunction->setVisibility(GlobalValue::HiddenVisibility);
    calleeFunction->setDSOLocal(true);
  }
  Value *calleeReturnPC =
      term.linked ? static_cast<Value *>(builder_.getInt32(term.guest_pc + 4u))
                  : return_pc_;
  Value *control = builder_.CreateOr(
      builder_.CreateZExt(builder_.getInt32(target),
                          Type::getInt64Ty(context_)),
      builder_.CreateShl(
          builder_.CreateZExt(calleeReturnPC, Type::getInt64Ty(context_)),
          builder_.getInt64(32)));
  SmallVector<Value *, 32> arguments = {ctx_};
  if (modern_runtime_)
    arguments.push_back(state_interface_);
  if (!closedCall)
    arguments.push_back(chain_);
  arguments.push_back(control);
  if (register_counters_) {
    arguments.push_back(loadCycles());
    arguments.push_back(calleeGuard(term.linked));
  }
  if (nativeTarget) {
    if (cyclesInResult)
      arguments.push_back(loadCycles());
    for (u32 state = 0; state < DOLIR_STATE_COUNT; state++) {
      auto stateSlot = static_cast<DolIRStateSlot>(state);
      if (stateInput(range, stateSlot))
        arguments.push_back(stateValue(stateSlot));
    }
  }
  CallInst *nativeCall = builder_.CreateCall(callee, arguments);
  nativeCall->setCallingConv(bodyCallingConvention());
  tagRuntimeCall(nativeCall, RegisterWrites::Any);
  if (nativeTarget) {
    if (callDepth)
      builder_.CreateStore(callDepth, guard_steps_);
    if (!cyclesInResult)
      reloadCallCounters();
    acceptNativeResult(nativeCall, range);
    if (cold_escapes_) {
      if (!term.linked) {
        if (native_abi_) {
          returnNative(return_pc_);
        } else {
          materialize(return_pc_);
          returnFromBody();
        }
        builder_.restoreIP(saved);
        return callBlock;
      }
      const u32 continuation = term.guest_pc + 4u;
      const bool local = continuation >= source_.guest_start &&
                         continuation < source_.guest_end &&
                         ((continuation - source_.guest_start) & 3u) == 0;
      const u32 continuationBlock =
          local ? (continuation - source_.guest_start) / 4u : 0u;
      if (!local || continuationBlock >= blocks_.size()) {
        sideExit(continuation);
      } else {
        builder_.CreateBr(blocks_[continuationBlock]);
      }
      builder_.restoreIP(saved);
      return callBlock;
    }
    Value *returnedPC = nativeResultPC(nativeCall);
    Value *continues = nativeResultContinues(nativeCall);
    if (!term.linked) {
      if (native_abi_) {
        BasicBlock *forward =
            BasicBlock::Create(context_, "tail_forward", function_);
        BasicBlock *stopped =
            BasicBlock::Create(context_, "tail_materialize", function_);
        builder_.CreateCondBr(continues, forward, stopped);
        builder_.SetInsertPoint(forward);
        returnNative(returnedPC);
        builder_.SetInsertPoint(stopped);
        materialize(returnedPC);
        returnFromBody();
      } else {
        materialize(returnedPC);
        returnFromBody();
      }
      builder_.restoreIP(saved);
      return callBlock;
    }
    const u32 continuation = term.guest_pc + 4u;
    const bool local = continuation >= source_.guest_start &&
                       continuation < source_.guest_end &&
                       ((continuation - source_.guest_start) & 3u) == 0;
    const u32 continuationBlock =
        local ? (continuation - source_.guest_start) / 4u : 0u;
    BasicBlock *resume = BasicBlock::Create(context_, "call_resume", function_);
    BasicBlock *mismatch =
        BasicBlock::Create(context_, "call_mismatch", function_);
    Value *matches = builder_.CreateAnd(
        continues,
        builder_.CreateICmpEQ(returnedPC, builder_.getInt32(continuation)));
    if (!local || continuationBlock >= blocks_.size())
      matches = builder_.getFalse();
    builder_.CreateCondBr(matches, resume, mismatch);
    builder_.SetInsertPoint(mismatch);
    materialize(returnedPC);
    returnFromBody();
    builder_.SetInsertPoint(resume);
    builder_.CreateBr(blocks_[continuationBlock]);
    builder_.restoreIP(saved);
    return callBlock;
  }
  if (!term.linked) {
    if (closed_ && !closedCall) {
      // An ordinary body ran in the closed body's place. It has stored the
      // PC it returned to: the caller's return address is the normal return,
      // anything else an escape.
      BasicBlock *normal =
          BasicBlock::Create(context_, "tail_returned", function_);
      BasicBlock *away = BasicBlock::Create(context_, "tail_away", function_);
      Value *returnedPC =
          loadOffset(Type::getInt32Ty(context_), offsetof(CPUState, pc));
      builder_.CreateCondBr(builder_.CreateICmpEQ(returnedPC, return_pc_),
                            normal, away);
      builder_.SetInsertPoint(normal);
      builder_.CreateRet(nativeCall);
      builder_.SetInsertPoint(away);
      builder_.CreateRet(withEscape(nativeCall));
      builder_.restoreIP(saved);
      return callBlock;
    }
    nativeCall->setTailCallKind(CallInst::TCK_MustTail);
    if (register_counters_)
      builder_.CreateRet(nativeCall);
    else
      builder_.CreateRetVoid();
    builder_.restoreIP(saved);
    return callBlock;
  }
  u32 continuation = term.guest_pc + 4u;
  u32 continuationBlock = 0;
  bool local = continuation >= source_.guest_start &&
               continuation < source_.guest_end &&
               ((continuation - source_.guest_start) & 3u) == 0;
  if (local)
    continuationBlock = (continuation - source_.guest_start) / 4u;
  BasicBlock *resume = BasicBlock::Create(context_, "call_resume", function_);
  BasicBlock *mismatch =
      BasicBlock::Create(context_, "call_mismatch", function_);
  if (closedCall) {
    // The callee either returned to the return address it was given or
    // escaped, and an escape goes up as it is.
    builder_.CreateCondBr(closedEscaped(nativeCall), mismatch, resume,
                          MDBuilder(context_).createBranchWeights(1, 2000));
    builder_.SetInsertPoint(mismatch);
    builder_.CreateRet(nativeCall);
    builder_.SetInsertPoint(resume);
    if (!local || continuationBlock >= blocks_.size()) {
      storeContext(DOLIR_STATE_PC, builder_.getInt32(continuation));
      builder_.CreateRet(withEscape(nativeCall));
    } else {
      acceptCounters(nativeCall);
      reloadUsedState();
      builder_.CreateBr(blocks_[continuationBlock]);
    }
    builder_.restoreIP(saved);
    return callBlock;
  }
  Value *returnedPC =
      loadOffset(Type::getInt32Ty(context_), offsetof(CPUState, pc));
  builder_.CreateCondBr(
      builder_.CreateICmpEQ(returnedPC, builder_.getInt32(continuation)),
      resume, mismatch);
  // A callee that came back somewhere else has set the PC and the counters
  // for whoever is above; they pass through unchanged (from a closed body,
  // as an escape).
  auto passThrough = [&] { passResultUp(nativeCall); };
  builder_.SetInsertPoint(mismatch);
  passThrough();
  builder_.SetInsertPoint(resume);
  if (!local || continuationBlock >= blocks_.size()) {
    passThrough();
  } else {
    if (register_counters_)
      acceptCounters(nativeCall);
    reloadCallCounters();
    reloadUsedState();
    builder_.CreateBr(blocks_[continuationBlock]);
  }
  builder_.restoreIP(saved);
  return callBlock;
}

// True when the call site at `pc` is listed in DOLRECOMP_INDIRECT_CALL_EXITS.
static bool indirectCallKeepsExit(u32 pc) {
  static const std::vector<u32> listed = [] {
    std::vector<u32> sites;
    const char *text = std::getenv("DOLRECOMP_INDIRECT_CALL_EXITS");
    while (text && *text) {
      char *end = nullptr;
      const unsigned long value = std::strtoul(text, &end, 16);
      if (end == text)
        break;
      sites.push_back(static_cast<u32>(value));
      text = *end == ',' ? end + 1 : end;
    }
    return sites;
  }();
  return std::find(listed.begin(), listed.end(), pc) != listed.end();
}

// DOLRECOMP_INDIRECT_GUARDS: see emitter.h. The direct call is the one a `bl`
// to the address would get, so everything a direct call relies on holds: the
// callee tests its own entry byte, and a callee that returns somewhere else
// passes its result up.
void FunctionEmitter::emitIndirectGuards(const DolIRTerminator &term,
                                         Value *target) {
  static const std::vector<std::pair<u32, u32>> listed = [] {
    std::vector<std::pair<u32, u32>> pairs;
    const char *text = std::getenv("DOLRECOMP_INDIRECT_GUARDS");
    while (text && *text) {
      char *end = nullptr;
      const unsigned long site = std::strtoul(text, &end, 16);
      if (end == text || *end != ':')
        break;
      do {
        text = end + 1;
        const unsigned long callee = std::strtoul(text, &end, 16);
        if (end == text)
          return pairs;
        pairs.emplace_back(static_cast<u32>(site), static_cast<u32>(callee));
      } while (*end == '+');
      text = *end == ',' ? end + 1 : end;
    }
    return pairs;
  }();
  if (indirectCallKeepsExit(term.guest_pc))
    return;
  for (const auto &[site, callee] : listed) {
    if (site != term.guest_pc)
      continue;
    DolIRTerminator direct = term;
    direct.kind = DOLIR_TERM_BRANCH;
    direct.targets[0] = DOLIR_NO_BLOCK;
    direct.target_addresses[0] = callee;
    BasicBlock *call = externalDestination(direct, 0);
    if (!call)
      continue;
    BasicBlock *other =
        BasicBlock::Create(context_, "indirect_other", function_);
    builder_.CreateCondBr(
        builder_.CreateICmpEQ(target, builder_.getInt32(callee)), call, other);
    builder_.SetInsertPoint(other);
  }
}

// bctrl/blrl as a call inside generated code (DOLRECOMP_INDIRECT_CALLS=native).
//
// Without this an indirect call leaves for the run loop, which dispatches the
// target. That unwinds every native frame, so the callee's return and each
// return above it is a dispatch of its own, entered through the resume
// switch with all registers reloaded.
//
// The target is looked up in two tables the module defines once:
// dolrecomp_native_index maps (address - dolrecomp_native_base) / 4 to a
// function number for every address inside a recompiled function, and
// dolrecomp_native_entries maps that number to the function's body, 0 to
// null. A target outside the tables, a null entry or a spent budget takes
// the exit the caller emits after this, exactly as before. The call itself
// follows the direct call's protocol: state written back, counters handed
// over, the callee's returned PC compared with the return address.
//
// The entries are the host's to change while the game runs: a host that
// checks guest code against the module before it dispatches to it keeps an
// entry null until that function is verified and clears it when the code is
// invalidated, so the entry is read at every call and never assumed
// constant. Addresses are the ones the module was linked for; a module with
// relocatable sections needs the host's mapping in front of this lookup.
// Interception needs nothing here: the callee asks the host at its own
// entry, whichever kind of call reached it.
void FunctionEmitter::emitNativeIndirectCall(const DolIRTerminator &term,
                                             Value *target) {
  const u32 continuation = term.guest_pc + 4u;
  const bool local = continuation >= source_.guest_start &&
                     continuation < source_.guest_end &&
                     ((continuation - source_.guest_start) & 3u) == 0;
  const u32 continuationBlock =
      local ? (continuation - source_.guest_start) / 4u : 0u;
  if (!local || continuationBlock >= blocks_.size() ||
      indirectCallKeepsExit(term.guest_pc))
    return;

  Type *i16 = Type::getInt16Ty(context_);
  Type *i32 = Type::getInt32Ty(context_);
  Type *i64 = Type::getInt64Ty(context_);
  Type *pointer = PointerType::getUnqual(context_);
  auto table = [&](const char *name, Type *type, bool constant = true) {
    auto *global = cast<GlobalVariable>(module_.getOrInsertGlobal(name, type));
    global->setConstant(constant);
    global->setVisibility(GlobalValue::HiddenVisibility);
    global->setDSOLocal(true);
    return global;
  };
  GlobalVariable *base = table("dolrecomp_native_base", i32);
  GlobalVariable *count = table("dolrecomp_native_count", i32);
  GlobalVariable *index =
      table("dolrecomp_native_index", ArrayType::get(i16, 0));
  GlobalVariable *entries =
      table("dolrecomp_native_entries", ArrayType::get(pointer, 0), false);

  BasicBlock *exit = BasicBlock::Create(context_, "indirect_exit", function_);
  BasicBlock *lookup =
      BasicBlock::Create(context_, "indirect_lookup", function_);
  BasicBlock *found = BasicBlock::Create(context_, "indirect_found", function_);
  BasicBlock *call = BasicBlock::Create(context_, "indirect_call", function_);
  MDNode *likely = MDBuilder(context_).createBranchWeights(2000, 1);
  MDNode *unlikely = MDBuilder(context_).createBranchWeights(1, 2000);

  Value *offset = builder_.CreateSub(target, builder_.CreateLoad(i32, base));
  Value *slot = builder_.CreateLShr(offset, builder_.getInt32(2));
  Value *inTable = builder_.CreateAnd(
      builder_.CreateICmpEQ(builder_.CreateAnd(offset, builder_.getInt32(3)),
                            builder_.getInt32(0)),
      builder_.CreateICmpULT(slot, builder_.CreateLoad(i32, count)));
  builder_.CreateCondBr(inTable, lookup, exit, likely);

  builder_.SetInsertPoint(lookup);
  Value *number = builder_.CreateLoad(
      i16, builder_.CreateInBoundsGEP(i16, index,
                                      builder_.CreateZExt(slot, i64)));
  Value *body = builder_.CreateLoad(
      pointer, builder_.CreateInBoundsGEP(pointer, entries,
                                          builder_.CreateZExt(number, i64)),
      "indirect.body");
  builder_.CreateCondBr(builder_.CreateIsNull(body), exit, found, unlikely);

  builder_.SetInsertPoint(found);
  builder_.CreateCondBr(budgetExhausted(), exit, call, unlikely);

  builder_.SetInsertPoint(call);
  countHere(term.guest_pc, 3, 0);
  syncDirtyState();
  flushCallCounters(true);
  Value *control = builder_.CreateOr(
      builder_.CreateZExt(target, i64),
      builder_.getInt64(static_cast<u64>(continuation) << 32));
  SmallVector<Value *, 5> arguments = {ctx_, chain_, control};
  if (register_counters_) {
    arguments.push_back(loadCycles());
    arguments.push_back(calleeGuard(true));
  }
  CallInst *nativeCall =
      builder_.CreateCall(bodyFunctionType(nullptr), body, arguments);
  nativeCall->setCallingConv(bodyCallingConvention());
  tagRuntimeCall(nativeCall, RegisterWrites::Any);
  BasicBlock *resume = BasicBlock::Create(context_, "call_resume", function_);
  BasicBlock *mismatch =
      BasicBlock::Create(context_, "call_mismatch", function_);
  Value *returnedPC = loadOffset(i32, offsetof(CPUState, pc));
  builder_.CreateCondBr(
      builder_.CreateICmpEQ(returnedPC, builder_.getInt32(continuation)),
      resume, mismatch);
  builder_.SetInsertPoint(mismatch);
  passResultUp(nativeCall);
  builder_.SetInsertPoint(resume);
  if (register_counters_)
    acceptCounters(nativeCall);
  reloadCallCounters();
  reloadUsedState();
  builder_.CreateBr(blocks_[continuationBlock]);

  builder_.SetInsertPoint(exit);
}

BasicBlock *FunctionEmitter::exitDestination(u32 pc) {
  BasicBlock *exit = BasicBlock::Create(context_, "side_exit", function_);
  IRBuilderBase::InsertPoint saved = builder_.saveIP();
  builder_.SetInsertPoint(exit);
  sideExit(pc);
  builder_.restoreIP(saved);
  return exit;
}

BasicBlock *
FunctionEmitter::fallbackDestination(const DolIRTerminator &terminator) {
  return fallbackEdge(terminator.guest_pc);
}

BasicBlock *FunctionEmitter::fallbackEdge(u32 pc) {
  BasicBlock *edge = BasicBlock::Create(context_, "fallback_edge", function_);
  IRBuilderBase::InsertPoint saved = builder_.saveIP();
  builder_.SetInsertPoint(edge);
  fallback_pc_->addIncoming(builder_.getInt32(pc), edge);
  builder_.CreateBr(fallback_block_);
  builder_.restoreIP(saved);
  return edge;
}

} // namespace dolllvm
