#include "backend/llvm/emitter.h"
#include "backend/codegen_switches.h"
#include "cpu/cpu.h"

#include <cstdlib>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>

namespace dolllvm {

using namespace llvm;

void FunctionEmitter::chargeCycles(u32 cycles) {
  if (defer_cycles_) {
    deferred_cycles_ += cycles;
    return;
  }
  chargeCycles(ConstantInt::get(Type::getInt64Ty(context_), cycles));
}

void FunctionEmitter::chargeCycles(Value *cycles) {
  storeCycles(builder_.CreateAdd(loadCycles(), cycles));
}

// The counter as the guest sees it here: what is stored plus the cost of the
// instructions emitted since the stored value was last brought up to date.
Value *FunctionEmitter::loadCycles() {
  Value *stored = builder_.CreateLoad(Type::getInt64Ty(context_), cycles_);
  return deferred_cycles_
             ? builder_.CreateAdd(stored, builder_.getInt64(deferred_cycles_))
             : stored;
}

// Sets the counter to `cycles` as of here. The deferred cost stays pending
// for the code that follows (this may be a cold path the straight-line code
// does not go through), so the stored value is lower by that much.
void FunctionEmitter::storeCycles(Value *cycles) {
  if (deferred_cycles_)
    cycles = builder_.CreateSub(cycles, builder_.getInt64(deferred_cycles_));
  builder_.CreateStore(cycles, cycles_);
}

// At the end of a straight-line run: the stored counter is brought up to date.
void FunctionEmitter::flushDeferredCycles() {
  if (!deferred_cycles_)
    return;
  Value *stored = builder_.CreateLoad(Type::getInt64Ty(context_), cycles_);
  builder_.CreateStore(
      builder_.CreateAdd(stored, builder_.getInt64(deferred_cycles_)), cycles_);
  deferred_cycles_ = 0;
}

Value *FunctionEmitter::emitTimebaseRead() {
  if (!modern_runtime_)
    return stateValue(DOLIR_STATE_TIMEBASE);

  Type *i32 = Type::getInt32Ty(context_);
  Type *i64 = Type::getInt64Ty(context_);
  Type *pointer = PointerType::getUnqual(context_);
  StructType *servicesType =
      StructType::get(context_, {i32, i32, pointer, pointer, pointer, pointer});
  Value *services = runtimeField(11);
  Value *function = builder_.CreateLoad(
      pointer, builder_.CreateStructGEP(servicesType, services, 5));
  Value *serviceContext = builder_.CreateLoad(
      pointer, builder_.CreateStructGEP(servicesType, services, 2));
  Value *elapsed = builder_.CreateAdd(
      builder_.CreateLoad(i64, guard_cycles_local_),
      loadCycles());
  CallInst *result = builder_.CreateCall(
      FunctionType::get(i64, {pointer, i64}, false), function,
      {serviceContext, elapsed});
  result->addFnAttr(Attribute::NoUnwind);
  return result;
}

void FunctionEmitter::emitTimebaseWrite(const DolIRInstruction &inst) {
  if (!modern_runtime_) {
    builder_.CreateStore(operand(inst, 0), state_[DOLIR_STATE_TIMEBASE]);
    noteStateWrite(DOLIR_STATE_TIMEBASE, operand(inst, 0));
    return;
  }

  BasicBlock *resume =
      BasicBlock::Create(context_, "timebase_write_resume", function_);
  sideExit(inst.guest_pc, 2);
  builder_.SetInsertPoint(resume);
}

void FunctionEmitter::settleCycles() {
  if (modern_runtime_) {
    Value *cycles = loadCycles();
    builder_.CreateStore(cycles, pending_cycles_);
    storeCycles(builder_.getInt64(0));
    return;
  }
  Value *downcount =
      loadOffset(Type::getInt64Ty(context_), offsetof(CPUState, downcount));
  Value *cycles = loadCycles();
  builder_.CreateStore(builder_.CreateSub(downcount, cycles),
                       bytePtr(offsetof(CPUState, downcount)));
  Value *guard =
      builder_.CreateLoad(Type::getInt64Ty(context_), guard_cycles_local_);
  builder_.CreateStore(builder_.CreateAdd(guard, cycles), guard_cycles_local_);
  storeCycles(builder_.getInt64(0));
  if (!register_counters_)
    builder_.CreateStore(builder_.getInt64(0), pending_cycles_);
}

// The two counters as a body returns them and a call passes them
// (DOLRECOMP_CALL_COUNTERS=registers).
Value *FunctionEmitter::counterResult() {
  Type *i64 = Type::getInt64Ty(context_);
  Value *result = PoisonValue::get(StructType::get(context_, {i64, i64}));
  result = builder_.CreateInsertValue(result, loadCycles(), 0);
  return builder_.CreateInsertValue(
      result, builder_.CreateLoad(i64, guard_cycles_local_), 1);
}

// What a callee is given: this function's guard one call deeper.
Value *FunctionEmitter::calleeGuard(bool deeper) {
  Value *guard = builder_.CreateLoad(Type::getInt64Ty(context_),
                                     guard_cycles_local_);
  return deeper ? builder_.CreateAdd(guard, builder_.getInt64(u64{1} << 48))
                : guard;
}

// After a call: the callee's cycles and charged cycles, this function's own
// depth.
void FunctionEmitter::acceptCounters(Value *result) {
  Type *i64 = Type::getInt64Ty(context_);
  const u64 chargedMask = (u64{1} << 48) - 1;
  storeCycles(builder_.CreateExtractValue(result, 0));
  Value *own = builder_.CreateLoad(i64, guard_cycles_local_);
  builder_.CreateStore(
      builder_.CreateOr(
          builder_.CreateAnd(builder_.CreateExtractValue(result, 1),
                             builder_.getInt64(chargedMask)),
          builder_.CreateAnd(own, builder_.getInt64(~chargedMask))),
      guard_cycles_local_);
}

void FunctionEmitter::flushCallCounters(bool forceCycles) {
  if (register_counters_)
    return;
  if (forceCycles || !nativeCyclesInResult(abi_range_))
    builder_.CreateStore(loadCycles(), pending_cycles_);
  if (!native_abi_ || !cold_escapes_)
    builder_.CreateStore(
        builder_.CreateLoad(Type::getInt64Ty(context_), guard_cycles_local_),
        guard_cycles_);
}

void FunctionEmitter::reloadCallCounters() {
  if (register_counters_)
    return;
  storeCycles(
      builder_.CreateLoad(Type::getInt64Ty(context_), pending_cycles_));
  if (!native_abi_ || !cold_escapes_)
    builder_.CreateStore(
        builder_.CreateLoad(Type::getInt64Ty(context_), guard_cycles_),
        guard_cycles_local_);
}

// Guest cycles a dispatch may run before generated code hands control back to
// the run loop at the next call or loop head. The run loop is where the
// runtime's downcount, timebase and pending interrupts are serviced, so this
// bounds interrupt latency and how stale a timebase read can be; every exit
// also unwinds the native call stack, after which each guest return is a
// dispatch of its own. DOLRECOMP_BUDGET_CYCLES overrides the default.
//
// DOLRECOMP_BUDGET_CYCLES=runtime reads the limit from CPUState.cycle_budget
// at every check instead. The runtime sets it before a dispatch to the cycles
// left until its next scheduled event and lowers it when a hardware access
// schedules an earlier one, so generated code leaves when the runtime has
// work to do rather than every fixed number of cycles. A runtime that leaves
// the field zero gets an exit at the first check, which is slow but correct.
static bool budgetFromRuntime() {
  static const bool enabled =
      dolrecomp_codegen_switch_is("DOLRECOMP_BUDGET_CYCLES", "runtime");
  return enabled;
}

static u64 budgetCycles() {
  static const u64 budget = [] {
    const char *v = std::getenv("DOLRECOMP_BUDGET_CYCLES");
    const u64 parsed = v ? std::strtoull(v, nullptr, 0) : 0;
    return parsed ? parsed : u64{256};
  }();
  return budget;
}

Value *FunctionEmitter::budgetExhausted(bool loopHead) {
  Type *i64 = Type::getInt64Ty(context_);
  // With the counters in registers the guard word holds the charged cycles
  // in its low 48 bits and the number of checks above them.
  const u64 chargedMask = (u64{1} << 48) - 1;
  Value *guard = builder_.CreateLoad(i64, guard_cycles_local_);
  if (closed_ && closed_defer_) {
    // DOLRECOMP_CLOSED_DEFER: only the host stack bounds calls, and only a
    // fixed cap on the dispatch's cycles (about twice the longest physics
    // task of F-Zero GX) bounds loops.
    if (!loopHead)
      return builder_.CreateICmpUGE(guard, builder_.getInt64(u64{2048} << 48));
    Value *spent = builder_.CreateAdd(
        builder_.CreateAnd(guard, builder_.getInt64(chargedMask)),
        loadCycles());
    return builder_.CreateICmpUGE(spent, builder_.getInt64(u64{1} << 18));
  }
  Value *cycles = builder_.CreateAdd(
      register_counters_ ? builder_.CreateAnd(guard, builder_.getInt64(chargedMask))
                         : guard,
      loadCycles());
  Value *limit;
  if (modern_runtime_)
    limit = builder_.CreateLoad(i64,
                                builder_.CreateStructGEP(chainType(), chain_, 4));
  else if (budgetFromRuntime())
    // Not tagged as a register or environment load: the runtime may change
    // it during any call out of generated code.
    limit = loadOffset(i64, offsetof(CPUState, cycle_budget));
  else
    limit = ConstantInt::get(i64, budgetCycles());
  Value *overCycles = builder_.CreateICmpUGE(cycles, limit);
  Value *exhausted = overCycles;
  if (register_counters_) {
    // The upper bits count native call depth (a call passes its guard plus
    // one step and keeps its own), so the limit bounds the host stack and
    // nothing else; a long loop of calls is not cut short by it.
    exhausted = builder_.CreateOr(
        overCycles,
        builder_.CreateICmpUGE(guard, builder_.getInt64(u64{2048} << 48)));
  } else if (!native_abi_ || !cold_escapes_) {
    Value *steps =
        builder_.CreateLoad(Type::getInt64Ty(context_), guard_steps_);
    Value *nextSteps = builder_.CreateAdd(
        steps, ConstantInt::get(Type::getInt64Ty(context_), 1));
    builder_.CreateStore(nextSteps, guard_steps_);
    Value *overSteps = builder_.CreateICmpUGE(
        nextSteps, ConstantInt::get(Type::getInt64Ty(context_), 2048));
    exhausted = builder_.CreateOr(overCycles, overSteps);
  }
  return exhausted;
}

void FunctionEmitter::emitBudgetGuard(u32 pc, bool loopHead) {
  Value *exhausted = budgetExhausted(loopHead);
  BasicBlock *run = BasicBlock::Create(context_, "budget_run", function_);
  BasicBlock *exit = BasicBlock::Create(context_, "budget_exit", function_);
  builder_.CreateCondBr(exhausted, exit, run);
  builder_.SetInsertPoint(exit);
  sideExit(pc);
  builder_.SetInsertPoint(run);
}

} // namespace dolllvm
