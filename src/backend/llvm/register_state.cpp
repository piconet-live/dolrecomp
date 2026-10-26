#include "backend/llvm/emitter.h"
#include "backend/llvm/native_abi.h"
#include "cpu/cpu.h"

#include <algorithm>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>

namespace dolllvm {

using namespace llvm;

bool FunctionEmitter::slotInMemory(DolIRStateSlot slot) const {
  if (!state_in_memory_ || native_abi_)
    return false;
  return !((slot >= DOLIR_STATE_CR0 && slot <= DOLIR_STATE_CR7) ||
           (slot >= DOLIR_STATE_XER_CA && slot <= DOLIR_STATE_XER_SO) ||
           slot == DOLIR_STATE_XER);
}

void FunctionEmitter::finalizeStateSSA() {
  SmallVector<AllocaInst *, DOLIR_STATE_COUNT + 64> registers;
  for (Value *slot : state_)
    if (auto *alloca = dyn_cast_or_null<AllocaInst>(slot))
      registers.push_back(alloca);
  for (AllocaInst *pair : pair_f32_)
    if (pair)
      registers.push_back(pair);
  for (AllocaInst *pair : pair_f64_)
    if (pair)
      registers.push_back(pair);
  if (registers.empty())
    return;
  DominatorTree dominators(*function_);
  PromoteMemToReg(registers, dominators);
}

FunctionEmitter::Tracking FunctionEmitter::saveTracking() const {
  return {fp_rep_,          fp_exact_single_, fp_ps0_single_,
          fp_single_canonical_, fp_track_view_, fp_ps0_float_, fp_ps1_float_,
          fp_denormal_safe_, fp_value_class_,  known_state_,
          single_values_,   pending_fprf_,    pending_fpcc_,
          unsynced_,        fp_available_checked_,
          psq_direct_proven_, psq_indexed_proven_};
}

void FunctionEmitter::restoreTracking(const Tracking &tracking) {
  fp_rep_ = tracking.rep;
  fp_exact_single_ = tracking.exact_single;
  fp_ps0_single_ = tracking.ps0_single;
  fp_single_canonical_ = tracking.single_canonical;
  fp_track_view_ = tracking.track_view;
  fp_ps0_float_ = tracking.ps0_float;
  fp_ps1_float_ = tracking.ps1_float;
  fp_denormal_safe_ = tracking.denormal_safe;
  fp_value_class_ = tracking.value_class;
  known_state_ = tracking.known_state;
  single_values_ = tracking.single_values;
  pending_fprf_ = tracking.pending_fprf;
  pending_fpcc_ = tracking.pending_fpcc;
  unsynced_ = tracking.unsynced;
  fp_available_checked_ = tracking.fp_available_checked;
  psq_direct_proven_ = tracking.psq_direct_proven;
  psq_indexed_proven_ = tracking.psq_indexed_proven;
}

void FunctionEmitter::resetFPRepresentations() {
  fp_rep_.fill(FPRepresentation::Raw);
  fp_exact_single_.fill(false);
  fp_ps0_single_.fill(false);
  fp_single_canonical_.fill(false);
  fp_track_view_.fill(false);
  fp_ps0_float_.fill(nullptr);
  fp_ps1_float_.fill(nullptr);
  single_values_.clear();
  fp_denormal_safe_.fill(false);
  fp_value_class_.fill(FPValueClass::Unknown);
}

void FunctionEmitter::invalidateFPRepresentations() {
  resetFPRepresentations();
}

// CR and its fields, XER and its bits, and FPSCR are packed into shared words
// in CPUState (and FPSCR carries the deferred FPRF), so writing one back is a
// read-modify-write. They stay host values across straight-line runs and are
// written where state leaves the function, as before.
bool FunctionEmitter::packedSlot(DolIRStateSlot slot) const {
  return slot >= DOLIR_STATE_CR && slot <= DOLIR_STATE_FPSCR;
}

// The registers whose values survive the RecompCore runtime's memory service
// and FP-available check when those return to the generated code: all of
// them. The runtime writes MSR, SRR0/1, DAR and DSISR only while delivering
// an exception, and the generated code then leaves without reading anything
// back. The bookkeeping slots (pending exception, timebase, downcount, the
// reservation) are the runtime's to change and are left out.
bool FunctionEmitter::runtimeLeavesSlot(DolIRStateSlot slot) const {
  return slot != DOLIR_STATE_PC && slot != DOLIR_STATE_TIMEBASE &&
         slot != DOLIR_STATE_EXCEPTION &&
         slot != DOLIR_STATE_PROGRAM_EXCEPTION &&
         slot != DOLIR_STATE_RESERVE_ADDR &&
         slot != DOLIR_STATE_RESERVE_VALID && slot != DOLIR_STATE_DOWNCOUNT;
}

static MDNode *aliasScopeList(LLVMContext &context, const char *name) {
  MDBuilder metadata(context);
  MDNode *domain = metadata.createAliasScopeDomain("dolrecomp.cpustate");
  return MDNode::get(context, {metadata.createAliasScope(name, domain)});
}

static bool floatingPointSlot(DolIRStateSlot slot) {
  return (slot >= DOLIR_STATE_FPR0 && slot <= DOLIR_STATE_PS1_31) ||
         slot == DOLIR_STATE_FPSCR;
}

Value *FunctionEmitter::tagRegisterLoad(LoadInst *load, DolIRStateSlot slot) {
  if (scoped_alias_ && runtimeLeavesSlot(slot))
    load->setMetadata(LLVMContext::MD_alias_scope,
                      aliasScopeList(context_, floatingPointSlot(slot)
                                                   ? "guest FP registers"
                                                   : "guest integer registers"));
  return load;
}

Value *FunctionEmitter::tagEnvironmentLoad(LoadInst *load) {
  if (scoped_alias_)
    load->setMetadata(LLVMContext::MD_alias_scope,
                      aliasScopeList(context_, "memory layout"));
  return load;
}

// Every call out of generated code leaves the memory layout fields (RAM
// pointers and sizes, the interception hook) alone. `writes` says which of
// the tagged registers it may write; whether it reads them does not matter
// here, because only loads are tagged: a store to CPUState is never moved
// across a call on the strength of this.
void FunctionEmitter::tagRuntimeCall(CallInst *call, RegisterWrites writes) {
  if (!scoped_alias_)
    return;
  MDNode *scopes = aliasScopeList(context_, "memory layout");
  if (writes != RegisterWrites::Any)
    scopes = MDNode::concatenate(
        scopes, aliasScopeList(context_, "guest integer registers"));
  if (writes == RegisterWrites::None)
    scopes = MDNode::concatenate(
        scopes, aliasScopeList(context_, "guest FP registers"));
  call->setMetadata(LLVMContext::MD_noalias, scopes);
}

void FunctionEmitter::markWritten(DolIRStateSlot slot) {
  if (sync_changed_ && !packedSlot(slot) && slot != DOLIR_STATE_PC)
    unsynced_.set(slot);
}

void FunctionEmitter::storeUnsynced(bool clear) {
  if (!sync_changed_)
    return;
  for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
    if (!unsynced_.test(slot))
      continue;
    auto stateSlot = static_cast<DolIRStateSlot>(slot);
    storeContext(stateSlot, builder_.CreateLoad(type(dolir_state_type(stateSlot)),
                                                state_[slot]));
  }
  if (clear)
    unsynced_.reset();
}

void FunctionEmitter::storeSlots(const std::bitset<DOLIR_STATE_COUNT> &slots) {
  for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
    if (!slots.test(slot))
      continue;
    auto stateSlot = static_cast<DolIRStateSlot>(slot);
    storeContext(stateSlot, builder_.CreateLoad(type(dolir_state_type(stateSlot)),
                                                state_[slot]));
  }
}

// The registers DOLRECOMP_STATE_LAZY may leave unwritten across a branch.
static bool lazySlot(u32 slot) {
  return (slot >= DOLIR_STATE_GPR0 && slot <= DOLIR_STATE_GPR31) ||
         (slot >= DOLIR_STATE_FPR0 && slot <= DOLIR_STATE_FPR31) ||
         (slot >= DOLIR_STATE_PS1_0 && slot <= DOLIR_STATE_PS1_31) ||
         slot == DOLIR_STATE_LR || slot == DOLIR_STATE_CTR;
}

// A block a run may branch to with registers unwritten: the first block of a
// run of this function that is emitted as code.
bool FunctionEmitter::lazySuccessor(u32 block) const {
  return block != DOLIR_NO_BLOCK && block < source_.block_count &&
         region_leaders_[block] &&
         source_.blocks[block].terminator.kind != DOLIR_TERM_FALLBACK;
}

// At the end of a run that goes on to `count` runs of this function: writes
// the unsynced registers no successor takes over, makes a block on each edge
// whose target takes over less than the other (emitTerminator branches to it
// instead), and leaves the rest unwritten. False when the run's end is not
// such an end; the caller then writes everything, as without the switch.
bool FunctionEmitter::carryUnsynced(const u32 *successors, u32 count) {
  lazy_edge_.fill(nullptr);
  if (lazy_allowed_.empty())
    return false;
  std::bitset<DOLIR_STATE_COUNT> anywhere;
  for (u32 i = 0; i < count; i++) {
    if (!lazySuccessor(successors[i]))
      return false;
    anywhere |= lazy_allowed_[successors[i]];
  }
  const std::bitset<DOLIR_STATE_COUNT> unsynced = unsynced_;
  storeSlots(unsynced & ~anywhere);
  for (u32 i = 0; i < count; i++) {
    const std::bitset<DOLIR_STATE_COUNT> here =
        unsynced & anywhere & ~lazy_allowed_[successors[i]];
    if (here.none())
      continue;
    BasicBlock *edge = BasicBlock::Create(context_, "lazy_edge", function_);
    IRBuilderBase::InsertPoint saved = builder_.saveIP();
    builder_.SetInsertPoint(edge);
    storeSlots(here);
    builder_.CreateBr(blocks_[successors[i]]);
    builder_.restoreIP(saved);
    lazy_edge_[i] = edge;
    // DOLRECOMP_SINGLE_FP=track: the edge provides what the run's closing
    // branch provides (it only stores doubles that exist already).
    recordSingleEdge(edge);
  }
  unsynced_.reset();
  return true;
}

// Which registers each run takes over unwritten. For a run R:
//   defs(R)   the lazy registers its instructions define;
//   plain(R)  it ends in a branch, conditional branch or fall-through to runs
//             of this function (the ends carryUnsynced accepts);
//   again(R)  = defs(R), plus again(S) of every successor when plain(R): a
//             register some path writes again before a run's end that writes
//             everything. Only such a register is worth leaving unwritten;
//   out(R)    = allowed(R) | defs(R) when plain(R), else nothing;
//   allowed(R) = again(R) & out of every run that branches to R, so that a
//             register is not carried to a place some path reaches with it
//             written (it would be written again there for nothing). One
//             exception, for loops: when R lies in a loop (the runs from a
//             backward branch's target to its source; the innermost such),
//             runs outside the loop that branch to R are not asked, so a
//             register the loop changes on every round is written when the
//             loop is left and not on every round. A run with an entry that
//             has everything written (the function's first run, a call's
//             continuation) takes nothing over.
// Whatever this computes, carryUnsynced keeps CPUState right; the sets only
// decide where the stores are.
void FunctionEmitter::scanLazyState() {
  lazy_allowed_.clear();
  if (!lazy_state_ || !closed_)
    return;
  using Mask = std::bitset<DOLIR_STATE_COUNT>;
  const u32 count = source_.block_count;
  struct Run {
    Mask defs, again, allowed, out;
    std::array<u32, 2> next{};
    u32 next_count = 0;
    bool leader = false, plain = false, clean_entry = false;
  };
  std::vector<Run> runs(count);
  for (u32 index = 0; index < count; index++) {
    if (!region_leaders_[index] ||
        source_.blocks[index].terminator.kind == DOLIR_TERM_FALLBACK)
      continue;
    Run &run = runs[index];
    run.leader = true;
    for (u32 current = index; current < count; current++) {
      const DolIRBlock &block = source_.blocks[current];
      if (current != index && region_leaders_[current]) {
        run.next = {current, current};
        run.next_count = 1;
        run.plain = lazySuccessor(current);
        break;
      }
      for (u32 i = 0; i < block.instruction_count; i++)
        for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++)
          if (lazySlot(slot) &&
              dolir_state_mask_test(block.instructions[i].state_defs,
                                    static_cast<DolIRStateSlot>(slot)))
            run.defs.set(slot);
      const DolIRTerminator &term = block.terminator;
      if (term.kind == DOLIR_TERM_FALLTHROUGH) {
        const u32 next = term.targets[0];
        if (next != DOLIR_NO_BLOCK && next == current + 1u && next < count &&
            !region_leaders_[next])
          continue;
      }
      if (!term.linked && (term.kind == DOLIR_TERM_FALLTHROUGH ||
                           term.kind == DOLIR_TERM_BRANCH)) {
        run.next = {term.targets[0], term.targets[0]};
        run.next_count = 1;
        run.plain = lazySuccessor(term.targets[0]);
      } else if (!term.linked && term.kind == DOLIR_TERM_COND_BRANCH) {
        run.next = {term.targets[0], term.targets[1]};
        run.next_count = 2;
        run.plain = lazySuccessor(term.targets[0]) &&
                    lazySuccessor(term.targets[1]);
      }
      break;
    }
  }
  runs[0].clean_entry = true;
  for (u32 block : continuations_)
    runs[block].clean_entry = true;
  // again: backward, to a fixed point.
  for (u32 pass = 0; pass < 64; pass++) {
    bool changed = false;
    for (u32 index = count; index-- > 0;) {
      Run &run = runs[index];
      if (!run.leader)
        continue;
      Mask again = run.defs;
      for (u32 i = 0; run.plain && i < run.next_count; i++)
        again |= runs[run.next[i]].again;
      changed = changed || again != run.again;
      run.again = again;
    }
    if (!changed)
      break;
  }
  // allowed and out: forward, growing from nothing.
  std::vector<std::vector<u32>> sources(count);
  // The innermost loop around each run, as [first, last] run indices.
  std::vector<std::array<u32, 2>> loop(count, {1u, 0u});
  for (u32 index = 0; index < count; index++) {
    for (u32 i = 0; runs[index].plain && i < runs[index].next_count; i++) {
      const u32 next = runs[index].next[i];
      if (i != 0 && next == runs[index].next[0])
        continue;
      sources[next].push_back(index);
      if (next > index)
        continue;
      for (u32 inside = next; inside <= index; inside++) {
        std::array<u32, 2> &span = loop[inside];
        if (span[0] > span[1] || index - next < span[1] - span[0])
          span = {next, index};
      }
    }
  }
  for (u32 pass = 0; pass < 64; pass++) {
    bool changed = false;
    for (u32 index = 0; index < count; index++) {
      Run &run = runs[index];
      if (!run.leader)
        continue;
      Mask arriving;
      bool asked = false;
      arriving.set();
      const bool in_loop = loop[index][0] <= loop[index][1];
      for (u32 source : sources[index]) {
        if (in_loop && (source < loop[index][0] || source > loop[index][1]))
          continue;
        arriving &= runs[source].out;
        asked = true;
      }
      if (!asked || run.clean_entry)
        arriving.reset();
      const Mask allowed = arriving & run.again;
      const Mask out = run.plain ? (allowed | run.defs) : Mask();
      changed = changed || allowed != run.allowed || out != run.out;
      run.allowed = allowed;
      run.out = out;
    }
    if (!changed)
      break;
  }
  lazy_allowed_.resize(count);
  for (u32 index = 0; index < count; index++)
    lazy_allowed_[index] = runs[index].allowed;
}

void FunctionEmitter::noteStateWrite(DolIRStateSlot slot, Value *value) {
  markWritten(slot);
  known_state_[slot] = dyn_cast<ConstantInt>(value);
  if (slot >= DOLIR_STATE_FPR0 && slot <= DOLIR_STATE_FPR31) {
    u32 reg = slot - DOLIR_STATE_FPR0;
    fp_rep_[reg] = FPRepresentation::Raw;
    fp_exact_single_[reg] = false;
    fp_ps0_single_[reg] = std::find(single_values_.begin(),
                                    single_values_.end(),
                                    value) != single_values_.end();
    fp_single_canonical_[reg] = false;
    fp_ps0_float_[reg] = nullptr;
    fp_denormal_safe_[reg] = false;
    fp_value_class_[reg] = FPValueClass::Unknown;
  } else if (slot >= DOLIR_STATE_PS1_0 && slot <= DOLIR_STATE_PS1_31) {
    u32 reg = slot - DOLIR_STATE_PS1_0;
    fp_rep_[reg] = FPRepresentation::Raw;
    fp_exact_single_[reg] = false;
    fp_single_canonical_[reg] = false;
    fp_ps1_float_[reg] = nullptr;
    fp_denormal_safe_[reg] = false;
    fp_value_class_[reg] = FPValueClass::Unknown;
  } else if (slot == DOLIR_STATE_FPSCR) {
    clearPendingFPRF();
    fp_denormal_safe_.fill(false);
  } else if (slot == DOLIR_STATE_HID2) {
    psq_direct_proven_ = false;
    psq_indexed_proven_ = false;
  } else if (slot == DOLIR_STATE_MSR) {
    fp_available_checked_ = false;
  }
}

void FunctionEmitter::scanState() {
  for (u32 b = 0; b < source_.block_count; b++) {
    const DolIRBlock &block = source_.blocks[b];
    for (u32 i = 0; i < block.instruction_count; i++) {
      const DolIRInstruction &inst = block.instructions[i];
      if (!modern_runtime_ && inst.op == DOLIR_OP_HELPER_CALL &&
          (inst.aux == DOLIR_HELPER_TIMEBASE_READ ||
           inst.aux == DOLIR_HELPER_TIMEBASE_WRITE)) {
        used_[DOLIR_STATE_TIMEBASE] = true;
        dirty_[DOLIR_STATE_TIMEBASE] = dirty_[DOLIR_STATE_TIMEBASE] ||
                                       inst.aux == DOLIR_HELPER_TIMEBASE_WRITE;
      }
      for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
        auto stateSlot = static_cast<DolIRStateSlot>(slot);
        bool reads = dolir_state_mask_test(inst.state_uses, stateSlot);
        bool writes = dolir_state_mask_test(inst.state_defs, stateSlot);
        used_[slot] = used_[slot] || reads || writes;
        dirty_[slot] = dirty_[slot] || writes;
      }
    }
  }
  if (!abi_range_)
    return;
  for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
    auto stateSlot = static_cast<DolIRStateSlot>(slot);
    used_[slot] = used_[slot] || stateInput(abi_range_, stateSlot) ||
                  stateOutput(abi_range_, stateSlot);
    dirty_[slot] = dirty_[slot] || stateOutput(abi_range_, stateSlot) ||
                   (native_abi_ && stateInput(abi_range_, stateSlot));
  }
}

void FunctionEmitter::scanContinuations() {
  for (u32 i = 0; i < source_.block_count; i++) {
    const DolIRTerminator &term = source_.blocks[i].terminator;
    if (!term.linked)
      continue;
    u32 continuation = term.guest_pc + 4u;
    u32 block = 0;
    if (continuation >= source_.guest_start &&
        continuation < source_.guest_end &&
        ((continuation - source_.guest_start) & 3u) == 0) {
      block = (continuation - source_.guest_start) / 4u;
      if (block < source_.block_count)
        continuations_.push_back(block);
    }
  }
}

void FunctionEmitter::scanLoopHeaders() {
  loop_headers_.assign(source_.block_count, false);
  for (u32 i = 0; i < source_.block_count; i++) {
    const DolIRTerminator &term = source_.blocks[i].terminator;
    u32 count = term.kind == DOLIR_TERM_COND_BRANCH ? 2u
                : term.kind == DOLIR_TERM_BRANCH    ? 1u
                                                    : 0u;
    for (u32 edge = 0; edge < count; edge++) {
      if (term.targets[edge] != DOLIR_NO_BLOCK && term.targets[edge] <= i)
        loop_headers_[term.targets[edge]] = true;
    }
  }
}

void FunctionEmitter::scanRegionLeaders() {
  region_leaders_.assign(source_.block_count, false);
  if (!source_.block_count)
    return;
  region_leaders_[0] = true;
  for (u32 i = 0; i < source_.block_count; i++) {
    if (modern_runtime_ && native_abi_ && needsInterpreter(source_.blocks[i])) {
      region_leaders_[i] = true;
      if (i + 1u < source_.block_count)
        region_leaders_[i + 1u] = true;
    }
    const DolIRTerminator &term = source_.blocks[i].terminator;
    if (term.kind == DOLIR_TERM_FALLBACK)
      region_leaders_[i] = true;
    if (i + 1u < source_.block_count && term.kind != DOLIR_TERM_FALLTHROUGH)
      region_leaders_[i + 1u] = true;
    // A block that writes MSR can return to the dispatcher and resume at the
    // NEXT instruction, so that instruction has to be enterable. Terminators
    // are already covered above; this catches the fallthrough case, which is
    // mtmsr.
    //
    // Keyed on an MSR write rather than on MAY_EXIT generally. MAY_EXIT is
    // carried by a large number of instructions -- the broader rule fired
    // 160,525 times on Colosseum -- and every extra leader fragments a region,
    // which costs optimisation. Measured: the broad rule cost plain llvm
    // 5-16% on the three titles that never needed it, while only a handful of
    // addresses are ever actually re-entered.
    //
    // Without this, re-entry lands on a non-leader address, the entry dispatch
    // has no case for it, and the runtime interprets from there until it
    // reaches the next leader. Colosseum calls OSRestoreInterrupts roughly
    // every 139 guest cycles, and its last two instructions sit exactly in that
    // gap: 166M interpreted instructions, 92% of all dispatches in that arm.
    // The C backend never pays this because it labels every address.
    //
    // REQUIRES the runtime to deliver pending external interrupts at the
    // post-mtmsr boundary. emitStateWrite side-exits to guest_pc+4 when MSR[EE]
    // goes 0->1; before this rule the address was not enterable, so the runtime
    // interpreted from there and delivered the interrupt as a side effect of
    // that detour. Removing the detour without the runtime half leaves the
    // guest spinning with EE set on an interrupt that never arrives -- Colosseum
    // advances 19 frames in 20s instead of 1069. native_exc is the tell: 312
    // while hung, 25,009 once delivery is restored.
    if (i + 1u < source_.block_count) {
      const DolIRBlock &body = source_.blocks[i];
      for (u32 n = 0; n < body.instruction_count; n++) {
        if (dolir_state_mask_test(body.instructions[n].state_defs,
                                  DOLIR_STATE_MSR)) {
          region_leaders_[i + 1u] = true;
          break;
        }
      }
    }
    u32 count = term.kind == DOLIR_TERM_COND_BRANCH ? 2u
                : term.kind == DOLIR_TERM_BRANCH    ? 1u
                : term.kind == DOLIR_TERM_INDIRECT  ? 2u
                                                    : 0u;
    for (u32 edge = 0; edge < count; edge++) {
      if (term.targets[edge] != DOLIR_NO_BLOCK)
        region_leaders_[term.targets[edge]] = true;
    }
  }
  for (u32 i = 0; i < entry_point_count_; i++) {
    u32 address = entry_points_[i];
    if (address < source_.guest_start || address >= source_.guest_end ||
        ((address - source_.guest_start) & 3u) != 0)
      continue;
    region_leaders_[(address - source_.guest_start) / 4u] = true;
  }
}

Value *FunctionEmitter::bytePtr(size_t offset) {
  return builder_.CreateInBoundsGEP(Type::getInt8Ty(context_), ctx_,
                                    builder_.getInt64(offset));
}

Value *FunctionEmitter::loadOffset(Type *valueType, size_t offset) {
  return builder_.CreateLoad(valueType, bytePtr(offset));
}

} // namespace dolllvm
