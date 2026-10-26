#include "backend/llvm/emitter.h"
#include "backend/codegen_switches.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>

#include <cstdio>
#include <utility>

// DOLRECOMP_SINGLE_FP=track: single-precision values stay single.
//
// A guest FP register is two doubles. The emitter also keeps a "single view"
// of a register: a pair of host floats (pair_f32_) whose widening is the
// register. Without the switch the view is written by single-result
// operations and psq loads, used by paired arithmetic, and forgotten at the
// start of every straight run.
//
// With the switch:
//   * an lfs load and the ps0-only moves (fmr, fneg, fabs, fnabs, fsel) give
//     or keep a view (noteTrackedLaneWrite; a move of ps0 alone gives the
//     lane-0 state, FPRepresentation::ScalarF32);
//   * a scalar single operation whose operands all have a usable view is
//     computed in single precision (scalar_fp.cpp);
//   * every single fused multiply-add is guarded: a result that is zero,
//     denormal, infinite or a NaN is recomputed the double way;
//   * views are carried from one run to the next. The facts a run may assume
//     at its start are found by emitting the body, recording the facts held
//     at each run's closing branch, and intersecting them over the actual
//     predecessors of each run's block (settleSinglePlan). The caller
//     (llvm_backend.cpp) repeats the emission until the assumptions hold.
//
// A view that exists only because of the switch is never stored to memory
// directly (storeViewAllowed): the store narrows the register's double, as
// the build without the switch does.

namespace dolllvm {

using namespace llvm;

bool FunctionEmitter::trackSingles() {
  static const bool enabled =
      dolrecomp_codegen_switch_is("DOLRECOMP_SINGLE_FP", "track") &&
      dolrecomp_codegen_switch_is("DOLRECOMP_SCALAR_FP", "inline") &&
      dolrecomp_codegen_switch_is("DOLRECOMP_PAIRED_FP", "jit64") &&
      dolrecomp_codegen_switch_is("DOLRECOMP_STATE_SYNC", "changed");
  return enabled;
}

bool FunctionEmitter::laneView(u32 reg) const {
  return (fp_rep_[reg] == FPRepresentation::PairF32 && fp_exact_single_[reg]) ||
         fp_rep_[reg] == FPRepresentation::ScalarF32;
}

bool FunctionEmitter::storeViewAllowed(u32 reg) const {
  if (track_singles_)
    return !fp_track_view_[reg];
  return fp_single_canonical_[reg] || !float_single_math_;
}

bool FunctionEmitter::usesFPRegisters() const {
  for (u32 slot = DOLIR_STATE_FPR0; slot <= DOLIR_STATE_PS1_31; slot++)
    if (used_[slot])
      return true;
  return false;
}

FunctionEmitter::SingleFacts FunctionEmitter::currentSingleFacts() const {
  SingleFacts facts;
  for (u32 reg = 0; reg < 32; reg++) {
    const u32 bit = 1u << reg;
    const bool pair =
        fp_rep_[reg] == FPRepresentation::PairF32 && fp_exact_single_[reg];
    const bool lane0 = pair || fp_rep_[reg] == FPRepresentation::ScalarF32;
    if (pair)
      facts.pair |= bit;
    if (lane0)
      facts.lane0 |= bit;
    if (lane0 && fp_denormal_safe_[reg])
      facts.safe |= bit;
    if (lane0 && fp_single_canonical_[reg])
      facts.canonical |= bit;
    if (lane0 || ps0Single(reg))
      facts.ps0_single |= bit;
  }
  return facts;
}

// The first round's structure, from the DolIR. An ordinary body can be
// entered at every run (the entry switch), so nothing is carried there. A
// closed body is entered at its first instruction; the run after a call
// starts from CPUState.
void FunctionEmitter::guessSinglePlan() {
  SinglePlan &plan = *single_plan_;
  const u32 count = source_.block_count;
  plan.used.assign(count, SingleFacts());
  if (plan.learned)
    return;
  plan.open.assign(count, closed_ ? 0 : 1);
  plan.preds.assign(count, {});
  plan.back.assign(count, SingleFacts::all());
  if (!closed_ || !count)
    return;
  plan.open[0] = 1;
  for (u32 block : continuations_)
    plan.open[block] = 1;
  u32 leader = 0;
  for (u32 i = 0; i < count; i++) {
    if (region_leaders_[i])
      leader = i;
    const DolIRTerminator &term = source_.blocks[i].terminator;
    const u32 edges = term.kind == DOLIR_TERM_COND_BRANCH ? 2u
                      : term.kind == DOLIR_TERM_BRANCH ||
                                term.kind == DOLIR_TERM_FALLTHROUGH
                          ? 1u
                          : 0u;
    for (u32 edge = 0; edge < edges; edge++) {
      const u32 target = term.targets[edge];
      if (target == DOLIR_NO_BLOCK || target >= count ||
          !region_leaders_[target] ||
          source_.blocks[target].terminator.kind == DOLIR_TERM_FALLBACK)
        continue;
      plan.preds[target].push_back(leader);
    }
  }
}

void FunctionEmitter::applySingleSeed(u32 index) {
  twins_.clear();
  current_region_ = index;
  if (!track_singles_ || !single_plan_)
    return;
  SinglePlan &plan = *single_plan_;
  SingleFacts facts;
  if (!plan.nothing && !plan.open[index] && !plan.preds[index].empty()) {
    facts = SingleFacts::all();
    for (u32 pred : plan.preds[index])
      facts.meet(region_done_[pred] ? region_out_[pred]
                 : plan.forward_only ? SingleFacts()
                                     : plan.back[pred]);
  }
  plan.used[index] = facts;
  for (u32 reg = 0; reg < 32; reg++) {
    const u32 bit = 1u << reg;
    if (facts.ps0_single & bit)
      fp_ps0_single_[reg] = true;
    if (!(facts.lane0 & bit))
      continue;
    if (facts.pair & bit) {
      fp_rep_[reg] = FPRepresentation::PairF32;
      fp_exact_single_[reg] = true;
    } else {
      fp_rep_[reg] = FPRepresentation::ScalarF32;
      fp_ps0_single_[reg] = true;
    }
    fp_denormal_safe_[reg] = (facts.safe & bit) != 0;
    fp_single_canonical_[reg] = (facts.canonical & bit) != 0;
    fp_track_view_[reg] = true;
  }
}

// `edge`: a block made on the way from the run's closing branch to the next
// run (DOLRECOMP_STATE_LAZY's edge stores); null for the closing branch's own
// block.
void FunctionEmitter::recordSingleEdge(BasicBlock *edge) {
  if (!track_singles_ || !single_plan_)
    return;
  const SingleFacts facts = currentSingleFacts();
  edge_facts_[edge ? edge : builder_.GetInsertBlock()] = {current_region_,
                                                          facts};
  region_out_[current_region_] = facts;
  region_done_[current_region_] = 1;
}

// True when every run's assumption is contained in what the predecessors of
// its LLVM block provide. In either case the plan takes the structure of the
// emitted body and the facts its runs ended with.
bool FunctionEmitter::settleSinglePlan() {
  SinglePlan &plan = *single_plan_;
  const u32 count = source_.block_count;
  bool sound = true;
  std::vector<char> open(count, 0);
  std::vector<std::vector<u32>> preds(count);
  for (u32 i = 0; i < count; i++) {
    if (!region_leaders_[i] ||
        source_.blocks[i].terminator.kind == DOLIR_TERM_FALLBACK)
      continue;
    SingleFacts provided = SingleFacts::all();
    bool any = false;
    bool unknown = false;
    for (BasicBlock *pred : predecessors(blocks_[i])) {
      any = true;
      auto found = edge_facts_.find(pred);
      if (found == edge_facts_.end()) {
        unknown = true;
        continue;
      }
      provided.meet(found->second.second);
      preds[i].push_back(found->second.first);
    }
    if (unknown || !any)
      provided = SingleFacts();
    open[i] = unknown || !any;
    if (!provided.contains(plan.used[i]))
      sound = false;
  }
  plan.open = std::move(open);
  plan.preds = std::move(preds);
  for (u32 i = 0; i < count; i++) {
    if (region_done_[i])
      plan.back[i].meet(region_out_[i]);
    else
      plan.back[i] = SingleFacts();
  }
  plan.learned = true;
  plan.rounds++;
  return sound;
}

void FunctionEmitter::reportSingleStats() const {
  static const bool enabled =
      dolrecomp_codegen_switch_is("DOLRECOMP_SINGLE_FP_STATS", "1");
  if (!enabled)
    return;
  u32 leaders = 0, open = 0, seeded = 0;
  for (u32 i = 0; i < source_.block_count; i++) {
    if (!region_leaders_[i] ||
        source_.blocks[i].terminator.kind == DOLIR_TERM_FALLBACK)
      continue;
    leaders++;
    if (single_plan_) {
      open += single_plan_->open[i] != 0;
      seeded += single_plan_->used[i].lane0 != 0;
    }
  }
  fprintf(stderr,
          "single-fp %s%s rounds=%u runs=%u open=%u seeded=%u single=%u "
          "double=%u\n",
          source_.name, closed_ ? "_closed" : "",
          single_plan_ ? single_plan_->rounds : 0u, leaders, open, seeded,
          stat_single_, stat_double_);
}

const FunctionEmitter::SingleTwin *FunctionEmitter::twinOf(Value *value) const {
  auto found = twins_.find(value);
  return found == twins_.end() ? nullptr : &found->second;
}

void FunctionEmitter::noteTwin(Value *value, Value *single, bool safe,
                               bool canonical) {
  if (value && single)
    twins_[value] = SingleTwin{single, safe, canonical};
}

// A write of one half of a register (STATE_WRITE of FPRn or PS1n) after
// noteStateWrite has dropped the register's view. `rep`, `exact`, `safe` and
// `canonical` describe the view before the write.
void FunctionEmitter::noteTrackedLaneWrite(DolIRStateSlot slot, Value *value,
                                           FPRepresentation rep, bool exact,
                                           bool safe, bool canonical) {
  const bool ps0 = slot >= DOLIR_STATE_FPR0 && slot <= DOLIR_STATE_FPR31;
  const u32 reg = slot - (ps0 ? DOLIR_STATE_FPR0 : DOLIR_STATE_PS1_0);
  const bool hadPair = rep == FPRepresentation::PairF32 && exact;
  const bool hadLane0 = hadPair || rep == FPRepresentation::ScalarF32;
  const SingleTwin *twin = twinOf(value);
  Type *pairType = FixedVectorType::get(Type::getFloatTy(context_), 2);
  if (ps0) {
    if (!twin)
      return;
    Value *pair = hadPair ? pairF32(reg)
                          : static_cast<Value *>(PoisonValue::get(pairType));
    pair = builder_.CreateInsertElement(pair, twin->single, uint64_t{0});
    builder_.CreateStore(pair, pair_f32_[reg]);
    fp_rep_[reg] =
        hadPair ? FPRepresentation::PairF32 : FPRepresentation::ScalarF32;
    fp_exact_single_[reg] = hadPair;
    fp_ps0_single_[reg] = true;
    fp_denormal_safe_[reg] = twin->safe && (!hadPair || safe);
    fp_single_canonical_[reg] = twin->canonical && (!hadPair || canonical);
    fp_track_view_[reg] = true;
    return;
  }
  if (!hadLane0)
    return;
  fp_ps0_single_[reg] = true;
  fp_track_view_[reg] = true;
  if (!twin) {
    // Lane 0 is still the register's ps0.
    fp_rep_[reg] = FPRepresentation::ScalarF32;
    fp_exact_single_[reg] = false;
    fp_denormal_safe_[reg] = safe;
    fp_single_canonical_[reg] = canonical;
    return;
  }
  Value *pair = builder_.CreateInsertElement(pairF32(reg), twin->single, 1u);
  builder_.CreateStore(pair, pair_f32_[reg]);
  fp_rep_[reg] = FPRepresentation::PairF32;
  fp_exact_single_[reg] = true;
  fp_denormal_safe_[reg] = safe && twin->safe;
  fp_single_canonical_[reg] = canonical && twin->canonical;
}

} // namespace dolllvm
