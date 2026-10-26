#ifndef DOLRECOMP_LLVM_EMITTER_H
#define DOLRECOMP_LLVM_EMITTER_H

#include "backend/llvm/llvm_backend.h"

#include <array>
#include <bitset>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/IRBuilder.h>

namespace llvm {
class AllocaInst;
class Argument;
class BasicBlock;
class ConstantInt;
class Function;
class GlobalVariable;
class FunctionType;
class LLVMContext;
class LoadInst;
class CallInst;
class Module;
class PHINode;
class StructType;
class Type;
class Value;
class raw_ostream;
} // namespace llvm

namespace dolllvm {

class FunctionEmitter final {
public:
  FunctionEmitter(llvm::LLVMContext &context, llvm::Module &module,
                  const DolIRFunction &source, const DolLLVMOptions &options);

  bool emit(llvm::raw_ostream &diagnostics);
  // The function is a callee imported into another chunk's module.
  void setImported() { imported_ = true; }
  // Emit the function's closed body (func_<address>_closed) instead of its
  // ordinary body and entry wrapper; see closed_ below.
  void setClosed() { closed_ = true; }
  // The function is in the closed set and this build can emit closed bodies.
  bool wantsClosedBody() const;
  // An imported callee that is to be given as its closed body
  // (DOLRECOMP_INLINE_CLOSED=1).
  bool wantsImportedClosedBody() const;

  // DOLRECOMP_SINGLE_FP=track: single views of FP registers are carried from
  // one straight run to the next, scalar single operations are computed in
  // single precision when every operand has a view, and moves keep views.
  //
  // What a run may assume about the views at its first instruction (one bit
  // per FP register in each mask): `pair` = both lanes of the register's
  // single pair are valid, `lane0` = lane 0 is, `safe` = the view may be
  // used for arithmetic under the current FPSCR mode, `canonical` = its bits
  // are what narrowing the register gives, `ps0_single` = ps0 is a value a
  // single can hold.
  struct SingleFacts {
    u32 pair = 0, lane0 = 0, safe = 0, canonical = 0, ps0_single = 0;
    static SingleFacts all() {
      return {~0u, ~0u, ~0u, ~0u, ~0u};
    }
    bool contains(const SingleFacts &other) const {
      return !(other.pair & ~pair) && !(other.lane0 & ~lane0) &&
             !(other.safe & ~safe) && !(other.canonical & ~canonical) &&
             !(other.ps0_single & ~ps0_single);
    }
    void meet(const SingleFacts &other) {
      pair &= other.pair;
      lane0 &= other.lane0;
      safe &= other.safe;
      canonical &= other.canonical;
      ps0_single &= other.ps0_single;
    }
  };
  // How the views are carried from one run to the next. The emitter is its
  // own transfer function. Runs are emitted in address order; a run starts
  // with the intersection of the facts its predecessors hold at their
  // closing branches: the facts just recorded for a predecessor emitted
  // earlier in this round, `back` for one that comes later (a loop), nothing
  // when the run is `open` (entered by an edge that provides nothing: the
  // entry switch, the block after a call, a jump table). After the body is
  // emitted, emit() compares what each run assumed (`used`) with the
  // intersection of the facts recorded on the actual predecessors of its
  // LLVM block. If every assumption is contained in it, the assumptions are
  // an invariant of the body and the body is kept. Otherwise the body is
  // deleted, `open`, `preds` and `back` are replaced by what the emitted
  // body showed (`back` only ever shrinks), singleSeedsRejected() is true
  // and the caller emits again with a new emitter. The first round guesses
  // the structure from the DolIR and assumes every view on back edges.
  struct SinglePlan {
    bool learned = false;
    bool forward_only = false; // back edges provide nothing
    bool nothing = false;      // no run assumes anything
    std::vector<char> open;
    std::vector<std::vector<u32>> preds;
    std::vector<SingleFacts> back;
    std::vector<SingleFacts> used;
    u32 rounds = 0;
  };
  void setSinglePlan(SinglePlan *plan) { single_plan_ = plan; }
  bool singleSeedsRejected() const { return single_seeds_rejected_; }
  static bool trackSingles();
  u32 blockCount() const { return source_.block_count; }

private:
  enum class FPRepresentation : u8 {
    Raw,
    ScalarF64,
    ScalarF32,
    PairF64,
    PairF32,
  };

  enum class FPValueClass : u8 {
    Unknown,
    NoNaN,
    Finite,
    NormalOrZero,
  };

  std::string blockName(u32 index) const;
  std::string symbolName(llvm::StringRef base) const;
  llvm::Type *type(DolIRType type);
  llvm::FunctionType *bodyFunctionType(const DolLLVMFunctionRange *range);
  llvm::CallingConv::ID bodyCallingConvention() const;
  llvm::Type *nativeResultType(const DolLLVMFunctionRange *range);
  llvm::StructType *chainType();
  llvm::StructType *runtimeType();
  llvm::StructType *stateInterfaceType();
  bool stateInput(const DolLLVMFunctionRange *range, DolIRStateSlot slot) const;
  bool stateOutput(const DolLLVMFunctionRange *range,
                   DolIRStateSlot slot) const;
  std::size_t stateOffset(DolIRStateSlot slot) const;
  llvm::Value *bytePtr(std::size_t offset);
  llvm::Value *runtimeField(u32 field);
  llvm::Value *stateField(u32 field);
  llvm::Value *callStateRead(u32 field, llvm::ArrayRef<llvm::Value *> arguments,
                             llvm::Type *result_type);
  void callStateWrite(u32 field, llvm::ArrayRef<llvm::Value *> arguments);
  llvm::Value *loadContext(DolIRStateSlot slot);
  void storeContext(DolIRStateSlot slot, llvm::Value *value);
  llvm::Value *loadOffset(llvm::Type *value_type, std::size_t offset);

  void scanState();
  void scanStableControls();
  void scanContinuations();
  void scanLoopHeaders();
  void scanRegionLeaders();
  void finalizeStateSSA();
  bool slotInMemory(DolIRStateSlot slot) const;

  void emitEntry();
  bool emitWrapper(llvm::raw_ostream &diagnostics);
  bool emitModernWrapper(llvm::raw_ostream &diagnostics);
  void chargeCycles(u32 cycles);
  llvm::Value *loadCycles();
  void storeCycles(llvm::Value *cycles);
  void flushDeferredCycles();
  void chargeCycles(llvm::Value *cycles);
  llvm::Value *emitTimebaseRead();
  void emitTimebaseWrite(const DolIRInstruction &inst);
  void syncDirtyState();
  void commitModernState();
  void reloadModernState();
  void settleCycles();
  void flushCallCounters(bool force_cycles = false);
  void reloadCallCounters();
  void returnFromBody();
  void returnNative(llvm::Value *pc);
  llvm::Value *nativeResult(llvm::Value *pc, bool native, bool from_context);
  llvm::Value *nativeOutputs(bool from_context);
  u32 nativeOutputLaneCount(const DolLLVMFunctionRange *range);
  bool nativeCyclesInResult(const DolLLVMFunctionRange *range);
  u32 nativeResultLaneCount(const DolLLVMFunctionRange *range);
  llvm::Value *nativeCycleValue(llvm::Value *result,
                                const DolLLVMFunctionRange *range);
  llvm::Value *nativeOutputValue(llvm::Value *result,
                                 const DolLLVMFunctionRange *range,
                                 DolIRStateSlot slot);
  llvm::Value *nativeResultPC(llvm::Value *result);
  llvm::Value *nativeResultContinues(llvm::Value *result);
  void acceptNativeResult(llvm::Value *result,
                          const DolLLVMFunctionRange *range);
  void materialize(u32 pc);
  void materialize(llvm::Value *pc);
  void sideExit(u32 pc, u32 reason = 0);
  llvm::Value *entryValue(DolIRStateSlot slot, const char *name);
  void emitBudgetGuard(u32 pc, bool loop_head = false);
  llvm::Value *budgetExhausted(bool loop_head = false);
  void emitNativeIndirectCall(const DolIRTerminator &term, llvm::Value *target);
  bool emitRegion(u32 index, llvm::raw_ostream &diagnostics);
  llvm::Value *operand(const DolIRInstruction &instruction, u32 index);
  llvm::Value *castValue(DolIROp op, llvm::Type *result_type,
                         llvm::Value *value);
  llvm::Value *bswap(llvm::Value *value);
  bool emitInstruction(const DolIRInstruction &instruction,
                       llvm::raw_ostream &diagnostics);
  void emitStateWrite(const DolIRInstruction &instruction);

  llvm::AllocaInst *temporary(llvm::Type *value_type, llvm::StringRef name);
  llvm::Value *stateValue(DolIRStateSlot slot);
  void syncState(DolIRStateSlot slot);
  void reloadState(DolIRStateSlot slot);
  void reloadUsedState();
  void resetFPRepresentations();
  void invalidateFPRepresentations();
  void noteStateWrite(DolIRStateSlot slot, llvm::Value *value);
  void initializeEntryControls();
  llvm::Value *gqrValue(u32 index);
  llvm::Value *gqrType(u32 index, bool load);
  llvm::Value *gqrScale(u32 index, bool load);
  llvm::Value *psqEnabled(bool indexed);
  llvm::Value *niEnabled();
  llvm::Value *pairF32(u32 reg);
  llvm::Value *pairF64(u32 reg);
  llvm::Value *architecturalPairF64(u32 reg);
  void writePairF32(u32 reg, llvm::Value *pair, bool denormal_safe,
                    FPValueClass value_class = FPValueClass::Unknown);
  void writePairF64(u32 reg, llvm::Value *pair, bool exact_single = false,
                    FPValueClass value_class = FPValueClass::Unknown);
  // What the emitter knows at the current insertion point about values it
  // has not written to the state yet. A cold path that leaves and re-enters
  // the function through CPUState (a memory access served by the runtime, a
  // psq helper) does not change any of it, so the knowledge is saved before
  // such a path is emitted and restored for the code after the join.
  struct Tracking {
    std::array<FPRepresentation, 32> rep;
    std::array<bool, 32> exact_single;
    std::array<bool, 32> ps0_single;
    std::array<bool, 32> single_canonical;
    std::array<bool, 32> track_view;
    std::array<llvm::Value *, 32> ps0_float;
    std::array<llvm::Value *, 32> ps1_float;
    std::array<bool, 32> denormal_safe;
    std::array<FPValueClass, 32> value_class;
    std::array<llvm::ConstantInt *, DOLIR_STATE_COUNT> known_state;
    std::vector<llvm::Value *> single_values;
    llvm::Value *pending_fprf;
    llvm::Value *pending_fpcc;
    std::bitset<DOLIR_STATE_COUNT> unsynced;
    bool fp_available_checked;
    bool psq_direct_proven;
    bool psq_indexed_proven;
  };
  Tracking saveTracking() const;
  void restoreTracking(const Tracking &tracking);
  void updateFPRF(llvm::Value *lane0);
  void updateFPCC(llvm::Value *compare);
  void clearPendingFPRF();
  void materializeFPRF(bool keep_pending = false);
  bool packedSlot(DolIRStateSlot slot) const;
  bool runtimeLeavesSlot(DolIRStateSlot slot) const;
  llvm::Value *tagRegisterLoad(llvm::LoadInst *load, DolIRStateSlot slot);
  llvm::Value *tagEnvironmentLoad(llvm::LoadInst *load);
  // What a call out of generated code may write among the guest registers
  // of runtimeLeavesSlot.
  enum class RegisterWrites { None, FloatingPoint, Any };
  void tagRuntimeCall(llvm::CallInst *call, RegisterWrites writes);
  void markWritten(DolIRStateSlot slot);
  void storeUnsynced(bool clear);
  llvm::Value *roundPairToSingle(llvm::Value *pair);
  llvm::Value *forcePairedMultiplierPrecision(llvm::Value *pair);
  llvm::Value *preferPairedNaN(llvm::Value *result, llvm::Value *left,
                               llvm::Value *right, bool operands_finite);
  llvm::Value *constrainedPairBinary(unsigned intrinsic, llvm::Value *left,
                                     llvm::Value *right);
  llvm::Value *constrainedPairFMA(llvm::Value *left, llvm::Value *multiplier,
                                  llvm::Value *addend);
  llvm::Value *correctPairedFMARounding(llvm::Value *result, llvm::Value *left,
                                        llvm::Value *multiplier,
                                        llvm::Value *addend);
  llvm::Value *preferPairedFMANaN(llvm::Value *result, llvm::Value *left,
                                  llvm::Value *addend, llvm::Value *multiplier,
                                  bool operands_finite);
  bool emitVectorPaired(DolIRExactPaired op, u32 d, u32 a, u32 b, u32 c);
  bool emitVectorPairedFMA(DolIRExactPaired op, u32 d, u32 a, u32 b, u32 c);
  void continueAfterRuntimeBoundary(llvm::StringRef prefix);
  void emitFPSCRUpdated();
  void emitHostFPMode(llvm::Value *before, llvm::Value *after);
  void emitFPSCRBit(u64 descriptor);
  void emitProgramException(const DolIRInstruction &instruction);
  llvm::Value *emitSPRRead(const DolIRInstruction &instruction);
  void emitSPRWrite(const DolIRInstruction &instruction);
  void emitLSWX(const DolIRInstruction &instruction);
  void emitCacheControl(const DolIRInstruction &instruction);
  void emitInstructionService(u32 pc);
  llvm::Value *emitRuntimeBoundary(const DolIRInstruction &instruction);
  void emitExactFloat(u64 descriptor);
  bool emitInlineScalarFloat(DolIRExactFloat op, u32 d, u32 a, u32 b, u32 c,
                             u32 crfd);
  llvm::Value *scalarF64(u32 reg);
  bool ps0Single(u32 reg) const;
  bool singleSafe(u32 reg) const;
  llvm::Value *scalarF32(u32 reg);
  void noteFloatLaneWrite(DolIRStateSlot slot, llvm::Value *value);
  llvm::Value *forceScalarMultiplierPrecision(llvm::Value *value);
  llvm::Value *constrainedScalar(unsigned intrinsic, llvm::Value *left,
                                 llvm::Value *right);
  llvm::Value *roundScalarToSingle(llvm::Value *value);
  llvm::Value *correctScalarFMARounding(llvm::Value *result, llvm::Value *left,
                                        llvm::Value *multiplier,
                                        llvm::Value *addend);
  void writeScalarSingle(u32 reg, llvm::Value *single);
  void writeScalarDouble(u32 reg, llvm::Value *value);
  void noteSingleValue(llvm::Value *value);
  void emitExactPaired(u64 descriptor);
  llvm::Value *emitPSQ(const DolIRInstruction &instruction);
  llvm::Value *emitPSQLoad(const DolIRInstruction &, llvm::Value *, llvm::Value *, llvm::Value *);
  llvm::Value *emitKnownPSQ(const DolIRInstruction &instruction, u32 type,
                            s32 scale, llvm::Value *guard = nullptr);
  std::pair<llvm::Value *, llvm::Value *> hostPSQFloats(llvm::Value *bits0,
                                                        llvm::Value *bits1);
  void emitStoreConditional(const DolIRInstruction &instruction);
  llvm::Value *emitFPAvailable(u32 pc);

  llvm::Value *normalizeAddress(llvm::Value *address);
  llvm::Value *provenMemoryPointer(const DolIRInstruction &instruction,
                                   llvm::Value *address, u32 width,
                                   llvm::Value **offset);
  llvm::Value *rangeCheck(llvm::Value *normalized, u32 base, llvm::Value *size,
                          u32 width);
  llvm::Value *endianLoad(llvm::Value *pointer, llvm::Type *result_type,
                          u32 width);
  llvm::Value *externalRead(llvm::Value *address, u32 width);
  llvm::Value *emitGuestLoad(const DolIRInstruction &instruction,
                             llvm::Value *address, llvm::Type *result_type,
                             u32 width, bool sign);
  llvm::Value *emitGuestLoad(llvm::Value *address, llvm::Type *result_type,
                             u32 width, bool sign);
  void clearReservation(llvm::Value *address);
  void journal(llvm::Value *offset, u32 width);
  void endianStore(llvm::Value *pointer, llvm::Value *value, u32 width);
  void externalWrite(llvm::Value *address, llvm::Value *value, u32 width);
  void emitGuestStore(const DolIRInstruction &instruction, llvm::Value *address,
                      llvm::Value *value, u32 width);
  void emitGuestStore(llvm::Value *address, llvm::Value *value, u32 width);
  void emitGatherPipeStore(llvm::Value *address, llvm::Value *value,
                           u32 width);

  llvm::BasicBlock *directDestination(const DolIRTerminator &terminator,
                                      u32 slot);
  const DolLLVMFunctionRange *rangeFor(u32 address) const;
  llvm::BasicBlock *externalDestination(const DolIRTerminator &terminator,
                                        u32 slot);
  llvm::BasicBlock *exitDestination(u32 pc);
  llvm::BasicBlock *fallbackDestination(const DolIRTerminator &terminator);
  llvm::BasicBlock *fallbackEdge(u32 pc);
  void emitFallbackHandler();
  void emitColdEntry(llvm::BasicBlock *entry_miss);
  bool emitTerminator(const DolIRTerminator &terminator,
                      llvm::raw_ostream &diagnostics);

  llvm::LLVMContext &context_;
  llvm::Module &module_;
  const DolIRFunction &source_;
  llvm::IRBuilder<> builder_;
  llvm::Function *function_ = nullptr;
  llvm::Argument *ctx_ = nullptr;
  llvm::Argument *state_interface_ = nullptr;
  llvm::Value *chain_ = nullptr;
  llvm::BasicBlock *entry_ = nullptr;
  llvm::AllocaInst *cycles_ = nullptr;
  llvm::AllocaInst *guard_cycles_local_ = nullptr;
  llvm::Value *pending_cycles_ = nullptr;
  // Shared across generated calls until control returns to the dispatcher.
  llvm::Value *guard_cycles_ = nullptr;
  // Termination backstop for zero-cycle loops.
  llvm::Value *guard_steps_ = nullptr;
  llvm::Value *control_pc_ = nullptr;
  llvm::Value *entry_pc_ = nullptr;
  llvm::Value *return_pc_ = nullptr;
  llvm::Value *initial_cycles_ = nullptr;
  std::array<llvm::Argument *, DOLIR_STATE_COUNT> native_inputs_{};
  llvm::Value *ram_ = nullptr;
  // ram_ minus the guest base address, so a checked MEM1 access indexes it
  // with the guest address itself.
  llvm::Value *ram_biased_ = nullptr;
  llvm::Value *ram_size_ = nullptr;
  llvm::Value *mem2_ = nullptr;
  llvm::Value *mem2_size_ = nullptr;
  llvm::BasicBlock *fallback_block_ = nullptr;
  llvm::PHINode *fallback_pc_ = nullptr;
  std::array<llvm::Value *, DOLIR_STATE_COUNT> state_{};
  std::array<llvm::AllocaInst *, 32> pair_f32_{};
  std::array<llvm::AllocaInst *, 32> pair_f64_{};
  std::array<FPRepresentation, 32> fp_rep_{};
  std::array<bool, 32> fp_exact_single_{};
  std::array<bool, 32> fp_denormal_safe_{};
  std::array<FPValueClass, 32> fp_value_class_{};
  std::array<llvm::ConstantInt *, DOLIR_STATE_COUNT> known_state_{};
  std::array<bool, 8> stable_gqr_{};
  bool stable_hid2_ = false;
  bool stable_ni_ = false;
  std::array<llvm::Value *, 8> entry_gqr_{};
  std::array<llvm::Value *, 8> entry_gqr_load_type_{};
  std::array<llvm::Value *, 8> entry_gqr_store_type_{};
  std::array<llvm::Value *, 8> entry_gqr_load_scale_{};
  std::array<llvm::Value *, 8> entry_gqr_store_scale_{};
  llvm::Value *entry_psq_direct_enabled_ = nullptr;
  llvm::Value *entry_psq_indexed_enabled_ = nullptr;
  llvm::Value *entry_ni_enabled_ = nullptr;
  bool psq_direct_proven_ = false;
  bool psq_indexed_proven_ = false;
  llvm::Value *pending_fprf_ = nullptr;
  // A compare's FPCC (the low four FPRF bits) that is not in FPSCR yet. It
  // applies on top of pending_fprf_, which still supplies the class bit.
  llvm::Value *pending_fpcc_ = nullptr;
  // ps0 of the register is known to hold a value a single can represent, so
  // rounding the multiplier to 25 bits (Force25Bit) would not change it.
  std::array<bool, 32> fp_ps0_single_{};
  // Values of the region being emitted that are known to be such singles.
  std::vector<llvm::Value *> single_values_;
  std::array<bool, DOLIR_STATE_COUNT> used_{};
  std::array<bool, DOLIR_STATE_COUNT> dirty_{};
  std::vector<llvm::BasicBlock *> blocks_;
  std::vector<bool> region_leaders_;
  std::vector<bool> loop_headers_;
  std::vector<llvm::Value *> values_;
  std::vector<u32> continuations_;
  const DolLLVMFunctionRange *ranges_ = nullptr;
  u32 range_count_ = 0;
  const DolLLVMFunctionRange *abi_range_ = nullptr;
  bool native_abi_ = false;
  bool cold_escapes_ = false;
  bool intrinsic_escapes_ = false;
  const u32 *entry_points_ = nullptr;
  u32 entry_point_count_ = 0;
  bool write_journal_ = false;
  DolLLVMSemantics semantics_ = DOLLLVM_SEMANTICS_EXACT;
  std::string symbol_suffix_;
  bool fixed_memory_layout_ = false;
  bool state_in_memory_ = false;
  bool modern_runtime_ = false;
  u32 expected_ram_size_ = 0;
  u32 expected_mem2_size_ = 0;
  u32 current_pc_ = 0;
  bool fp_available_checked_ = false;
  bool inline_scalar_fp_ = false;
  // DOLRECOMP_PAIRED_FP=jit64: the paired-single fix-ups follow Jit64 too
  // (no NaN reselection, the plain 25-bit multiplier rounding, its tie
  // correction), like the inline scalar path.
  bool jit64_paired_fp_ = false;
  // DOLRECOMP_PS_MADD=guarded: the packed single fused multiply-add of the
  // ps_madd family (taken when all three operands have single views that are
  // safe) is kept only when both lanes' results are normal numbers. A result
  // with exponent field 0 or 255 (zero, denormal, infinity, NaN) is computed
  // again by the double sequence, which is what Jit64 always runs: for a
  // denormal result the two differ by one unit in round-to-nearest with NI
  // clear. Without the switch the single result is used unchecked.
  bool guard_paired_fma_ = false;
  llvm::Value *singleFMAUnusual(llvm::Value *result);
  // DOLRECOMP_FMA=hw: functions that contain a fused multiply-add are
  // compiled with the host's FMA instructions instead of calling fma().
  bool hardware_fma_ = false;
  bool uses_fma_ = false;
  bool triple_is_x86_ = false;
  // DOLRECOMP_FPRF=lazy: a pending FPRF is written to FPSCR where the state
  // leaves the function (a call, an exit, a runtime service), not before
  // every memory access.
  bool lazy_fprf_ = false;
  // DOLRECOMP_STATE_SYNC=changed: a guest register the function changes is
  // written to CPUState once, at the end of the straight-line run that
  // changed it (or earlier, in a cold path that hands control to the
  // runtime), instead of every register the function ever writes being
  // written at every call, exit and slow path. unsynced_ is the set changed
  // since the last write-back at the current insertion point.
  bool sync_changed_ = false;
  std::bitset<DOLIR_STATE_COUNT> unsynced_;
  // DOLRECOMP_STATE_LAZY=regions (closed bodies only, on top of
  // DOLRECOMP_STATE_SYNC=changed): a general register, a floating-point
  // register (either half), LR or CTR that a straight-line run changed is not
  // written to CPUState at the run's end when the run ends in a branch, a
  // conditional branch or a fall-through to a run of the same function that
  // takes it over. It stays in unsynced_ there, so everything that writes
  // the unsynced registers (a call, a return, an exit, every cold path that
  // hands control to the runtime) writes it as before. No register is ever
  // left unwritten because it is dead: the store only moves to the first
  // place where something outside the function's straight-line code can run.
  //
  // lazy_allowed_[leader] is what the run starting there takes over. It comes
  // from a dataflow (scanLazyState) and only places stores: at each run's end
  // the registers really unsynced are compared with what each successor takes
  // over, and the others are stored on that edge (carryUnsynced). A run
  // entered in any other way (the entry, a call's continuation, a jump table)
  // is entered with everything written, which any set covers.
  bool lazy_state_ = false;
  std::vector<std::bitset<DOLIR_STATE_COUNT>> lazy_allowed_;
  std::array<llvm::BasicBlock *, 2> lazy_edge_{};
  void scanLazyState();
  bool lazySuccessor(u32 block) const;
  void storeSlots(const std::bitset<DOLIR_STATE_COUNT> &slots);
  bool carryUnsynced(const u32 *successors, u32 count);
  bool imported_ = false;
  // DOLRECOMP_STATE_ALIAS=scoped: loads of guest registers from CPUState and
  // the runtime calls that neither read nor write those registers carry
  // scoped-noalias metadata, so LLVM can keep a register value across such a
  // call. Stores carry none and are never moved across one.
  bool scoped_alias_ = false;
  // DOLRECOMP_INDIRECT_CALLS=native: bctrl/blrl look the target up in the
  // module's entry table and call it like a direct call, instead of leaving
  // for the run loop to dispatch it. Call sites listed in
  // DOLRECOMP_INDIRECT_CALL_EXITS (hex guest addresses, comma separated) keep
  // leaving, for instrumentation that watches dispatches.
  bool native_indirect_calls_ = false;
  // DOLRECOMP_INDIRECT_GUARDS=<site>:<target>[+<target>...][,<site>:...]
  // (hex guest addresses): at the listed bctrl/blrl sites the target is
  // first compared with the listed addresses, and a match is called as a
  // direct call to that address would be (from a closed body to a closed
  // function with the closed protocol), without the table lookup. Any other
  // target takes the lookup as before. The targets come from a profile.
  void emitIndirectGuards(const DolIRTerminator &term, llvm::Value *target);
  // Every function body starts by testing a byte of its own,
  // dolrecomp_native_valid_<address>, and leaves for the run loop with
  // nothing changed when it is zero. The byte is defined set, so a host that
  // never touches it gets the body as before. A host that checks guest code
  // against the module clears it while the function's code is unverified,
  // invalidated or found different, which makes a direct or indirect call
  // from other native code respect that state the same way a dispatch does.
  // DOLRECOMP_ENTRY_CHECK=off leaves the test out (to measure its cost).
  bool entry_check_ = false;
  // DOLRECOMP_ENTRY_QUERY=off: an ordinary body does not ask the runtime at
  // its entry whether the host intercepts the function
  // (ppc_native_region_available, asked whenever a host call hook is
  // installed). For a host whose hook answers only addresses outside all
  // compiled code: there the answer is always "enter", and the question is a
  // call at every function entry. The valid byte is still tested.
  bool entry_query_ = true;
  // DOLRECOMP_CALL_COUNTERS=registers: a function body takes the cycle count
  // and the budget guard as two more arguments and returns both, instead of
  // every call and return writing them to the call chain's memory and reading
  // them back. The guard word carries the cycles already charged to the
  // runtime in its low 48 bits and the number of budget checks of this
  // dispatch above them. A guest return hands its cycles to the caller
  // uncharged; the entry wrapper charges what is left when the body returns
  // to the run loop. Only for builds without native-ABI functions.
  bool register_counters_ = false;
  llvm::Argument *initial_guard_ = nullptr;
  llvm::Value *counterResult();
  void acceptCounters(llvm::Value *result);
  llvm::Value *calleeGuard(bool deeper);
  // DOLRECOMP_MEM_TRACE=1 (analysis builds only): every guest load and store
  // first calls fz_trace(address, width | write << 8), which the module
  // forwards to the host when it has registered a listener.
  void traceAccess(llvm::Value *address, u32 width, bool write);
  // DOLRECOMP_MEM2_TRACK=1: before every access through the second window,
  // the window's last eight bytes are read as a host pointer; when it is not
  // null it points to a record whose first member is a function, called as
  // record(pointer, window, offset, width | write << 8) before the access.
  // The host records an access's words there (a host's own per-thread copy
  // of the window: fzgx-multiplayer port/moderngekko/src/lanes_hooks.cpp).
  // The call changes no guest register. Off: no code at all.
  void trackMem2(llvm::Value *offset, u32 width, bool write);
  // DOLRECOMP_COUNT=1 (analysis builds only): the module counts the straight
  // runs it enters, mode changes, indirect calls, runtime-served accesses and
  // where memory accesses point (count.cpp).
  bool counting() const;
  bool unmappedReads(u32 *first, u32 *last) const;
  llvm::GlobalVariable *countHere(u32 address, u32 kind, u32 detail,
                                  bool increment = true);
  void setCountDetail(llvm::GlobalVariable *record, u32 kind, u32 detail);
  void countAccess(llvm::Value *address, bool write);
  u32 count_sequence_ = 0;
  // DOLRECOMP_ENTRY_LOADS=split: an entry at the function's first instruction
  // (every call) and an entry at any other address (a resume from the run
  // loop) each load the registers on their own path, instead of one set of
  // loads ahead of the entry switch. The call path then loads only what is
  // live from the first instruction, and no register has to be kept across
  // the switch for the benefit of resume points.
  bool split_entry_loads_ = false;
  // DOLRECOMP_CYCLES=deferred: the cycle cost of the instructions emitted so
  // far in the current straight-line run is a compile-time sum, added to the
  // counter where the counter is read (a call, an exit, a slow path, the end
  // of the run) instead of one add per guest instruction.
  bool defer_cycles_ = false;
  u64 deferred_cycles_ = 0;
  // DOLRECOMP_FPSCR=inline: mtfsb0/1 on the control bits and the update after
  // a write to FPSCR are emitted inline, with a direct MXCSR load when the
  // rounding or non-IEEE bits change.
  bool inline_fpscr_ = false;
  llvm::Value *fpscr_before_write_ = nullptr;
  // DOLRECOMP_SINGLE_MATH=float: a register loaded by lfs has a single view
  // like the result of a single operation, and a scalar single operation
  // whose inputs all have one is computed in single precision. With a
  // 53-bit intermediate, rounding the double result of +, -, x, / to single
  // gives the same value as the single operation, and the single fused
  // multiply-add is what the double one plus the tie correction computes.
  // DOLRECOMP_CLOSED_ROOTS: the closed set (pipeline.c marks its functions
  // with DOLLLVM_FUNCTION_ABI_CLOSED). A function in it gets a second body,
  // func_<address>_closed(ctx, control, cycles, guard) -> {cycles, guard}:
  //   - entered at the function's first instruction only, so there is no
  //     entry switch and every other instruction is reached only from inside
  //     the body; no interception query (the ordinary body hands over to the
  //     closed one only while the host has no call hook installed);
  //   - a direct call or branch to another closed function's first
  //     instruction calls that function's closed body; the callee compares
  //     the address its guest return goes to with the return address in
  //     `control` and returns normally when they are equal, without storing
  //     the PC. Everything else a closed body does on the way out is an
  //     escape: the state is written to CPUState as for an ordinary exit,
  //     the PC says where to continue, and bit 63 of the returned guard word
  //     is set. A caller that gets an escape back returns it unchanged; the
  //     run loop then continues at that PC in ordinary code. Nothing is
  //     undone and nothing is run twice;
  //   - calls to functions outside the set and indirect calls use the
  //     ordinary protocol in place (the ordinary body never dereferences the
  //     chain argument when the counters are in registers);
  //   - an indirect branch that is not the return (a jump table) goes to
  //     the block at that address when the function has one.
  // The ordinary body of a closed function forwards an entry at its first
  // instruction to the closed body. Needs DOLRECOMP_CALL_COUNTERS=registers
  // and DOLRECOMP_ENTRY_LOADS=split, and no native-ABI functions.
  bool closed_ = false;
  // DOLRECOMP_CLOSED_DEFER=1: a closed body does not leave when the runtime's
  // cycle budget is spent or when the guest enables interrupts. The end of an
  // emulator time slice and a pending interrupt are then served when control
  // next reaches the run loop, at the latest when the outermost closed body
  // returns. What still makes a closed body leave on its own account: native
  // call depth (the host stack) at calls, and at loop heads a fixed cap on
  // the cycles of the dispatch, so that a loop waiting for something only an
  // interrupt can change still gets back to the run loop.
  bool closed_defer_ = false;
  bool closedCapable() const;
  bool closedTarget(const DolLLVMFunctionRange *range, u32 target) const;
  llvm::FunctionType *closedFunctionType();
  llvm::FunctionCallee closedBody(u32 start);
  llvm::Value *closedEscaped(llvm::Value *result);
  llvm::Value *withEscape(llvm::Value *result);
  // Hands a callee's result up unchanged: as an escape from a closed body.
  void passResultUp(llvm::Value *result);
  void emitClosedHandover(llvm::Value *host_call);
  // dolrecomp_closed_ok_<address>, one byte per function in a build with a
  // closed set, defined in the function's object: 0 = the closed body must
  // not be entered (the host has not verified the code, or it intercepts the
  // function); 1 = it may; 2 (the initial value) = the host has said nothing,
  // and the closed body may be entered only while no call hook is installed.
  // The host sets it through the module's staticrecomp_set_native_valid. The
  // ordinary body tests it before handing over, and the closed body tests it
  // at its entry (a closed caller calls it directly) and escapes when zero.
  llvm::GlobalVariable *closedPermission();
  static bool closedBuild();
  void emitClosedIndirect(llvm::Value *target, bool through_ctr);
  llvm::BasicBlock *closed_jump_block_ = nullptr;
  llvm::PHINode *closed_jump_target_ = nullptr;
  // DOLRECOMP_GATHER_PIPE=inline: stores to the write-gather pipe's page
  // append to the host's pipe buffer in generated code (memory.cpp).
  bool inline_gather_pipe_ = false;
  bool float_single_math_ = false;
  bool track_singles_ = false;
  SinglePlan *single_plan_ = nullptr;
  bool single_seeds_rejected_ = false;
  // The facts held where a run's closing branch was emitted, by the LLVM
  // block that holds the branch, with the run's leader.
  std::map<llvm::BasicBlock *, std::pair<u32, SingleFacts>> edge_facts_;
  std::vector<SingleFacts> region_out_;
  std::vector<char> region_done_;
  u32 current_region_ = 0;
  // Single-result scalar operations emitted in single precision and in
  // double (DOLRECOMP_SINGLE_FP_STATS=1 prints them per body).
  u32 stat_single_ = 0;
  u32 stat_double_ = 0;
  SingleFacts currentSingleFacts() const;
  void guessSinglePlan();
  void applySingleSeed(u32 index);
  void recordSingleEdge(llvm::BasicBlock *edge = nullptr);
  bool settleSinglePlan();
  void reportSingleStats() const;
  bool usesFPRegisters() const;
  // A double value of the run being emitted and the single it is the
  // widening of (a register read, an lfs load, a sign operation on one).
  struct SingleTwin {
    llvm::Value *single = nullptr;
    bool safe = false;
    bool canonical = false;
  };
  std::map<llvm::Value *, SingleTwin> twins_;
  const SingleTwin *twinOf(llvm::Value *value) const;
  void noteTwin(llvm::Value *value, llvm::Value *single, bool safe,
                bool canonical);
  void noteTrackedLaneWrite(DolIRStateSlot slot, llvm::Value *value,
                            FPRepresentation rep, bool exact, bool safe,
                            bool canonical);
  bool laneView(u32 reg) const;
  // The view exists only because of the switch (an lfs load, a move, a view
  // carried in from another run): the build without the switch would hold a
  // plain double here, so a store must narrow the double as that build does.
  std::array<bool, 32> fp_track_view_{};
  bool storeViewAllowed(u32 reg) const;
  // The single view holds exactly the bits cvtpd2ps of the register gives
  // (results and psq loads do; an lfs value under denormals-are-zero may not).
  std::array<bool, 32> fp_single_canonical_{};
  // The single each half was last widened from, when it was (lfs).
  std::array<llvm::Value *, 32> fp_ps0_float_{};
  std::array<llvm::Value *, 32> fp_ps1_float_{};
};
} // namespace dolllvm

#endif
