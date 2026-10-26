#include <cstdlib>
#include "backend/llvm/emitter.h"
#include "backend/codegen_switches.h"
#include "backend/llvm/fp_fixups.h"
#include "cpu/cpu.h"

#include <cstdio>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

namespace dolllvm {

using namespace llvm;

FunctionEmitter::FunctionEmitter(LLVMContext &context, Module &module,
                                 const DolIRFunction &source,
                                 const DolLLVMOptions &options)
    : context_(context), module_(module), source_(source), builder_(context),
      ranges_(options.function_ranges),
      range_count_(options.function_range_count),
      entry_points_(options.entry_points),
      entry_point_count_(options.entry_point_count),
      write_journal_(options.instrumentation ==
                     DOLLLVM_INSTRUMENTATION_LOCKSTEP),
      semantics_(options.semantics),
      symbol_suffix_(options.symbol_suffix ? options.symbol_suffix : ""),
      fixed_memory_layout_(options.fixed_memory_layout != 0),
      state_in_memory_(options.state_in_memory != 0),
      modern_runtime_(options.runtime == DOLLLVM_RUNTIME_MODERNGEKKO),
      expected_ram_size_(options.ram_size),
      expected_mem2_size_(options.mem2_size) {
  const Triple triple(module_.getTargetTriple());
  triple_is_x86_ = triple.isX86();
  inline_scalar_fp_ = !modern_runtime_ && dolrecomp_codegen_switch_is(
                                              "DOLRECOMP_SCALAR_FP", "inline");
  jit64_paired_fp_ =
      dolrecomp_codegen_switch_is("DOLRECOMP_PAIRED_FP", "jit64");
  hardware_fma_ = dolrecomp_codegen_switch_is("DOLRECOMP_FMA", "hw");
  guard_paired_fma_ =
      dolrecomp_codegen_switch_is("DOLRECOMP_PS_MADD", "guarded");
  lazy_fprf_ = !modern_runtime_ &&
               dolrecomp_codegen_switch_is("DOLRECOMP_FPRF", "lazy");
  defer_cycles_ = !modern_runtime_ &&
                  dolrecomp_codegen_switch_is("DOLRECOMP_CYCLES", "deferred");
  inline_fpscr_ = !modern_runtime_ && triple_is_x86_ &&
                  dolrecomp_codegen_switch_is("DOLRECOMP_FPSCR", "inline");
  float_single_math_ =
      inline_scalar_fp_ &&
      dolrecomp_codegen_switch_is("DOLRECOMP_SINGLE_MATH", "float");
  track_singles_ = !modern_runtime_ && trackSingles();
  if (track_singles_)
    float_single_math_ = true;
  native_indirect_calls_ =
      !modern_runtime_ &&
      dolrecomp_codegen_switch_is("DOLRECOMP_INDIRECT_CALLS", "native");
  closed_defer_ = dolrecomp_codegen_switch_is("DOLRECOMP_CLOSED_DEFER", "1");
  inline_gather_pipe_ =
      !modern_runtime_ && !native_abi_ && !write_journal_ &&
      dolrecomp_codegen_switch_is("DOLRECOMP_GATHER_PIPE", "inline");
  entry_check_ = !modern_runtime_ && !native_abi_ &&
                 !dolrecomp_codegen_switch_is("DOLRECOMP_ENTRY_CHECK", "off");
  register_counters_ =
      !modern_runtime_ && !native_abi_ &&
      dolrecomp_codegen_switch_is("DOLRECOMP_CALL_COUNTERS", "registers");
  for (u32 i = 0; register_counters_ && i < range_count_; i++)
    if (ranges_[i].abi_flags & DOLLLVM_FUNCTION_ABI_NATIVE)
      register_counters_ = false;
  entry_query_ = !dolrecomp_codegen_switch_is("DOLRECOMP_ENTRY_QUERY", "off");
  split_entry_loads_ =
      !modern_runtime_ && !native_abi_ &&
      dolrecomp_codegen_switch_is("DOLRECOMP_ENTRY_LOADS", "split");
  scoped_alias_ = !modern_runtime_ &&
                  dolrecomp_codegen_switch_is("DOLRECOMP_STATE_ALIAS", "scoped");
  sync_changed_ = !modern_runtime_ && !state_in_memory_ &&
                  dolrecomp_codegen_switch_is("DOLRECOMP_STATE_SYNC", "changed");
  lazy_state_ = sync_changed_ &&
                dolrecomp_codegen_switch_is("DOLRECOMP_STATE_LAZY", "regions");
  intrinsic_escapes_ = triple.isX86() && !modern_runtime_;
  cold_escapes_ = modern_runtime_ || intrinsic_escapes_ ||
                  (triple.isAArch64() && !triple.isOSWindows());
  for (u32 index = 0; index < range_count_; index++) {
    if (ranges_[index].start != source_.guest_start)
      continue;
    abi_range_ = &ranges_[index];
    native_abi_ = (abi_range_->abi_flags & DOLLLVM_FUNCTION_ABI_NATIVE) != 0;
    break;
  }
  if (native_abi_) {
    sync_changed_ = false;
    inline_gather_pipe_ = false;
    if (track_singles_)
      float_single_math_ = dolrecomp_codegen_switch_is("DOLRECOMP_SINGLE_MATH",
                                                       "float");
    track_singles_ = false;
  }
}

bool FunctionEmitter::emit(raw_ostream &diagnostics) {
  if (modern_runtime_ && !native_abi_)
    return true;
  if (closed_ && !closedCapable()) {
    diagnostics << "dolllvm: no closed body possible for " << source_.name
                << "\n";
    return false;
  }
  auto *type = closed_ ? closedFunctionType() : bodyFunctionType(abi_range_);
  const std::string bodyName =
      symbolName(std::string(source_.name) + (closed_ ? "_closed" : "_budget"));
  function_ = module_.getFunction(bodyName);
  if (!function_)
    function_ =
        Function::Create(type, GlobalValue::ExternalLinkage, bodyName, module_);
  if (function_->getFunctionType() != type || !function_->empty()) {
    diagnostics << "dolllvm: conflicting native body " << bodyName << "\n";
    return false;
  }
  function_->setCallingConv(bodyCallingConvention());
  function_->setVisibility(GlobalValue::HiddenVisibility);
  function_->setDSOLocal(true);
  function_->addFnAttr("dolrecomp-guest-body");
  if (!closed_)
    countHere(source_.guest_start, 6,
              (source_.guest_end - source_.guest_start) / 4u, false);
  if (imported_) {
    function_->setLinkage(GlobalValue::AvailableExternallyLinkage);
    function_->addFnAttr(Attribute::AlwaysInline);
  }
  u32 fixedArgument = 0;
  ctx_ = function_->getArg(fixedArgument++);
  ctx_->setName("ctx");
  ctx_->addAttr(Attribute::NonNull);
  ctx_->addAttr(Attribute::NoAlias);
  if (!modern_runtime_)
    ctx_->addAttr(
        Attribute::getWithDereferenceableBytes(context_, sizeof(CPUState)));
  if (modern_runtime_) {
    state_interface_ = function_->getArg(fixedArgument++);
    state_interface_->setName("state_interface");
    state_interface_->addAttr(Attribute::NonNull);
  }
  if (closed_) {
    // Never dereferenced with the counters in registers.
    chain_ = ConstantPointerNull::get(PointerType::getUnqual(context_));
  } else {
    chain_ = function_->getArg(fixedArgument++);
    chain_->setName("chain");
  }
  control_pc_ = function_->getArg(fixedArgument++);
  control_pc_->setName("control_pc");
  if (register_counters_) {
    initial_cycles_ = function_->getArg(fixedArgument++);
    initial_cycles_->setName("cycles_in");
    initial_guard_ = function_->getArg(fixedArgument++);
    initial_guard_->setName("guard_in");
  }
  if (native_abi_) {
    u32 argument = fixedArgument;
    if (nativeCyclesInResult(abi_range_)) {
      initial_cycles_ = function_->getArg(argument++);
      initial_cycles_->setName("initial_cycles");
    }
    for (u32 slot = 0; slot < DOLIR_STATE_COUNT; slot++) {
      auto stateSlot = static_cast<DolIRStateSlot>(slot);
      if (!stateInput(abi_range_, stateSlot))
        continue;
      native_inputs_[slot] = function_->getArg(argument++);
      native_inputs_[slot]->setName("in.s" + std::to_string(slot));
    }
  }

  entry_ = BasicBlock::Create(context_, "entry", function_);
  fallback_block_ =
      BasicBlock::Create(context_, "fallback_dispatch", function_);
  IRBuilder<> fallbackBuilder(fallback_block_);
  fallback_pc_ =
      fallbackBuilder.CreatePHI(Type::getInt32Ty(context_), 0, "fallback_pc");
  scanState();
  scanStableControls();
  scanContinuations();
  scanLoopHeaders();
  scanRegionLeaders();
  scanLazyState();
  if (track_singles_ && single_plan_) {
    if (usesFPRegisters()) {
      region_out_.assign(source_.block_count, SingleFacts());
      region_done_.assign(source_.block_count, 0);
      guessSinglePlan();
    } else {
      single_plan_ = nullptr;
    }
  }
  blocks_.resize(source_.block_count);
  BasicBlock *region = nullptr;
  for (u32 i = 0; i < source_.block_count; i++) {
    if (source_.blocks[i].terminator.kind == DOLIR_TERM_FALLBACK) {
      blocks_[i] = fallbackEdge(source_.blocks[i].guest_address);
      region = nullptr;
      continue;
    }
    if (region_leaders_[i])
      region = BasicBlock::Create(context_, blockName(i), function_);
    if (!region) {
      diagnostics << "dolllvm: instruction 0x"
                  << format_hex_no_prefix(source_.blocks[i].guest_address, 8)
                  << " has no native region\n";
      return false;
    }
    blocks_[i] = region;
  }
  emitEntry();
  for (u32 i = 0; i < source_.block_count; i++) {
    if (!region_leaders_[i] ||
        source_.blocks[i].terminator.kind == DOLIR_TERM_FALLBACK)
      continue;
    if (!emitRegion(i, diagnostics))
      return false;
  }
  emitFallbackHandler();
  if (track_singles_ && single_plan_ && !settleSinglePlan()) {
    // The assumptions about single views did not hold on every edge; they
    // have been narrowed. Drop this body; the caller emits it again.
    single_seeds_rejected_ = true;
    function_->deleteBody();
    const std::string prefix = std::string(source_.name) + "_fallback_";
    SmallVector<GlobalVariable *, 4> stale;
    for (GlobalVariable &global : module_.globals()) {
      if (!global.hasPrivateLinkage() || !global.getName().starts_with(prefix))
        continue;
      global.removeDeadConstantUsers();
      if (global.use_empty())
        stale.push_back(&global);
    }
    for (GlobalVariable *global : stale)
      global->eraseFromParent();
    return true;
  }
  if (track_singles_)
    reportSingleStats();
  finalizeStateSSA();
  if (hardware_fma_ && uses_fma_ && triple_is_x86_) {
    // Only where a fused multiply-add is emitted. The backend turns this mark
    // into the "fma" target feature after the IR has been optimized, so that
    // instruction selection is the only thing that sees it.
    function_->addFnAttr(kHostFMAAttribute);
    markFMAFixups(module_);
  }
  if (verifyFunction(*function_, &diagnostics))
    return false;
  if (imported_ || closed_)
    return true;
  return emitWrapper(diagnostics);
}

std::string FunctionEmitter::blockName(u32 index) const {
  char text[40];
  snprintf(text, sizeof(text), "guest_%08X_b%u",
           source_.blocks[index].guest_address, index);
  return text;
}

std::string FunctionEmitter::symbolName(StringRef base) const {
  return base.str() + symbol_suffix_;
}

Type *FunctionEmitter::type(DolIRType t) {
  switch (t) {
  case DOLIR_TYPE_I1:
    return Type::getInt1Ty(context_);
  case DOLIR_TYPE_I8:
    return Type::getInt8Ty(context_);
  case DOLIR_TYPE_I16:
    return Type::getInt16Ty(context_);
  case DOLIR_TYPE_I32:
    return Type::getInt32Ty(context_);
  case DOLIR_TYPE_I64:
    return Type::getInt64Ty(context_);
  case DOLIR_TYPE_F32:
    return Type::getFloatTy(context_);
  case DOLIR_TYPE_F64:
    return Type::getDoubleTy(context_);
  case DOLIR_TYPE_V2F32:
    return FixedVectorType::get(Type::getFloatTy(context_), 2);
  case DOLIR_TYPE_V2F64:
    return FixedVectorType::get(Type::getDoubleTy(context_), 2);
  default:
    return Type::getVoidTy(context_);
  }
}

size_t FunctionEmitter::stateOffset(DolIRStateSlot slot) const {
  if (slot >= DOLIR_STATE_GPR0 && slot <= DOLIR_STATE_GPR31)
    return offsetof(CPUState, gpr) + 4u * (slot - DOLIR_STATE_GPR0);
  if (slot >= DOLIR_STATE_FPR0 && slot <= DOLIR_STATE_FPR31)
    return offsetof(CPUState, fpr) + 8u * (slot - DOLIR_STATE_FPR0);
  if (slot >= DOLIR_STATE_PS1_0 && slot <= DOLIR_STATE_PS1_31)
    return offsetof(CPUState, ps1) + 8u * (slot - DOLIR_STATE_PS1_0);
  if (slot >= DOLIR_STATE_SR0 && slot <= DOLIR_STATE_SR15)
    return offsetof(CPUState, sr) + 4u * (slot - DOLIR_STATE_SR0);
  if (slot >= DOLIR_STATE_GQR0 && slot <= DOLIR_STATE_GQR7)
    return offsetof(CPUState, gqr) + 4u * (slot - DOLIR_STATE_GQR0);
  switch (slot) {
  case DOLIR_STATE_PC:
    return offsetof(CPUState, pc);
  case DOLIR_STATE_LR:
    return offsetof(CPUState, lr);
  case DOLIR_STATE_CTR:
    return offsetof(CPUState, ctr);
  case DOLIR_STATE_CR:
    return offsetof(CPUState, cr);
  case DOLIR_STATE_XER:
    return offsetof(CPUState, xer);
  case DOLIR_STATE_FPSCR:
    return offsetof(CPUState, fpscr);
  case DOLIR_STATE_MSR:
    return offsetof(CPUState, msr);
  case DOLIR_STATE_SRR0:
    return offsetof(CPUState, srr0);
  case DOLIR_STATE_SRR1:
    return offsetof(CPUState, srr1);
  case DOLIR_STATE_DAR:
    return offsetof(CPUState, dar);
  case DOLIR_STATE_DSISR:
    return offsetof(CPUState, dsisr);
  case DOLIR_STATE_EAR:
    return offsetof(CPUState, ear);
  case DOLIR_STATE_HID2:
    return offsetof(CPUState, hid2);
  case DOLIR_STATE_TIMEBASE:
    return offsetof(CPUState, timebase);
  case DOLIR_STATE_EXCEPTION:
    return offsetof(CPUState, exception);
  case DOLIR_STATE_PROGRAM_EXCEPTION:
    return offsetof(CPUState, program_exception);
  case DOLIR_STATE_RESERVE_ADDR:
    return offsetof(CPUState, reserve_addr);
  case DOLIR_STATE_RESERVE_VALID:
    return offsetof(CPUState, reserve_valid);
  case DOLIR_STATE_DOWNCOUNT:
    return offsetof(CPUState, downcount);
  default:
    return 0;
  }
}

} // namespace dolllvm
