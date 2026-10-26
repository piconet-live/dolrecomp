#include "backend/llvm/llvm_backend.h"

#include "backend/codegen_switches.h"

#include "backend/llvm/emitter.h"
#include "backend/llvm/fp_fixups.h"
#include "backend/llvm/native_abi.h"
#include "backend/llvm/passes.h"
#include "backend/llvm/target.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <llvm/Analysis/ModuleSummaryAnalysis.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

namespace {

using namespace llvm;

int selectedCodegenLevel(int ir_level) {
  const char *configured = std::getenv("DOLRECOMP_LLVM_CODEGEN_LEVEL");
  if (!configured || !configured[0])
    return ir_level > 0 ? 2 : 0;
  char *end = nullptr;
  long level = std::strtol(configured, &end, 10);
  return end && !*end && level >= 0 && level <= 3 ? static_cast<int>(level)
                                                  : (ir_level > 0 ? 2 : 0);
}

bool readableProfile(const char *path, FILE *diagnostics) {
  if (!path || !path[0])
    return true;
  FILE *file = fopen(path, "rb");
  if (!file) {
    fprintf(diagnostics, "dolllvm: cannot read profile '%s'\n", path);
    return false;
  }
  fclose(file);
  return true;
}

void applyTargetAttributes(Module &module,
                           const dolllvm::TargetProfile &profile,
                           const DolLLVMOptions &options) {
  for (Function &function : module) {
    if (function.isDeclaration())
      continue;
    function.addFnAttr("target-cpu", profile.cpu);
    if (!profile.features.empty())
      function.addFnAttr("target-features", profile.features);
    function.addFnAttr(Attribute::NoUnwind);
    if (options.semantics == DOLLLVM_SEMANTICS_EXACT)
      function.addFnAttr("strictfp");
  }
}

bool writeIR(Module &module, const DolLLVMOptions &options, FILE *diagnostics) {
  if (!options.emit_ir || !options.ir_path)
    return true;
  std::error_code error;
  raw_fd_ostream file(options.ir_path, error, sys::fs::OF_Text);
  if (error) {
    fprintf(diagnostics, "dolllvm: cannot write IR: %s\n",
            error.message().c_str());
    return false;
  }
  module.print(file, nullptr);
  return true;
}

bool writeObject(Module &module, TargetMachine &machine, const char *path,
                 FILE *diagnostics) {
  std::error_code error;
  raw_fd_ostream file(path, error, sys::fs::OF_None);
  if (error) {
    fprintf(diagnostics, "dolllvm: cannot write object: %s\n",
            error.message().c_str());
    return false;
  }
  legacy::PassManager passes;
  if (machine.addPassesToEmitFile(passes, file, nullptr,
                                  CodeGenFileType::ObjectFile)) {
    fprintf(diagnostics, "dolllvm: target cannot emit objects\n");
    return false;
  }
  passes.run(module);
  file.flush();
  return true;
}

bool writeThinLTO(Module &module, const DolLLVMOptions &options,
                  FILE *diagnostics) {
  if (!options.emit_thinlto || !options.thinlto_path)
    return true;
  std::error_code error;
  raw_fd_ostream file(options.thinlto_path, error, sys::fs::OF_None);
  if (error) {
    fprintf(diagnostics, "dolllvm: cannot write ThinLTO bitcode: %s\n",
            error.message().c_str());
    return false;
  }
  ProfileSummaryInfo profile(module);
  ModuleSummaryIndex index = buildModuleSummaryIndex(module, {}, &profile);
  WriteBitcodeToFile(module, file, false, &index, true);
  file.flush();
  return true;
}

} // namespace

extern "C" bool dolllvm_emit_object(const DolIRModule *source,
                                    const char *object_path,
                                    const DolLLVMOptions *given,
                                    FILE *diagnostics) {
  if (!source || !object_path || !diagnostics)
    return false;
  DolLLVMOptions options{};
  if (given)
    options = *given;
  else
    options.optimization_level = 3;
  std::vector<DolLLVMFunctionRange> ranges;
  if (options.function_ranges && options.function_range_count) {
    ranges.assign(options.function_ranges,
                  options.function_ranges + options.function_range_count);
    dolllvm::prepareModuleABIs(*source, ranges, options.runtime);
    dolllvm_apply_native_abi_policy(ranges.data(),
                                    static_cast<u32>(ranges.size()),
                                    options.native_abi_policy);
    options.function_ranges = ranges.data();
  }
  if (!readableProfile(options.profile_use_path, diagnostics))
    return false;

  dolllvm::TargetProfile profile;
  std::string error;
  if (!dolllvm::resolveTargetProfile(&options, profile, error)) {
    fprintf(diagnostics, "dolllvm: %s\n", error.c_str());
    return false;
  }
  std::unique_ptr<TargetMachine> machine = dolllvm::createTargetMachine(
      profile, selectedCodegenLevel(options.optimization_level),
      options.semantics, error);
  if (!machine) {
    fprintf(diagnostics, "dolllvm: %s\n", error.c_str());
    return false;
  }

  LLVMContext context;
  Module module("dolrecomp_native", context);
  module.setTargetTriple(profile.triple);
  module.setDataLayout(machine->createDataLayout());
  module.setSourceFileName("dolrecomp://deterministic/module");
  std::string text;
  raw_string_ostream stream(text);
  const bool timings = std::getenv("DOLRECOMP_LLVM_TIMINGS") != nullptr;
  auto checkpoint = std::chrono::steady_clock::now();
  auto report = [&](const char *stage) {
    if (!timings)
      return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - checkpoint);
    fprintf(diagnostics, "dolllvm timing %s %s: %lld ms\n", object_path, stage,
            static_cast<long long>(elapsed.count()));
    checkpoint = std::chrono::steady_clock::now();
  };

  for (u32 index = 0; index < source->function_count; index++) {
    const bool imported =
        options.own_function_count && index >= options.own_function_count;
    bool wantsClosed = false;
    // DOLRECOMP_SINGLE_FP=track: a body is emitted with assumptions about
    // the single views at the start of each run and emitted again with
    // narrower ones until the emitter finds them true on every edge of the
    // body it produced (single_track.cpp). The last round but one lets back
    // edges provide nothing; the last assumes nothing at all, which is the
    // build without the switch.
    auto emitBody = [&](bool closedBody) {
      const bool track = dolllvm::FunctionEmitter::trackSingles();
      dolllvm::FunctionEmitter::SinglePlan plan;
      const int rounds = track ? 8 : 1;
      for (int round = 0; round < rounds; round++) {
        dolllvm::FunctionEmitter emitter(context, module,
                                         source->functions[index], options);
        if (imported)
          emitter.setImported();
        if (closedBody)
          emitter.setClosed();
        if (track) {
          plan.forward_only = round >= rounds - 2;
          plan.nothing = round == rounds - 1;
          emitter.setSinglePlan(&plan);
        }
        if (!emitter.emit(stream))
          return false;
        if (emitter.singleSeedsRejected())
          continue;
        if (!closedBody)
          wantsClosed = emitter.wantsClosedBody();
        return true;
      }
      stream << "dolllvm: single-view assumptions did not settle for "
             << source->functions[index].name << "\n";
      return false;
    };
    // DOLRECOMP_INLINE_CLOSED=1: an imported callee of the closed set comes
    // as its closed body (for closed callers to inline), not as its ordinary
    // one.
    bool importedClosed = false;
    if (imported) {
      dolllvm::FunctionEmitter probe(context, module, source->functions[index],
                                     options);
      probe.setImported();
      importedClosed = probe.wantsImportedClosedBody();
    }
    // A function of the closed set gets its closed body beside the ordinary
    // one (DOLRECOMP_CLOSED_ROOTS).
    if (importedClosed ? !emitBody(true)
                       : (!emitBody(false) || (wantsClosed && !emitBody(true)))) {
      stream.flush();
      fprintf(diagnostics, "%s", text.c_str());
      return false;
    }
  }
  dolllvm::raiseOptimizerLimits();
  applyTargetAttributes(module, profile, options);
  // An imported callee's fused multiply-adds end up in its callers.
  bool anyHostFMA = false;
  for (Function &function : module)
    anyHostFMA |= function.hasFnAttribute(dolllvm::kHostFMAAttribute);
  report("lower");
  // DOLRECOMP_FMA=hw. The feature is added for instruction selection only,
  // after the IR optimizer has run (see below).
  // DOLRECOMP_HOST_FEATURES=<LLVM feature string, e.g. "+avx2,+bmi2,+movbe">
  // and DOLRECOMP_HOST_TUNE=<cpu name>: the host's newer instructions and a
  // processor to tune for, for code generation only. Like the "fma" feature
  // they are attached after the IR optimizer has run, so the optimized IR is
  // the baseline's to the bit and only instruction selection, register
  // allocation and scheduling see them. A module built this way must not be
  // loaded on a processor without the features named.
  const char *lateFeatures = std::getenv("DOLRECOMP_HOST_FEATURES");
  const char *lateTune = std::getenv("DOLRECOMP_HOST_TUNE");
  const std::string hostFeatures =
      lateFeatures && lateFeatures[0]
          ? (profile.features.empty() ? std::string(lateFeatures)
                                      : profile.features + "," + lateFeatures)
          : profile.features;
  auto exposeHostFMA = [&] {
    for (Function &function : module) {
      if (!function.hasFnAttribute(dolllvm::kHostFMAAttribute) &&
          !(anyHostFMA && !function.isDeclaration() &&
            function.hasFnAttribute("dolrecomp-guest-body")))
        continue;
      function.removeFnAttr(dolllvm::kHostFMAAttribute);
      function.addFnAttr("target-features", hostFeatures.empty()
                                                ? std::string("+fma")
                                                : hostFeatures + ",+fma");
    }
  };
  auto exposeHostFeatures = [&] {
    for (Function &function : module) {
      if (function.isDeclaration())
        continue;
      if (lateFeatures && lateFeatures[0] &&
          !function.hasFnAttribute(dolllvm::kHostFMAAttribute))
        function.addFnAttr("target-features", hostFeatures);
      if (lateTune && lateTune[0])
        function.addFnAttr("tune-cpu", lateTune);
    }
  };
  // For investigating that: DOLRECOMP_FMA_EARLY=1 lets the optimizer see the
  // feature, and DOLRECOMP_OPT_BISECT=<n> stops the optimizer after its n-th
  // pass (LLVM's -opt-bisect-limit; the pass list goes to stderr).
  if (std::getenv("DOLRECOMP_FMA_EARLY"))
    exposeHostFMA();
  if (verifyModule(module, &stream) ||
      !dolllvm::optimizeModule(module, *machine, options, stream) ||
      verifyModule(module, &stream)) {
    stream.flush();
    fprintf(diagnostics, "%s", text.c_str());
    return false;
  }
  // DOLRECOMP_FMA=hw. The feature is added only now, for instruction
  // selection. When the IR optimizer sees it too (it implies AVX, and the
  // optimized IR then differs widely, with 256-bit vectors throughout), the
  // result was not bit-exact for F-Zero GX against Dolphin's Jit64; the same
  // optimized IR compiled with the feature was. Which IR transformation is
  // responsible has not been isolated.
  exposeHostFeatures();
  exposeHostFMA();
  // DOLRECOMP_PRESERVE_ALL_VZEROUPPER=off: the out-of-line helpers with the
  // preserve_all convention (the fused multiply-add tie corrections, the
  // unusual FPRF classes) are compiled without LLVM's automatic vzeroupper.
  // LLVM 20 saves and restores the YMM registers such a function uses and
  // then puts a vzeroupper in front of its return, which zeroes the upper
  // half of every YMM register: the ones just restored and the ones the
  // caller keeps live across the call because the convention preserves
  // them. As long as no caller holds a 256-bit value that is invisible (the
  // optimizer never sees AVX, so the IR has none); with the "fma" feature
  // visible to the optimizer the SLP vectorizer makes 256-bit values of the
  // guest FP registers, and the two upper lanes come back as zero from a
  // tie correction. That is what broke exactness with DOLRECOMP_FMA_EARLY.
  if (dolrecomp_codegen_switch_is("DOLRECOMP_PRESERVE_ALL_VZEROUPPER", "off")) {
    for (Function &function : module) {
      if (function.isDeclaration() ||
          function.getCallingConv() != CallingConv::PreserveAll)
        continue;
      const std::string features =
          function.getFnAttribute("target-features").getValueAsString().str();
      function.addFnAttr("target-features", features.empty()
                                                ? std::string("-vzeroupper")
                                                : features + ",-vzeroupper");
    }
  }
  report("optimize");
  if (!writeIR(module, options, diagnostics) ||
      !writeThinLTO(module, options, diagnostics) ||
      !writeObject(module, *machine, object_path, diagnostics))
    return false;
  report("codegen");
  return true;
}

extern "C" bool dolllvm_effective_triple(const DolLLVMOptions *options,
                                         char *out, size_t size) {
  if (!out || !size)
    return false;
  dolllvm::TargetProfile profile;
  std::string error;
  if (!dolllvm::resolveTargetProfile(options, profile, error) ||
      profile.triple.size() + 1 > size)
    return false;
  memcpy(out, profile.triple.c_str(), profile.triple.size() + 1);
  return true;
}

extern "C" bool dolllvm_object_matches_options(const char *path,
                                               const DolLLVMOptions *options) {
  dolllvm::TargetProfile profile;
  std::string error;
  return dolllvm::resolveTargetProfile(options, profile, error) &&
         dolllvm::objectMatchesProfile(path, profile);
}

extern "C" bool dolllvm_codegen_fingerprint(const DolLLVMOptions *options,
                                            char *out, size_t size) {
  if (!out || !size)
    return false;
  dolllvm::TargetProfile profile;
  std::string error;
  if (!dolllvm::resolveTargetProfile(options, profile, error))
    return false;
  const int written = std::snprintf(
      out, size,
      "llvm=%s|triple=%s|cpu=%s|features=%s|native-abi=%u|native-policy=%u|"
      "runtime=%u|"
      "mask-words=%u|"
      "state-count=%u|calling=fastcc|control=pc32x2|return=i64-lanes|"
      "escape=sjlj|memory=proven-mem1-v1|cycles=return-or-chain|"
      "x86-return-registers=3|"
      "aarch64-return-registers=8|reloc=pic|pipeline=default-per-module",
      LLVM_VERSION_STRING, profile.triple.c_str(), profile.cpu.c_str(),
      profile.features.c_str(), DOLLLVM_NATIVE_ABI_VERSION,
      static_cast<unsigned>(options ? options->native_abi_policy
                                    : DOLLLVM_NATIVE_ABI_UNRESTRICTED),
      static_cast<unsigned>(options ? options->runtime
                                    : DOLLLVM_RUNTIME_RECOMPCORE),
      DOLIR_STATE_MASK_WORDS, DOLIR_STATE_COUNT);
  return written >= 0 && static_cast<size_t>(written) < size;
}
