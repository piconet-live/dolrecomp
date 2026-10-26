#include "backend/llvm/passes.h"
#include "backend/codegen_switches.h"

#include <cstdlib>

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Target/TargetMachine.h>

namespace dolllvm {

using namespace llvm;

void raiseOptimizerLimits() {
  static const bool once = [] {
    auto &options = cl::getRegisteredOptions();
    auto set = [&](const char *name, const char *value) {
      auto found = options.find(name);
      if (found != options.end())
        found->second->addOccurrence(1, name, value);
    };
    if (!dolrecomp_codegen_switch_is("DOLRECOMP_STATE_ALIAS", "scoped"))
      return true;
    set("gvn-max-num-deps", "1000");
    set("memdep-block-scan-limit", "500");
    set("memdep-block-number-limit", "2000");
    return true;
  }();
  (void)once;
}

static OptimizationLevel optimizationLevel(int level) {
  if (level <= 0)
    return OptimizationLevel::O0;
  if (level == 1)
    return OptimizationLevel::O1;
  if (level == 2)
    return OptimizationLevel::O2;
  return OptimizationLevel::O3;
}

bool optimizeModule(Module &module, TargetMachine &machine,
                    const DolLLVMOptions &options, raw_ostream &diagnostics) {
  if (options.optimization_level <= 0)
    return true;
  LoopAnalysisManager loops;
  FunctionAnalysisManager functions;
  CGSCCAnalysisManager call_graph;
  ModuleAnalysisManager modules;
  PipelineTuningOptions tuning;
  tuning.LoopVectorization = true;
  tuning.SLPVectorization = true;
  std::optional<PGOOptions> pgo;
  if (options.profile_generate_path && options.profile_generate_path[0]) {
    pgo.emplace(options.profile_generate_path, "", "", "",
                vfs::getRealFileSystem(), PGOOptions::IRInstr);
  } else if (options.profile_use_path && options.profile_use_path[0]) {
    pgo.emplace(options.profile_use_path, "", "", "", vfs::getRealFileSystem(),
                PGOOptions::IRUse);
  }
  // DOLRECOMP_OPT_BISECT=<n>: LLVM's pass gate, for bisecting which pass is
  // responsible for a difference. Every pass is named on stderr.
  PassInstrumentationCallbacks callbacks;
  StandardInstrumentations instrumentations(module.getContext(), false);
  const char *bisect = std::getenv("DOLRECOMP_OPT_BISECT");
  if (bisect && bisect[0]) {
    auto &registered = cl::getRegisteredOptions();
    auto found = registered.find("opt-bisect-limit");
    if (found != registered.end())
      found->second->addOccurrence(1, "opt-bisect-limit", bisect);
    instrumentations.registerCallbacks(callbacks, &modules);
  }
  PassBuilder builder(&machine, tuning, pgo,
                      bisect && bisect[0] ? &callbacks : nullptr);
  builder.registerModuleAnalyses(modules);
  builder.registerCGSCCAnalyses(call_graph);
  builder.registerFunctionAnalyses(functions);
  builder.registerLoopAnalyses(loops);
  builder.crossRegisterProxies(loops, functions, call_graph, modules);
  ModulePassManager pipeline = builder.buildPerModuleDefaultPipeline(
      optimizationLevel(options.optimization_level));
  pipeline.run(module, modules);
  return true;
}

} // namespace dolllvm
