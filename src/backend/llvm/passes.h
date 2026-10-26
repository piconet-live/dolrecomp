#ifndef DOLRECOMP_LLVM_PASSES_H
#define DOLRECOMP_LLVM_PASSES_H

#include "backend/llvm/llvm_backend.h"

namespace llvm {
class Module;
class TargetMachine;
class raw_ostream;
} // namespace llvm

namespace dolllvm {

// Generated functions have hundreds of blocks (every memory access has a cold
// path); LLVM's default limits for load forwarding stop well short of that.
void raiseOptimizerLimits();
bool optimizeModule(llvm::Module &module, llvm::TargetMachine &machine,
                    const DolLLVMOptions &options,
                    llvm::raw_ostream &diagnostics);

} // namespace dolllvm

#endif
