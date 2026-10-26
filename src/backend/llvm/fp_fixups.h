#pragma once

namespace llvm {
class Function;
class Module;
class Type;
} // namespace llvm

namespace dolllvm {

llvm::Function *getPairNaNFixup(llvm::Module &module, llvm::Type *pairType);
llvm::Function *getFPRFUnusualFixup(llvm::Module &module, bool wide = false);
llvm::Function *getScalarFMARoundingFixup(llvm::Module &module);
llvm::Function *getPairFMANaNFixup(llvm::Module &module, llvm::Type *pairType);
llvm::Function *getPairFMARoundingFixup(llvm::Module &module,
                                       bool jit64 = false);
void markFMAFixups(llvm::Module &module);

// Marks a function whose fused multiply-adds should use the host's FMA
// instructions (DOLRECOMP_FMA=hw).
inline constexpr const char *kHostFMAAttribute = "dolrecomp-host-fma";

} // namespace dolllvm
