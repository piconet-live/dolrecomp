#include "backend/llvm/emitter.h"
#include "backend/codegen_switches.h"

#include <cstdio>

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>

// DOLRECOMP_COUNT=1 (analysis builds only): the module counts what it runs.
//
// Every counted place owns one record {u32 guest address, u32 info, u64
// count} in the section ".dolcnt$m", which the linker lays out as one array
// between the records a module's own C code puts into ".dolcnt$a" and
// ".dolcnt$z"; that code writes the array to a file when the process ends.
// info = kind << 24 | detail:
//   0  a straight run of an ordinary body entered (detail: its length in
//      guest instructions; every instruction of a run executes once per
//      entry, apart from a run left through an exception)
//   1  the same for a closed body
//   2  the host's floating-point mode loaded (MXCSR) for a changed rounding
//      mode or non-IEEE bit, at this guest instruction
//   3  an indirect call made inside generated code at this site
//   4  a load served by the runtime (neither memory window), 5 a store
//   6  not a count: a function's extent (address: its first instruction,
//      detail: its length in instructions), for attributing runs to functions
// Loads and stores also count into dolrecomp_ea_hist[write][address >> 20],
// which the module defines: where memory accesses point, by megabyte.
//
// A count is a load, an add and a store in the block it counts: no branch,
// so a function's control flow is what it is without the switch.

namespace dolllvm {

using namespace llvm;

bool FunctionEmitter::counting() const {
  static const bool enabled =
      dolrecomp_codegen_switch_is("DOLRECOMP_COUNT", "1");
  return enabled && !modern_runtime_ && !imported_;
}

GlobalVariable *FunctionEmitter::countHere(u32 address, u32 kind, u32 detail,
                                           bool increment) {
  if (!counting())
    return nullptr;
  Type *i32 = Type::getInt32Ty(context_);
  Type *i64 = Type::getInt64Ty(context_);
  StructType *recordType = StructType::get(context_, {i32, i32, i64});
  char name[64];
  // External and hidden: the optimizer leaves the fields of such a record
  // together. The sequence number keeps two counts of one kind at one
  // address apart (an instruction can hold two accesses).
  snprintf(name, sizeof(name), "dolrecomp_cnt_%08X_%u%s_%u", address, kind,
           closed_ ? "c" : "", count_sequence_++);
  auto *record = new GlobalVariable(
      module_, recordType, false, GlobalValue::ExternalLinkage,
      ConstantStruct::get(recordType,
                          {builder_.getInt32(address),
                           builder_.getInt32(kind << 24 | (detail & 0xFFFFFFu)),
                           builder_.getInt64(increment ? 0 : 1)}),
      symbolName(name));
  record->setVisibility(GlobalValue::HiddenVisibility);
  record->setDSOLocal(true);
  record->setSection(".dolcnt$m");
  record->setAlignment(Align(16));
  if (!increment)
    return record;
  Value *slot = builder_.CreateStructGEP(recordType, record, 2);
  builder_.CreateStore(
      builder_.CreateAdd(builder_.CreateLoad(i64, slot), builder_.getInt64(1)),
      slot);
  return record;
}

void FunctionEmitter::setCountDetail(GlobalVariable *record, u32 kind,
                                     u32 detail) {
  if (!record)
    return;
  auto *initial = cast<ConstantStruct>(record->getInitializer());
  record->setInitializer(ConstantStruct::get(
      cast<StructType>(record->getValueType()),
      {initial->getOperand(0),
       builder_.getInt32(kind << 24 | (detail & 0xFFFFFFu)),
       initial->getOperand(2)}));
}

void FunctionEmitter::countAccess(Value *address, bool write) {
  if (!counting())
    return;
  Type *i64 = Type::getInt64Ty(context_);
  ArrayType *tableType = ArrayType::get(i64, 2 * 4096);
  auto *table = cast<GlobalVariable>(
      module_.getOrInsertGlobal("dolrecomp_ea_hist", tableType));
  table->setVisibility(GlobalValue::HiddenVisibility);
  table->setDSOLocal(true);
  Value *index = builder_.CreateAdd(
      builder_.CreateZExt(builder_.CreateLShr(address, builder_.getInt32(20)),
                          i64),
      builder_.getInt64(write ? 4096 : 0));
  Value *slot = builder_.CreateInBoundsGEP(tableType, table,
                                           {builder_.getInt64(0), index});
  builder_.CreateStore(
      builder_.CreateAdd(builder_.CreateLoad(i64, slot), builder_.getInt64(1)),
      slot);
}

} // namespace dolllvm
