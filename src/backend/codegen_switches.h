#ifndef DOLRECOMP_CODEGEN_SWITCHES_H
#define DOLRECOMP_CODEGEN_SWITCHES_H

#include <stdlib.h>
#include <string.h>

/* Environment switches that change the code the LLVM backend generates. Every
   one of them is part of the object cache key (pipeline.c), so an experiment
   can never pick up objects built under a different setting. */
static const char* const dolrecomp_codegen_switches[] = {
    "DOLRECOMP_MEM2_BASE",
    "DOLRECOMP_PRESERVE_NONE",
    "DOLRECOMP_BUDGET_CYCLES",
    "DOLRECOMP_SCALAR_FP",
    "DOLRECOMP_PAIRED_FP",
    "DOLRECOMP_FMA",
    "DOLRECOMP_FPRF",
    "DOLRECOMP_RESERVATION",
    "DOLRECOMP_STATE_SYNC",
    "DOLRECOMP_PSQ_GQR",
    "DOLRECOMP_INLINE_CALLEES",
    "DOLRECOMP_STATE_ALIAS",
    "DOLRECOMP_CYCLES",
    "DOLRECOMP_FPSCR",
    "DOLRECOMP_SINGLE_MATH",
    "DOLRECOMP_INDIRECT_CALLS",
    "DOLRECOMP_INDIRECT_CALL_EXITS",
    "DOLRECOMP_ENTRY_LOADS",
    "DOLRECOMP_ENTRY_CHECK",
    "DOLRECOMP_CALL_COUNTERS",
    "DOLRECOMP_MEM_TRACE",
    "DOLRECOMP_MEM2_TRACK",
    "DOLRECOMP_GATHER_PIPE",
    "DOLRECOMP_CLOSED_ROOTS",
    "DOLRECOMP_CLOSED_DEFER",
    "DOLRECOMP_CLOSED_EXCLUDE",
    "DOLRECOMP_PS_MADD",
    "DOLRECOMP_ENTRY_QUERY",
    "DOLRECOMP_COUNT",
    "DOLRECOMP_HOST_FEATURES",
    "DOLRECOMP_HOST_TUNE",
    "DOLRECOMP_UNMAPPED_READS",
    "DOLRECOMP_INDIRECT_GUARDS",
    "DOLRECOMP_INLINE_DEPTH",
    "DOLRECOMP_INLINE_CLOSED",
    "DOLRECOMP_PARTITION_AT",
    "DOLRECOMP_PRESERVE_ALL_VZEROUPPER",
    "DOLRECOMP_STATE_LAZY",
    "DOLRECOMP_SINGLE_FP",
    /* Not read by the backend: any text, to keep the objects of a modified
       backend apart from those of the build before it. */
    "DOLRECOMP_CODEGEN_REV",
};

/* True when the switch is set to exactly `value`. */
static inline int dolrecomp_codegen_switch_is(const char* name,
                                              const char* value) {
    const char* set = getenv(name);
    return set && strcmp(set, value) == 0;
}

#define DOLRECOMP_CODEGEN_SWITCH_COUNT \
    (sizeof(dolrecomp_codegen_switches) / sizeof(dolrecomp_codegen_switches[0]))

#endif
