#include "app/pipeline.h"
#include "analysis/code_section.h"
#include "analysis/embedded_data.h"
#include "analysis/smc.h"
#include "app/paths.h"
#include "backend/codegen.h"
#include "backend/codegen_switches.h"
#include "backend/dispatch.h"
#include "backend/emitter.h"
#include "backend/native_module.h"
#include "backend/symbols.h"
#include "backend/variant_output.h"
#include "frontend/container/dol.h"
#include "frontend/container/rel.h"
#include "frontend/container/rpx.h"
#include "frontend/decoder.h"
#include "platform/fs.h"
#include "platform/pathlist.h"
#include "platform/strutil.h"
#ifdef DOLRECOMP_ENABLE_LLVM
#include "backend/llvm/llvm_backend.h"
#include "cpu/cpu.h"
#include "ir/dolir_builder.h"
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#else
#include <process.h>
#endif

#define DOLC_DEFAULT_CHUNK_INSTRUCTIONS 4096u

static u32 c_chunk_instructions(void) {
    const char* configured = getenv("DOLRECOMP_C_CHUNK_INSTRUCTIONS");
    if (!configured || !configured[0])
        return DOLC_DEFAULT_CHUNK_INSTRUCTIONS;

    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(configured, &end, 10);
    if (errno || !end || *end || value < 128u || value > 4096u) {
        fprintf(stderr,
                "warning: DOLRECOMP_C_CHUNK_INSTRUCTIONS must be 128..4096; "
                "using %u\n",
                DOLC_DEFAULT_CHUNK_INSTRUCTIONS);
        return DOLC_DEFAULT_CHUNK_INSTRUCTIONS;
    }
    return (u32)value;
}

#ifdef DOLRECOMP_ENABLE_LLVM
#define DOLLLVM_DEFAULT_CHUNK_INSTRUCTIONS 128u
#define DOLLLVM_DEFAULT_WORKER_BATCH 1u
// SSA regions, ABI v4 variants and ThinLTO summaries.
#define DOLLLVM_CACHE_VERSION "dolllvm-v29"

#define LLVM_MAX_IMPORTS 32u

typedef struct {
    u32 start;
    u32 count;
} LLVMImport;

typedef struct {
    const PPCInst* insts;
    u32 count;
    u32 function_address;
    u32 index;
    u32 total;
    const DolLLVMFunctionRange* ranges;
    u32 range_count;
    const u32* entry_points;
    u32 entry_point_count;
    DolLLVMTargetProfile target_profile;
    DolLLVMSemantics semantics;
    DolLLVMInstrumentation instrumentation;
    DolLLVMNativeABIPolicy native_abi_policy;
    DolLLVMRuntime runtime;
    const char* profile_generate_path;
    const char* profile_use_path;
    u64 partition_seed;
    int state_in_memory;
    u32 ram_size;
    u32 mem2_size;
    char symbol_suffix[32];
    char thinlto_path[1400];
    u64 hash;
    char name[128];
    char path[1400];
    char cache_path[1400];
    char cache_bitcode_path[1400];
    /* Small callees whose code is also given to this chunk's module, as
       definitions LLVM may inline but does not emit (their own chunks still
       define them). */
    const LoadedCodeSection* sections;
    u32 section_count;
    LLVMImport imports[LLVM_MAX_IMPORTS];
    u32 import_count;
} LLVMChunkJob;

static u32 parse_llvm_target_set(const char* text,
                                 DolLLVMTargetProfile profiles[5]) {
    char copy[128];
    if (!text || strlen(text) >= sizeof(copy))
        return 0;
    snprintf(copy, sizeof(copy), "%s", text);
    u32 count = 0;
    char* cursor = copy;
    while (cursor && *cursor && count < 5u) {
        char* comma = strchr(cursor, ',');
        if (comma)
            *comma = '\0';
        if (!dolllvm_parse_target_profile(cursor, &profiles[count]))
            return 0;
        for (u32 i = 0; i < count; i++)
            if (profiles[i] == profiles[count])
                return 0;
        count++;
        cursor = comma ? comma + 1 : NULL;
    }
    return count;
}

static u32 llvm_chunk_instructions(void) {
    const char* configured = getenv("DOLRECOMP_LLVM_CHUNK_INSTRUCTIONS");
    if (!configured || !configured[0])
        return DOLLLVM_DEFAULT_CHUNK_INSTRUCTIONS;
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(configured, &end, 10);
    if (errno || !end || *end || value < 64u || value > 4096u) {
        fprintf(stderr,
                "warning: DOLRECOMP_LLVM_CHUNK_INSTRUCTIONS must be 64..4096; "
                "using %u\n",
                DOLLLVM_DEFAULT_CHUNK_INSTRUCTIONS);
        return DOLLLVM_DEFAULT_CHUNK_INSTRUCTIONS;
    }
    return (u32)value;
}

static u32 llvm_worker_batch_size(void) {
    const char* configured = getenv("DOLRECOMP_LLVM_WORKER_BATCH");
    if (!configured || !configured[0])
        return DOLLLVM_DEFAULT_WORKER_BATCH;
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(configured, &end, 10);
    if (errno || !end || *end || value < 1u || value > 64u) {
        fprintf(stderr,
                "warning: DOLRECOMP_LLVM_WORKER_BATCH must be 1..64; "
                "using %u\n",
                DOLLLVM_DEFAULT_WORKER_BATCH);
        return DOLLLVM_DEFAULT_WORKER_BATCH;
    }
    return (u32)value;
}

// Validate the object format selected by the target triple.
static int valid_object_file(const LLVMChunkJob* job, const char* path) {
    DolLLVMOptions options = {0};
    options.target_triple = getenv("DOLRECOMP_LLVM_TARGET");
    options.target_profile = job->target_profile;
    return dolllvm_object_matches_options(path, &options) ? 1 : 0;
}

static int llvm_job_stamp_path(const LLVMChunkJob* job, char* path,
                               size_t size) {
    int written = snprintf(path, size, "%s.hash", job->path);
    return written > 0 && written < (int)size;
}

static int valid_llvm_job_stamp(const LLVMChunkJob* job) {
    char path[1440];
    if (!llvm_job_stamp_path(job, path, sizeof(path)))
        return 0;
    FILE* file = fopen(path, "r");
    if (!file)
        return 0;
    unsigned long long hash = 0;
    int valid = fscanf(file, "%llx", &hash) == 1 && hash == job->hash;
    fclose(file);
    return valid;
}

static void write_llvm_job_stamp(const LLVMChunkJob* job) {
    char path[1440];
    char temp[1480];
    if (!llvm_job_stamp_path(job, path, sizeof(path)))
        return;
#ifdef _WIN32
    int process_id = _getpid();
#else
    int process_id = (int)getpid();
#endif
    if (snprintf(temp, sizeof(temp), "%s.tmp.%d", path, process_id) >=
        (int)sizeof(temp))
        return;
    FILE* file = fopen(temp, "w");
    if (!file)
        return;
    int ok = fprintf(file, "%016llx\n", (unsigned long long)job->hash) > 0;
    if (fclose(file) != 0)
        ok = 0;
    if (!ok) {
        remove(temp);
        return;
    }
    remove(path);
    if (rename(temp, path) != 0)
        remove(temp);
}

static int copy_file(const char* source, const char* destination) {
    FILE* in = fopen(source, "rb");
    if (!in)
        return 0;
    FILE* out = fopen(destination, "wb");
    if (!out) {
        fclose(in);
        return 0;
    }
    unsigned char buffer[64 * 1024];
    int ok = 1;
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), in)) != 0) {
        if (fwrite(buffer, 1, count, out) != count) {
            ok = 0;
            break;
        }
    }
    if (ferror(in))
        ok = 0;
    if (fclose(out) != 0)
        ok = 0;
    fclose(in);
    if (!ok)
        remove(destination);
    return ok;
}

static u64 hash_bytes(u64 hash, const void* data, size_t size) {
    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static u64 hash_file_contents(u64 hash, const char* path) {
    if (!path || !path[0])
        return hash;
    FILE* file = fopen(path, "rb");
    if (!file)
        return hash_bytes(hash, path, strlen(path));
    unsigned char bytes[4096];
    size_t count;
    while ((count = fread(bytes, 1, sizeof(bytes), file)) != 0)
        hash = hash_bytes(hash, bytes, count);
    fclose(file);
    return hash;
}

static const u8* guest_code(const LoadedCodeSection* sections,
                            u32 section_count, u32 start, u32 count) {
    for (u32 s = 0; s < section_count; s++) {
        const LoadedCodeSection* section = &sections[s];
        if (section->data && start >= section->address &&
            start - section->address + count * 4u <= section->size)
            return section->data + (start - section->address);
    }
    return NULL;
}

static PPCInst* decode_guest_code(const LoadedCodeSection* sections,
                                  u32 section_count, u32 start, u32 count) {
    const LoadedCodeSection* section = NULL;
    for (u32 s = 0; s < section_count; s++)
        if (sections[s].data && start >= sections[s].address &&
            start - sections[s].address + count * 4u <= sections[s].size)
            section = &sections[s];
    if (!section)
        return NULL;
    PPCInst* insts = (PPCInst*)malloc((size_t)count * sizeof(*insts));
    if (!insts)
        return NULL;
    for (u32 i = 0; i < count; i++) {
        u32 raw = read_be32(section->data + (start - section->address) + i * 4u);
        insts[i] = ppc_decode(raw, start + i * 4u);
        if (insts[i].op == PPC_OP_UNKNOWN &&
            embedded_data_word(section->embedded_data_mode, raw))
            insts[i].embedded_data = true;
    }
    return insts;
}

/* DOLRECOMP_INLINE_CALLEES=N: a chunk of at most N guest instructions that
   this chunk calls or branches to directly (and, to a small depth, what
   those call) is compiled into this chunk's module too, so LLVM can inline
   it. 0 or unset: off. */
static u32 llvm_inline_callee_limit(void) {
    const char* configured = getenv("DOLRECOMP_INLINE_CALLEES");
    if (!configured || !configured[0])
        return 0;
    char* end = NULL;
    unsigned long value = strtoul(configured, &end, 10);
    return end && !*end && value <= 4096u ? (u32)value : 0;
}

static const DolLLVMFunctionRange* llvm_range_at(
    const DolLLVMFunctionRange* ranges, u32 range_count, u32 address) {
    u32 lo = 0;
    u32 hi = range_count;
    while (lo < hi) {
        u32 mid = lo + (hi - lo) / 2u;
        if (address < ranges[mid].start)
            hi = mid;
        else if (address >= ranges[mid].end)
            lo = mid + 1u;
        else
            return &ranges[mid];
    }
    return NULL;
}

static void collect_llvm_imports(LLVMChunkJob* job, const PPCInst* insts,
                                 u32 count, u32 address, u32 limit) {
    job->import_count = 0;
    if (!limit)
        return;
    /* Breadth first from this chunk; entries added to the list are scanned
       in turn, to a depth of three calls (DOLRECOMP_INLINE_DEPTH=<1..24>
       for another depth: the compiler's register save and restore helpers
       are chains of up to eighteen one-instruction functions, each falling
       into the next). */
    u32 max_depth = 3u;
    /* Sixteen imports a chunk, as before the switch existed; a deeper import
       may take all LLVM_MAX_IMPORTS. */
    u32 max_imports = 16u;
    const char* configured_depth = getenv("DOLRECOMP_INLINE_DEPTH");
    if (configured_depth && configured_depth[0]) {
        char* end = NULL;
        unsigned long value = strtoul(configured_depth, &end, 10);
        if (end && !*end && value >= 1u && value <= 24u) {
            max_depth = (u32)value;
            max_imports = LLVM_MAX_IMPORTS;
        }
    }
    u32 scanned = 0;
    for (u32 depth = 0; depth < max_depth; depth++) {
        u32 first = depth == 0 ? 0 : scanned;
        u32 last = depth == 0 ? 1 : job->import_count;
        for (u32 item = first; item < last; item++) {
            const PPCInst* code = insts;
            u32 code_count = count;
            u32 code_address = address;
            PPCInst* owned = NULL;
            if (depth != 0) {
                code_count = job->imports[item].count;
                code_address = job->imports[item].start;
                owned = decode_guest_code(job->sections, job->section_count,
                                          code_address, code_count);
                code = owned;
                if (!code)
                    continue;
            }
            DolIRModule module;
            dolir_module_init(&module);
            if (dolir_build_chunk(&module, code, code_count, code_address)) {
                const DolIRFunction* function = &module.functions[0];
                for (u32 b = 0; b < function->block_count; b++) {
                    const DolIRTerminator* term =
                        &function->blocks[b].terminator;
                    u32 targets = term->kind == DOLIR_TERM_COND_BRANCH ? 2u
                                  : term->kind == DOLIR_TERM_BRANCH ||
                                          term->kind == DOLIR_TERM_FALLTHROUGH
                                      ? 1u
                                      : 0u;
                    for (u32 t = 0; t < targets; t++) {
                        if (term->targets[t] != DOLIR_NO_BLOCK)
                            continue;
                        const DolLLVMFunctionRange* target = llvm_range_at(
                            job->ranges, job->range_count,
                            term->target_addresses[t]);
                        if (!target || target->start == address ||
                            (target->end - target->start) / 4u > limit ||
                            job->import_count >= max_imports)
                            continue;
                        bool known = false;
                        for (u32 i = 0; i < job->import_count; i++)
                            known |= job->imports[i].start == target->start;
                        if (known)
                            continue;
                        job->imports[job->import_count].start = target->start;
                        job->imports[job->import_count].count =
                            (target->end - target->start) / 4u;
                        job->import_count++;
                    }
                }
            }
            dolir_module_free(&module);
            free(owned);
        }
        if (depth != 0)
            scanned = last;
    }
}

static u64 llvm_job_hash(const LLVMChunkJob* job) {
    u64 hash = 1469598103934665603ull;
    hash =
        hash_bytes(hash, DOLLLVM_CACHE_VERSION, strlen(DOLLLVM_CACHE_VERSION));
#ifdef DOLRECOMP_CODEGEN_SOURCE_HASH
    /* A hand-edited version string cannot invalidate the cache when the code
       that generates the objects changes -- it only invalidates when somebody
       remembers to bump it. CMake hashes the emitter sources and passes the
       digest in, so editing any of them changes every job key automatically. */
    hash = hash_bytes(hash, DOLRECOMP_CODEGEN_SOURCE_HASH,
                      strlen(DOLRECOMP_CODEGEN_SOURCE_HASH));
#endif
    hash =
        hash_bytes(hash, &job->function_address, sizeof(job->function_address));
    hash = hash_bytes(hash, &job->count, sizeof(job->count));
    u32 state_size = (u32)sizeof(CPUState);
    hash = hash_bytes(hash, &state_size, sizeof(state_size));
    hash = hash_bytes(hash, &job->target_profile, sizeof(job->target_profile));
    hash = hash_bytes(hash, job->symbol_suffix, strlen(job->symbol_suffix));
    hash = hash_bytes(hash, &job->semantics, sizeof(job->semantics));
    hash =
        hash_bytes(hash, &job->instrumentation, sizeof(job->instrumentation));
    hash = hash_bytes(hash, &job->native_abi_policy,
                      sizeof(job->native_abi_policy));
    hash = hash_bytes(hash, &job->runtime, sizeof(job->runtime));
    hash = hash_bytes(hash, &job->partition_seed, sizeof(job->partition_seed));
    hash =
        hash_bytes(hash, &job->state_in_memory, sizeof(job->state_in_memory));
    hash = hash_bytes(hash, &job->ram_size, sizeof(job->ram_size));
    hash = hash_bytes(hash, &job->mem2_size, sizeof(job->mem2_size));
    hash = hash_file_contents(hash, job->profile_use_path);
    if (job->profile_generate_path)
        hash = hash_bytes(hash, job->profile_generate_path,
                          strlen(job->profile_generate_path));
    const char* codegen_level = getenv("DOLRECOMP_LLVM_CODEGEN_LEVEL");
    if (!codegen_level || !codegen_level[0])
        codegen_level = "2";
    hash = hash_bytes(hash, codegen_level, strlen(codegen_level));
    /* Codegen environment switches key the cache too. */
    for (size_t e = 0; e < DOLRECOMP_CODEGEN_SWITCH_COUNT; e++) {
        const char* name = dolrecomp_codegen_switches[e];
        const char* v = getenv(name);
        hash = hash_bytes(hash, name, strlen(name));
        if (v)
            hash = hash_bytes(hash, v, strlen(v));
    }
    const char* write_journal = getenv("DOLRECOMP_LLVM_WRITE_JOURNAL");
    if (!write_journal)
        write_journal = "0";
    hash = hash_bytes(hash, write_journal, strlen(write_journal));
    // Host triples must distinguish caches when no target was requested.
    char triple[256];
    DolLLVMOptions target_options = {0};
    target_options.target_triple = getenv("DOLRECOMP_LLVM_TARGET");
    target_options.target_profile = job->target_profile;
    target_options.native_abi_policy = job->native_abi_policy;
    target_options.runtime = job->runtime;
    if (dolllvm_effective_triple(&target_options, triple, sizeof(triple)))
        hash = hash_bytes(hash, triple, strlen(triple));
    char codegen[1024];
    if (dolllvm_codegen_fingerprint(&target_options, codegen, sizeof(codegen)))
        hash = hash_bytes(hash, codegen, strlen(codegen));
    for (u32 i = 0; i < job->count; i++) {
        hash = hash_bytes(hash, &job->insts[i].address,
                          sizeof(job->insts[i].address));
        hash = hash_bytes(hash, &job->insts[i].raw, sizeof(job->insts[i].raw));
        hash = hash_bytes(hash, &job->insts[i].embedded_data,
                          sizeof(job->insts[i].embedded_data));
    }
    for (u32 i = 0; i < job->range_count; i++)
        hash = hash_bytes(hash, &job->ranges[i], sizeof(job->ranges[i]));
    hash = hash_bytes(hash, &job->import_count, sizeof(job->import_count));
    for (u32 i = 0; i < job->import_count; i++) {
        hash = hash_bytes(hash, &job->imports[i], sizeof(job->imports[i]));
        const u8* code = guest_code(job->sections, job->section_count,
                                    job->imports[i].start,
                                    job->imports[i].count);
        if (code)
            hash = hash_bytes(hash, code, (size_t)job->imports[i].count * 4u);
    }
    for (u32 i = 0; i < job->entry_point_count; i++) {
        u32 address = job->entry_points[i];
        if (address >= job->function_address &&
            address < job->function_address + job->count * 4u)
            hash = hash_bytes(hash, &address, sizeof(address));
    }
    return hash;
}

static int compare_u32(const void* left, const void* right) {
    u32 a = *(const u32*)left;
    u32 b = *(const u32*)right;
    return a < b ? -1 : a > b;
}

static int llvm_code_address(const LoadedCodeSection* sections,
                             u32 section_count, u32 address) {
    for (u32 i = 0; i < section_count; i++) {
        const LoadedCodeSection* section = &sections[i];
        if (section->data && address >= section->address &&
            address < section->address + section->size &&
            ((address - section->address) & 3u) == 0)
            return 1;
    }
    return 0;
}

static u32* collect_llvm_partition_points(const LoadedCodeSection* sections,
                                          u32 section_count, u32 program_entry,
                                          const DolRecompSymbolMap* symbols,
                                          u32* result_count) {
    /* DOLRECOMP_PARTITION_AT=<hex address>[,...]: further places where a
       function starts. For code that is entered by a plain branch from other
       functions in the middle of what would otherwise be one function (several
       small routines sharing one tail): an entry anywhere but at a function's
       first instruction goes through the entry switch of the ordinary body,
       with every register reloaded, and never reaches a closed body. */
    const char* extra_points = getenv("DOLRECOMP_PARTITION_AT");
    size_t capacity = 1u + (symbols ? symbols->count : 0u) +
                      (extra_points ? strlen(extra_points) / 2u + 1u : 0u);
    for (u32 i = 0; i < section_count; i++)
        capacity += sections[i].size / 4u;
    u32* points = (u32*)malloc(capacity * sizeof(*points));
    if (!points)
        return NULL;
    u32 count = 0;
    if (llvm_code_address(sections, section_count, program_entry))
        points[count++] = program_entry;
    while (extra_points && *extra_points) {
        char* end = NULL;
        unsigned long address = strtoul(extra_points, &end, 16);
        if (end == extra_points)
            break;
        if (llvm_code_address(sections, section_count, (u32)address))
            points[count++] = (u32)address;
        extra_points = *end == ',' ? end + 1 : end;
    }
    if (symbols) {
        for (u32 i = 0; i < symbols->count; i++) {
            u32 address = symbols->symbols[i].address;
            if (llvm_code_address(sections, section_count, address))
                points[count++] = address;
        }
    }
    for (u32 section_index = 0; section_index < section_count;
         section_index++) {
        const LoadedCodeSection* section = &sections[section_index];
        for (u32 offset = 0; section->data && offset + 4u <= section->size;
             offset += 4u) {
            u32 address = section->address + offset;
            PPCInst inst =
                ppc_decode(read_be32(section->data + offset), address);
            if ((inst.op == PPC_OP_B || inst.op == PPC_OP_BC) && inst.lk &&
                llvm_code_address(sections, section_count, inst.branch_target))
                points[count++] = inst.branch_target;
            if (inst.op == PPC_OP_BCLR && !inst.lk && inst.bo == 20u &&
                llvm_code_address(sections, section_count, address + 4u))
                points[count++] = address + 4u;
        }
    }
    qsort(points, count, sizeof(*points), compare_u32);
    u32 unique = 0;
    for (u32 i = 0; i < count; i++) {
        if (!unique || points[i] != points[unique - 1u])
            points[unique++] = points[i];
    }
    *result_count = unique;
    return points;
}

static u32 next_llvm_partition_point(const u32* points, u32 count, u32 start,
                                     u32 end) {
    u32 low = 0;
    u32 high = count;
    while (low < high) {
        u32 middle = low + (high - low) / 2u;
        if (points[middle] <= start)
            low = middle + 1u;
        else
            high = middle;
    }
    return low < count && points[low] < end ? points[low] : end;
}

static DolLLVMFunctionRange*
build_llvm_ranges(const LoadedCodeSection* sections, u32 section_count,
                  u32 chunk_instructions, const u32* partition_points,
                  u32 point_count, u32* result_count) {
    u32 capacity = point_count;
    for (u32 i = 0; i < section_count; i++)
        capacity += ((sections[i].size / 4u) + chunk_instructions - 1u) /
                    chunk_instructions;
    DolLLVMFunctionRange* ranges =
        (DolLLVMFunctionRange*)calloc(capacity, sizeof(*ranges));
    if (!ranges)
        return NULL;
    u32 count = 0;
    for (u32 i = 0; i < section_count; i++) {
        u32 start = sections[i].address;
        u32 section_end = start + (sections[i].size & ~3u);
        while (start < section_end) {
            u32 remaining = section_end - start;
            u32 span = chunk_instructions * 4u;
            u32 end = remaining < span ? section_end : start + span;
            end = next_llvm_partition_point(partition_points, point_count,
                                            start, end);
            ranges[count].start = start;
            ranges[count].end = end;
            count++;
            start = end;
        }
    }
    *result_count = count;
    return ranges;
}

static const DolLLVMFunctionRange*
llvm_range_for(const DolLLVMFunctionRange* ranges, u32 count, u32 address) {
    for (u32 i = 0; i < count; i++)
        if (address >= ranges[i].start && address < ranges[i].end)
            return &ranges[i];
    return NULL;
}

static int append_llvm_abi_edge(DolLLVMCallEdge** edges, u32* count,
                                u32* capacity, u32 caller, u32 callee,
                                const u64* live_after,
                                const u64* defined_before) {
    if (*count == *capacity) {
        u32 next = *capacity ? *capacity * 2u : 256u;
        DolLLVMCallEdge* resized =
            (DolLLVMCallEdge*)realloc(*edges, (size_t)next * sizeof(**edges));
        if (!resized)
            return 0;
        *edges = resized;
        *capacity = next;
    }
    (*edges)[*count].caller_start = caller;
    (*edges)[*count].callee_address = callee;
    memcpy((*edges)[*count].live_after, live_after,
           sizeof((*edges)[*count].live_after));
    memcpy((*edges)[*count].defined_before, defined_before,
           sizeof((*edges)[*count].defined_before));
    (*count)++;
    return 1;
}

/* DOLRECOMP_CLOSED_ROOTS=<hex address>[,...]: the closed set is every function
   that starts at a listed address or is reached from one by direct calls and
   branches to a function's first instruction. "all" selects every function. */
static void mark_llvm_closed_set(DolLLVMFunctionRange* ranges, u32 range_count,
                                 const u32* pairs, u32 pair_count) {
    const char* text = getenv("DOLRECOMP_CLOSED_ROOTS");
    if (!text || !*text)
        return;
    u32 marked = 0;
    if (strcmp(text, "all") == 0) {
        for (u32 i = 0; i < range_count; i++) {
            ranges[i].abi_flags |= DOLLLVM_FUNCTION_ABI_CLOSED;
            marked++;
        }
    } else {
        while (*text) {
            char* end = NULL;
            u32 address = (u32)strtoul(text, &end, 16);
            if (end == text)
                break;
            for (u32 i = 0; i < range_count; i++)
                if (ranges[i].start == address)
                    ranges[i].abi_flags |= DOLLLVM_FUNCTION_ABI_CLOSED;
            text = *end == ',' ? end + 1 : end;
        }
        int changed = 1;
        while (changed) {
            changed = 0;
            for (u32 e = 0; e < pair_count; e++) {
                u32 caller = pairs[e * 2u], callee = pairs[e * 2u + 1u];
                if ((ranges[caller].abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED) &&
                    !(ranges[callee].abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED)) {
                    ranges[callee].abi_flags |= DOLLLVM_FUNCTION_ABI_CLOSED;
                    if (getenv("DOLRECOMP_CLOSED_DEBUG"))
                        fprintf(stderr, "  closed edge %08X -> %08X\n",
                                ranges[caller].start, ranges[callee].start);
                    changed = 1;
                }
            }
        }
        for (u32 i = 0; i < range_count; i++)
            marked += (ranges[i].abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED) != 0;
    }
    /* DOLRECOMP_CLOSED_EXCLUDE=<hex address>[,...]: functions that never get a
       closed body, whatever reaches them (a loop the host skips as idle must
       keep ending dispatches at its head, which only an ordinary body does).
       Calls to them use the ordinary protocol. */
    const char* excluded = getenv("DOLRECOMP_CLOSED_EXCLUDE");
    while (excluded && *excluded) {
        char* end = NULL;
        u32 address = (u32)strtoul(excluded, &end, 16);
        if (end == excluded)
            break;
        for (u32 i = 0; i < range_count; i++)
            if (address >= ranges[i].start && address < ranges[i].end)
                ranges[i].abi_flags &= ~(u32)DOLLLVM_FUNCTION_ABI_CLOSED;
        excluded = *end == ',' ? end + 1 : end;
    }
    /* What a closed body may write before it returns normally: its own
       writes and those of the closed bodies it calls or branches to. A
       transfer to a function outside the set uses the ordinary protocol and
       makes the caller opaque. */
    u32 opaque = 0;
    for (u32 i = 0; i < range_count; i++)
        memcpy(ranges[i].closed_may_def, ranges[i].may_def_state,
               sizeof(ranges[i].closed_may_def));
    int grown = 1;
    while (grown) {
        grown = 0;
        for (u32 e = 0; e < pair_count; e++) {
            DolLLVMFunctionRange* caller = &ranges[pairs[e * 2u]];
            const DolLLVMFunctionRange* callee = &ranges[pairs[e * 2u + 1u]];
            if (!(caller->abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED))
                continue;
            if (!caller->closed_opaque &&
                (callee->closed_opaque ||
                 !(callee->abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED))) {
                caller->closed_opaque = 1;
                grown = 1;
            }
            for (u32 w = 0; w < DOLIR_STATE_MASK_WORDS; w++) {
                u64 merged = caller->closed_may_def[w] | callee->closed_may_def[w];
                if (merged != caller->closed_may_def[w]) {
                    caller->closed_may_def[w] = merged;
                    grown = 1;
                }
            }
        }
    }
    for (u32 i = 0; i < range_count; i++) {
        if (!(ranges[i].abi_flags & DOLLLVM_FUNCTION_ABI_CLOSED))
            continue;
        opaque += ranges[i].closed_opaque != 0;
        if (getenv("DOLRECOMP_CLOSED_DEBUG"))
            fprintf(stderr, "  closed member %08X%s\n", ranges[i].start,
                    ranges[i].closed_opaque ? " opaque" : "");
    }
    fprintf(stderr, "  closed set: %u functions, %u of them opaque\n", marked,
            opaque);
}

static int prepare_llvm_function_abis(const LoadedCodeSection* sections,
                                      u32 section_count,
                                      DolLLVMFunctionRange* ranges,
                                      u32 range_count, DolLLVMRuntime runtime) {
    DolLLVMCallEdge* edges = NULL;
    u32 edge_count = 0;
    u32 edge_capacity = 0;
    /* Control transfers to another range's first instruction, as pairs of
       range indexes, for the closed set. */
    u32* closed_pairs = NULL;
    u32 closed_pair_count = 0;
    u32 closed_pair_capacity = 0;
    const int want_closed = getenv("DOLRECOMP_CLOSED_ROOTS") != NULL;
    for (u32 range_index = 0; range_index < range_count; range_index++) {
        const LoadedCodeSection* section = NULL;
    for (u32 s = 0; s < section_count; s++) {
            u32 section_end = sections[s].address + sections[s].size;
            if (ranges[range_index].start >= sections[s].address &&
                ranges[range_index].end <= section_end) {
                section = &sections[s];
                break;
            }
        }
        if (!section || !section->data)
            goto fail;
        u32 start = (ranges[range_index].start - section->address) / 4u;
        u32 count = (ranges[range_index].end - ranges[range_index].start) / 4u;
            PPCInst* instructions =
                (PPCInst*)malloc((size_t)count * sizeof(*instructions));
            if (!instructions)
                goto fail;
            for (u32 i = 0; i < count; i++) {
                u32 raw = read_be32(section->data + (start + i) * 4u);
            instructions[i] =
                ppc_decode(raw, section->address + (start + i) * 4u);
                if (instructions[i].op == PPC_OP_UNKNOWN &&
                    embedded_data_word(section->embedded_data_mode, raw))
                    instructions[i].embedded_data = true;
            }
            DolIRModule module;
            dolir_module_init(&module);
            int ok = dolir_build_chunk(&module, instructions, count,
                                       section->address + start * 4u);
            free(instructions);
        if (!ok || !dolllvm_analyze_function_abi(&module.functions[0],
                                              &ranges[range_index])) {
                dolir_module_free(&module);
                goto fail;
            }
            const DolIRFunction* function = &module.functions[0];
            ranges[range_index].closed_opaque = 0;
            for (u32 block = 0; block < function->block_count; block++) {
            const DolIRTerminator* term = &function->blocks[block].terminator;
            /* For the closed set: an indirect call, or a runtime helper that
               may write registers the IR does not list, makes the function
               opaque (see DolLLVMFunctionRange.closed_opaque). */
            if (term->kind == DOLIR_TERM_INDIRECT && term->linked)
                ranges[range_index].closed_opaque = 1;
            for (u32 n = 0; n < function->blocks[block].instruction_count; n++) {
                const DolIRInstruction* inst =
                    &function->blocks[block].instructions[n];
                if (inst->op == DOLIR_OP_HELPER_CALL &&
                    inst->aux != DOLIR_HELPER_FP_AVAILABLE &&
                    inst->aux != DOLIR_HELPER_FPSCR_UPDATED &&
                    inst->aux != DOLIR_HELPER_FPSCR_BIT &&
                    inst->aux != DOLIR_HELPER_PSQ_LOAD &&
                    inst->aux != DOLIR_HELPER_PSQ_STORE &&
                    inst->aux != DOLIR_HELPER_MEMORY_FENCE &&
                    inst->aux != DOLIR_HELPER_EXACT_FLOAT &&
                    inst->aux != DOLIR_HELPER_EXACT_PAIRED)
                    ranges[range_index].closed_opaque = 1;
            }
            u64 live_after[DOLIR_STATE_MASK_WORDS] = {0};
            u64 defined_before[DOLIR_STATE_MASK_WORDS] = {0};
            if (!dolllvm_analyze_callsite_state(
                    function, block, ranges[range_index].may_def_state,
                    live_after, defined_before)) {
                dolir_module_free(&module);
                goto fail;
            }
            u32 targets = term->kind == DOLIR_TERM_COND_BRANCH ? 2u
                          : term->kind == DOLIR_TERM_FALLTHROUGH ||
                                  term->kind == DOLIR_TERM_BRANCH
                              ? 1u
                          : term->kind == DOLIR_TERM_INDIRECT ? 2u
                                                              : 0u;
                for (u32 slot = 0; slot < targets; slot++) {
                    const DolLLVMFunctionRange* target = llvm_range_for(
                        ranges, range_count, term->target_addresses[slot]);
                    if (target && target->start != ranges[range_index].start &&
                    !append_llvm_abi_edge(&edges, &edge_count, &edge_capacity,
                            ranges[range_index].start,
                                          term->target_addresses[slot],
                                          live_after, defined_before)) {
                        dolir_module_free(&module);
                        goto fail;
                    }
                    /* Not the address after an unconditional blr/bctr, which
                       the terminator lists but control never reaches. */
                    u32 raw = function->blocks[block].raw;
                    int never = term->kind == DOLIR_TERM_INDIRECT &&
                                slot == 1u && ((raw >> 21) & 0x14u) == 0x14u;
                    /* Into the middle of another function: the ordinary
                       protocol. */
                    if (target && !never &&
                        target->start != ranges[range_index].start &&
                        target->start != term->target_addresses[slot])
                        ranges[range_index].closed_opaque = 1;
                    if (want_closed && target && !never &&
                        target->start != ranges[range_index].start &&
                        target->start == term->target_addresses[slot]) {
                        if (closed_pair_count == closed_pair_capacity) {
                            u32 next = closed_pair_capacity
                                           ? closed_pair_capacity * 2u
                                           : 1024u;
                            u32* resized = (u32*)realloc(
                                closed_pairs, (size_t)next * 2u * sizeof(u32));
                            if (!resized) {
                                dolir_module_free(&module);
                                goto fail;
                            }
                            closed_pairs = resized;
                            closed_pair_capacity = next;
                        }
                        closed_pairs[closed_pair_count * 2u] = range_index;
                        closed_pairs[closed_pair_count * 2u + 1u] =
                            (u32)(target - ranges);
                        closed_pair_count++;
                    }
                }
            }
            dolir_module_free(&module);
        }
    if (runtime == DOLLLVM_RUNTIME_MODERNGEKKO)
        dolllvm_enable_native_services(ranges, range_count);
    if (!dolllvm_propagate_function_abis(ranges, range_count, edges,
                                         edge_count))
        goto fail;
    mark_llvm_closed_set(ranges, range_count, closed_pairs, closed_pair_count);
    free(closed_pairs);
    free(edges);
    return 1;
fail:
    free(closed_pairs);
    free(edges);
    return 0;
}

static u32* collect_llvm_entry_points(const LoadedCodeSection* sections,
                                      u32 section_count, u32 program_entry,
                                      const DolRecompSymbolMap* symbols,
                                      u32* result_count) {
    size_t capacity = 1u + (symbols ? symbols->count : 0u);
    for (u32 i = 0; i < section_count; i++)
        capacity += sections[i].size / 4u;
    u32* points = (u32*)malloc(capacity * sizeof(*points));
    if (!points)
        return NULL;
    u32 count = 0;
    if (llvm_code_address(sections, section_count, program_entry))
        points[count++] = program_entry;
    if (symbols) {
        for (u32 i = 0; i < symbols->count; i++) {
            u32 address = symbols->symbols[i].address;
            if (llvm_code_address(sections, section_count, address))
                points[count++] = address;
        }
    }
    for (u32 section_index = 0; section_index < section_count;
         section_index++) {
        const LoadedCodeSection* section = &sections[section_index];
        for (u32 offset = 0; section->data && offset + 4u <= section->size;
             offset += 4u) {
            u32 address = section->address + offset;
            PPCInst inst =
                ppc_decode(read_be32(section->data + offset), address);
            if ((inst.op == PPC_OP_B || inst.op == PPC_OP_BC) &&
                llvm_code_address(sections, section_count, inst.branch_target))
                points[count++] = inst.branch_target;
        }
    }
    qsort(points, count, sizeof(*points), compare_u32);
    u32 unique = 0;
    for (u32 i = 0; i < count; i++) {
        if (!unique || points[i] != points[unique - 1u])
            points[unique++] = points[i];
    }
    *result_count = unique;
    return points;
}

static int llvm_cache_dir(char* path, size_t size) {
    const char* configured = getenv("DOLRECOMP_LLVM_CACHE");
    if (configured && (!configured[0] || strcmp(configured, "off") == 0))
        return 0;
    if (configured) {
        if (snprintf(path, size, "%s", configured) >= (int)size)
            return 0;
    } else {
#ifdef _WIN32
        const char* root = getenv("LOCALAPPDATA");
        if (!root ||
            snprintf(path, size, "%s\\DolRecomp\\llvm", root) >= (int)size)
            return 0;
#else
        const char* root = getenv("XDG_CACHE_HOME");
        if (root) {
            if (snprintf(path, size, "%s/dolrecomp/llvm", root) >= (int)size)
                return 0;
        } else {
            root = getenv("HOME");
            if (!root || snprintf(path, size, "%s/.cache/dolrecomp/llvm",
                                  root) >= (int)size)
                return 0;
        }
#endif
    }
    return make_dir_tree(path);
}

static int reuse_llvm_object(const LLVMChunkJob* job) {
    if (getenv("DOLRECOMP_LLVM_RESUME") && valid_object_file(job, job->path) &&
        file_exists(job->thinlto_path) && valid_llvm_job_stamp(job))
        return 1;
    if (!job->cache_path[0] || !valid_object_file(job, job->cache_path) ||
        !file_exists(job->cache_bitcode_path) ||
        !copy_file(job->cache_path, job->path) ||
        !copy_file(job->cache_bitcode_path, job->thinlto_path))
        return 0;
    write_llvm_job_stamp(job);
    return 1;
}

static void cache_llvm_object(const LLVMChunkJob* job) {
    if (!job->cache_path[0] || (valid_object_file(job, job->cache_path) &&
         file_exists(job->cache_bitcode_path)))
        return;
    char temp[1440];
#ifdef _WIN32
    int process_id = _getpid();
#else
    int process_id = (int)getpid();
#endif
    if (snprintf(temp, sizeof(temp), "%s.tmp.%d", job->cache_path,
                 process_id) >= (int)sizeof(temp))
        return;
    remove(temp);
    if (!copy_file(job->path, temp))
        return;
    if (rename(temp, job->cache_path) != 0)
        remove(temp);
    if (snprintf(temp, sizeof(temp), "%s.tmp.%d", job->cache_bitcode_path,
                 process_id) >= (int)sizeof(temp))
        return;
    remove(temp);
    if (!copy_file(job->thinlto_path, temp))
        return;
    if (rename(temp, job->cache_bitcode_path) != 0)
        remove(temp);
}

/* DOLRECOMP_LLVM_ONLY="8006E1F0,801BA73C,...": emit only the chunks that
   contain one of the listed guest addresses and skip every other one, for
   inspecting generated code in seconds. The output is not a linkable module. */
static int llvm_job_selected(const LLVMChunkJob* job) {
    const char* only = getenv("DOLRECOMP_LLVM_ONLY");
    if (!only || !only[0])
        return 1;
    const u32 start = job->function_address;
    const u32 end = start + job->count * 4u;
    for (const char* p = only; *p;) {
        char* next = NULL;
        unsigned long address = strtoul(p, &next, 16);
        if (next == p)
            break;
        if ((u32)address >= start && (u32)address < end)
            return 1;
        p = next;
        while (*p == ',' || *p == ' ')
            p++;
    }
    return 0;
}

static int emit_llvm_chunk_job(const void* data, void* user) {
    const LLVMChunkJob* job = (const LLVMChunkJob*)data;
    (void)user;
    if (!llvm_job_selected(job))
        return 1;
    if (reuse_llvm_object(job))
        return 1;
    char temp_path[1440];
#ifdef _WIN32
    int process_id = _getpid();
#else
    int process_id = (int)getpid();
#endif
    if (snprintf(temp_path, sizeof(temp_path), "%s.tmp.%d", job->path,
                 process_id) >= (int)sizeof(temp_path))
        return 0;
    remove(temp_path);
    DolIRModule module;
    dolir_module_init(&module);
    if (!dolir_build_chunk(&module, job->insts, job->count,
                           job->function_address) ||
        !dolir_verify(&module, stderr)) {
        dolir_module_free(&module);
        return 0;
    }
    for (u32 i = 0; i < job->import_count; i++) {
        PPCInst* imported = decode_guest_code(job->sections, job->section_count,
                                              job->imports[i].start,
                                              job->imports[i].count);
        int built = imported &&
                    dolir_build_chunk(&module, imported, job->imports[i].count,
                                      job->imports[i].start);
        free(imported);
        if (!built || !dolir_verify(&module, stderr)) {
            dolir_module_free(&module);
            return 0;
        }
    }
    DolLLVMOptions options = {0};
    options.own_function_count = job->import_count ? 1u : 0u;
    options.target_triple = getenv("DOLRECOMP_LLVM_TARGET");
    options.target_profile = job->target_profile;
    options.semantics = job->semantics;
    options.instrumentation = job->instrumentation;
    options.native_abi_policy = job->native_abi_policy;
    options.runtime = job->runtime;
    options.symbol_suffix = job->symbol_suffix;
    options.profile_generate_path = job->profile_generate_path;
    options.profile_use_path = job->profile_use_path;
    options.partition_seed = job->partition_seed;
    options.state_in_memory = job->state_in_memory;
    options.emit_thinlto = 1;
    options.thinlto_path = job->thinlto_path;
    options.fixed_memory_layout = 1;
    options.ram_size = job->ram_size;
    options.mem2_size = job->mem2_size;
    options.optimization_level = 3;
    options.verify = 1;
    options.function_ranges = job->ranges;
    options.function_range_count = job->range_count;
    options.entry_points = job->entry_points;
    options.entry_point_count = job->entry_point_count;
    const char* write_journal = getenv("DOLRECOMP_LLVM_WRITE_JOURNAL");
    if (write_journal && !strcmp(write_journal, "1"))
        options.instrumentation = DOLLLVM_INSTRUMENTATION_LOCKSTEP;
    char ir_path[1440];
    const char* dump_ir = getenv("DOLRECOMP_LLVM_DUMP_IR");
    if (dump_ir && (!strcmp(dump_ir, "1") || strstr(job->name, dump_ir))) {
        if (snprintf(ir_path, sizeof(ir_path), "%s.ll", job->path) >=
            (int)sizeof(ir_path)) {
            dolir_module_free(&module);
            return 0;
        }
        options.emit_ir = 1;
        options.ir_path = ir_path;
    }
    int ok = dolllvm_emit_object(&module, temp_path, &options, stderr);
    dolir_module_free(&module);
    if (ok) {
        remove(job->path);
        if (rename(temp_path, job->path) != 0) {
            fprintf(stderr, "error: cannot publish LLVM object %s: %s\n",
                    job->path, strerror(errno));
            ok = 0;
        }
    }
    if (ok) {
        write_llvm_job_stamp(job);
        cache_llvm_object(job);
    }
    if (!ok)
        remove(temp_path);
    return ok;
}

static void report_llvm_progress(const LLVMChunkJob* jobs,
                                 const unsigned char* states, u32 count,
                                 u32* next_report) {
    while (*next_report < count && states[*next_report]) {
        const LLVMChunkJob* job = &jobs[*next_report];
        printf("[%u/%u] %s LLVM object %s\n", job->index, job->total,
               states[*next_report] == 2 ? "Reusing cached" : "Emitting",
               job->name);
        (*next_report)++;
    }
    fflush(stdout);
}

static int run_llvm_chunk_jobs(const LLVMChunkJob* jobs, u32 count,
                               u32 requested_jobs) {
    u32 workers = effective_chunk_jobs(count, requested_jobs);
#ifdef _WIN32
    for (u32 i = 0; i < count; i++) {
        printf("[%u/%u] Emitting LLVM object %s\n", jobs[i].index,
               jobs[i].total, jobs[i].name);
        fflush(stdout);
    }
    return run_parallel_jobs(jobs, sizeof(*jobs), count, workers,
                             emit_llvm_chunk_job, NULL);
#else
    typedef struct {
        pid_t pid;
        u32 batch_start;
        u32 batch_count;
    } LLVMWorker;
    LLVMWorker* active_workers =
        (LLVMWorker*)calloc(workers, sizeof(*active_workers));
    u32* pending = (u32*)calloc(count, sizeof(*pending));
    u32* retry = (u32*)calloc(count, sizeof(*retry));
    unsigned char* states = (unsigned char*)calloc(count, sizeof(*states));
    if (!active_workers || !pending || !retry || !states) {
        free(active_workers);
        free(pending);
        free(retry);
        free(states);
        return 0;
    }

    u32 completed = 0;
    u32 pending_count = 0;
    u32 next_report = 0;
    for (u32 i = 0; i < count; i++) {
        if (reuse_llvm_object(&jobs[i])) {
            completed++;
            states[i] = 2;
        } else {
            pending[pending_count++] = i;
        }
    }
    report_llvm_progress(jobs, states, count, &next_report);

    int failed = 0;
    u32 next = 0;
    u32 active = 0;
    u32 retry_count = 0;
    const u32 batch_size = llvm_worker_batch_size();
    while (next < pending_count || active != 0) {
        while (next < pending_count && active < workers) {
            const u32 batch_start = next;
            u32 batch_count = pending_count - next;
            if (batch_count > batch_size)
                batch_count = batch_size;
            next += batch_count;
            fflush(NULL);
            const pid_t pid = fork();
            if (pid == 0) {
                int ok = 1;
                for (u32 i = 0; i < batch_count; i++) {
                    const u32 job = pending[batch_start + i];
                    if (!emit_llvm_chunk_job(&jobs[job], NULL))
                        ok = 0;
                }
                _exit(ok ? 0 : 1);
            }
            if (pid < 0) {
                fprintf(stderr, "error: can't start LLVM worker process\n");
                for (u32 i = 0; i < batch_count; i++)
                    retry[retry_count++] = pending[batch_start + i];
                continue;
            }
            active_workers[active++] =
                (LLVMWorker){pid, batch_start, batch_count};
        }
        if (!active)
            break;

        int status = 0;
        pid_t finished;
        do {
            finished = waitpid(-1, &status, 0);
        } while (finished < 0 && errno == EINTR);
        if (finished < 0) {
            failed = 1;
            break;
        }

        u32 slot = 0;
        while (slot < active && active_workers[slot].pid != finished)
            slot++;
        if (slot == active) {
            failed = 1;
            continue;
        }
        const u32 batch_start = active_workers[slot].batch_start;
        const u32 batch_count = active_workers[slot].batch_count;
        active_workers[slot] = active_workers[--active];
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            for (u32 i = 0; i < batch_count; i++) {
                const u32 job = pending[batch_start + i];
                completed++;
                states[job] = 1;
            }
            report_llvm_progress(jobs, states, count, &next_report);
        } else {
            for (u32 i = 0; i < batch_count; i++)
                retry[retry_count++] = pending[batch_start + i];
        }
    }

    while (next < pending_count)
        retry[retry_count++] = pending[next++];

    for (u32 i = 0; i < retry_count; i++) {
        const u32 job = retry[i];
        fprintf(stderr, "retrying LLVM object %s after worker failure\n",
                jobs[job].name);
        fflush(NULL);
        const pid_t pid = fork();
        int status = 0;
        if (pid == 0)
            _exit(emit_llvm_chunk_job(&jobs[job], NULL) ? 0 : 1);
        if (pid < 0 || waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) ||
            WEXITSTATUS(status) != 0) {
            failed = 1;
            continue;
        }
        completed++;
        states[job] = 1;
        report_llvm_progress(jobs, states, count, &next_report);
    }
    free(active_workers);
    free(pending);
    free(retry);
    free(states);
    return !failed && completed == count;
#endif
}

static int emit_code_sections_llvm(const LoadedCodeSection* sections,
                                   u32 section_count, const char* output_path,
                                   DolRecompCPU cpu, u32 entry_point,
                                   u32 requested_jobs, int local_chunks_dir,
                                   const DolRecompSymbolMap* symbols,
                                   const CliOptions* options) {
    DolLLVMTargetProfile profiles[5];
    u32 profile_count = parse_llvm_target_set(options->llvm_targets, profiles);
    if (!profile_count) {
        fprintf(stderr, "error: invalid or duplicate LLVM target set\n");
        return 0;
    }
    const int modern_runtime =
        options->llvm_runtime == DOLLLVM_RUNTIME_MODERNGEKKO;
    if (modern_runtime && profile_count != 1u) {
        fprintf(
            stderr,
            "error: ModernGekko modules currently require one LLVM target\n");
        return 0;
    }
    if (modern_runtime && !options->game_id) {
        fprintf(stderr, "error: ModernGekko modules require a game ID\n");
        return 0;
    }
    char stem[1024];
    char header_path[1100];
    char symbol_header_path[1100];
    char fallback_path[1100];
    char chunks_dir[1100];
    char include_name[512];
    if (!make_output_stem(output_path, stem, sizeof(stem)) ||
        !split_include_name(stem, include_name, sizeof(include_name)))
        return 0;
    if (snprintf(header_path, sizeof(header_path), "%s.h", stem) >=
            (int)sizeof(header_path) ||
        snprintf(symbol_header_path, sizeof(symbol_header_path), "%s_symbols.h",
                 stem) >= (int)sizeof(symbol_header_path) ||
        snprintf(fallback_path, sizeof(fallback_path), "%s_fallbacks.csv",
                 stem) >= (int)sizeof(fallback_path)) {
        fprintf(stderr, "error: output path is too long\n");
        return 0;
    }
    if (local_chunks_dir) {
        char output_dir[1024];
        if (!path_dirname(output_path, output_dir, sizeof(output_dir)) ||
            !join_path(chunks_dir, sizeof(chunks_dir), output_dir, "chunks"))
            return 0;
    } else if (snprintf(chunks_dir, sizeof(chunks_dir), "%s_chunks", stem) >=
               (int)sizeof(chunks_dir)) {
        return 0;
    }
    if (!make_dir_tree(chunks_dir))
        return 0;

    FILE* manifest = fopen(output_path, "w");
    FILE* header = fopen(header_path, "w");
    FILE* fallback_report = fopen(fallback_path, "w");
    if (!manifest || !header || !fallback_report) {
        fprintf(stderr, "error: cannot create LLVM output files\n");
        if (manifest)
            fclose(manifest);
        if (header)
            fclose(header);
        if (fallback_report)
            fclose(fallback_report);
        return 0;
    }
    fprintf(fallback_report, "address,raw,opcode,reason,detail\n");
    fprintf(manifest, "#include \"%s\"\n", include_name);
    if (modern_runtime) {
        fprintf(header, "#ifndef RECOMP_GENERATED_H\n"
                        "#define RECOMP_GENERATED_H\n\n"
                        "#include <stdbool.h>\n"
                        "#include <stdatomic.h>\n"
                        "#include <stdint.h>\n"
                        "#include \"Core/PowerPC/Native/NativeModuleABI.h\"\n\n"
                        "#define DOLRECOMP_BACKEND_LLVM 1\n\n");
    } else {
    emit_header_for_cpu(header, cpu);
    fprintf(header, "#define DOLRECOMP_BACKEND_LLVM 1\n\n");
    }
    if (symbols) {
        u32 symbol_count = count_code_symbols(symbols, sections, section_count);
        if (!symbol_count) {
            fprintf(stderr, "error: symbol map has no executable entries\n");
            fclose(header);
            fclose(manifest);
            fclose(fallback_report);
            return 0;
        }
        FILE* symbol_header = fopen(symbol_header_path, "w");
        if (!symbol_header ||
            !emit_symbol_definitions(symbol_header, symbols, sections,
                                     section_count)) {
            fprintf(stderr, "error: failed to emit symbol map\n");
            if (symbol_header)
                fclose(symbol_header);
            fclose(header);
            fclose(manifest);
            fclose(fallback_report);
            return 0;
        }
        fclose(symbol_header);
    } else {
        remove(symbol_header_path);
    }
    fprintf(header, "\n// Function entry points\n");

    FunctionList funcs = {0};
    SMCAnalysis smc = {0};
    DolLLVMFunctionRange* ranges = NULL;
    u32* entry_points = NULL;
    u32* partition_points = NULL;
    u32 entry_point_count = 0;
    entry_points = collect_llvm_entry_points(
        sections, section_count, entry_point, symbols, &entry_point_count);
    if (!entry_points)
        goto fail;
    u32 partition_point_count = 0;
    partition_points = collect_llvm_partition_points(
        sections, section_count, entry_point, symbols, &partition_point_count);
    if (!partition_points)
        goto fail;
    u32 file_count = 0;
    u32 range_count = 0;
    const u32 chunk_instructions = options->partition_instructions
                                       ? options->partition_instructions
                                       : llvm_chunk_instructions();
    ranges = build_llvm_ranges(sections, section_count, chunk_instructions,
                               partition_points, partition_point_count,
                               &range_count);
    if (!ranges)
        goto fail;
    char cache_dir[1100] = "";
    if (!llvm_cache_dir(cache_dir, sizeof(cache_dir)))
        cache_dir[0] = '\0';
    if (!prepare_llvm_function_abis(sections, section_count, ranges,
                                    range_count,
                                    (DolLLVMRuntime)options->llvm_runtime))
        goto fail;
    dolllvm_apply_native_abi_policy(
        ranges, range_count, (DolLLVMNativeABIPolicy)options->llvm_native_abi);
    if (getenv("DOLRECOMP_LLVM_ABI_STATS")) {
        for (u32 variant = 0; variant < profile_count; variant++) {
            DolLLVMOptions stats_options = {0};
            char triple[128];
            stats_options.target_profile = profiles[variant];
            if (dolllvm_effective_triple(&stats_options, triple,
                                         sizeof(triple)))
                dolllvm_report_abi_stats(ranges, range_count, triple, stdout);
        }
    }
    u32 emitted_range_count = 0;
    for (u32 s = 0; s < section_count; s++) {
        const LoadedCodeSection* section = &sections[s];
        if (!section->data || !section->size)
            continue;
        u32 num_insts = section->size / 4u;
        PPCInst* insts = (PPCInst*)malloc((size_t)num_insts * sizeof(*insts));
        if (!insts) {
            fprintf(stderr, "error: out of memory\n");
            goto fail;
        }
        u32 embedded = 0;
        u32 unknown = 0;
        for (u32 i = 0; i < num_insts; i++) {
            u32 raw = read_be32(section->data + i * 4u);
            insts[i] = ppc_decode(raw, section->address + i * 4u);
            if (insts[i].op == PPC_OP_UNKNOWN &&
                embedded_data_word(section->embedded_data_mode, raw))
                insts[i].embedded_data = true;
            embedded += insts[i].embedded_data;
            unknown += insts[i].op == PPC_OP_UNKNOWN && !insts[i].embedded_data;
        }
        printf("decoding %s[%u]: %u instructions at 0x%08X\n", section->label,
               section->index, num_insts, section->address);
        printf("  %u known, %u embedded data, %u unknown\n",
               num_insts - embedded - unknown, embedded, unknown);
        if (section->embedded_data_mode == EMBEDDED_DATA_DOL) {
            analyze_smc_section(sections, section_count, insts, num_insts,
                                &smc);
            if (smc.allocation_failed) {
                free(insts);
                goto fail;
            }
        }

        u32 first_range = emitted_range_count;
        u32 section_end = section->address + section->size;
        while (emitted_range_count < range_count &&
               ranges[emitted_range_count].start >= section->address &&
               ranges[emitted_range_count].end <= section_end)
            emitted_range_count++;
        u32 chunk_total = emitted_range_count - first_range;
        u32 job_total = chunk_total * profile_count;
        LLVMChunkJob* chunk_jobs =
            (LLVMChunkJob*)calloc(job_total, sizeof(*chunk_jobs));
        if (!chunk_jobs) {
            free(insts);
            goto fail;
        }
        for (u32 chunk_index = 0; chunk_index < chunk_total; chunk_index++) {
            u32 output_index = first_range + chunk_index;
            const DolLLVMFunctionRange* range = &ranges[output_index];
            u32 function_address = range->start;
            u32 start = (function_address - section->address) / 4u;
            u32 chunk_count = (range->end - range->start) / 4u;
            LLVMChunkJob* job = &chunk_jobs[chunk_index * profile_count];
            for (u32 variant = 0; variant < profile_count; variant++) {
                LLVMChunkJob* target_job = &job[variant];
                const char* target_suffix =
                    dolllvm_target_profile_suffix(profiles[variant]);
                if (variant)
                    snprintf(target_job->symbol_suffix,
                             sizeof(target_job->symbol_suffix), "__%s",
                             target_suffix);
                int name_length =
                    variant
                    ? snprintf(target_job->name, sizeof(target_job->name),
                                   "chunk_%04u_%s%u_%08X_%s.o", output_index,
                                   section->label, section->index,
                                   function_address, target_suffix)
                    : snprintf(target_job->name, sizeof(target_job->name),
                               "chunk_%04u_%s%u_%08X.o", output_index,
                               section->label, section->index,
                               function_address);
                if (name_length >= (int)sizeof(target_job->name) ||
                    !join_path(target_job->path, sizeof(target_job->path),
                               chunks_dir, target_job->name)) {
                    free(chunk_jobs);
                    free(insts);
                    goto fail;
                }
                if (snprintf(target_job->thinlto_path,
                             sizeof(target_job->thinlto_path), "%s.bc",
                             target_job->path) >=
                    (int)sizeof(target_job->thinlto_path)) {
                    free(chunk_jobs);
                    free(insts);
                    goto fail;
                }
                target_job->insts = insts + start;
                target_job->count = chunk_count;
                target_job->function_address = function_address;
                target_job->index = output_index * profile_count + variant + 1u;
                target_job->total = range_count * profile_count;
                target_job->ranges = ranges;
                target_job->range_count = range_count;
                target_job->entry_points = entry_points;
                target_job->entry_point_count = entry_point_count;
                target_job->target_profile = profiles[variant];
                target_job->semantics = options->fast_semantics
                                            ? DOLLLVM_SEMANTICS_FAST
                                            : DOLLLVM_SEMANTICS_EXACT;
                target_job->instrumentation =
                    options->lockstep_instrumentation
                                                  ? DOLLLVM_INSTRUMENTATION_LOCKSTEP
                                                  : DOLLLVM_INSTRUMENTATION_NONE;
                target_job->native_abi_policy =
                    (DolLLVMNativeABIPolicy)options->llvm_native_abi;
                target_job->runtime = (DolLLVMRuntime)options->llvm_runtime;
                target_job->profile_generate_path =
                    options->profile_generate_path;
                target_job->profile_use_path = options->profile_use_path;
                target_job->partition_seed = options->partition_seed;
                target_job->state_in_memory = options->state_in_memory;
                target_job->sections = sections;
                target_job->section_count = section_count;
                collect_llvm_imports(target_job, target_job->insts,
                                     target_job->count,
                                     target_job->function_address,
                                     llvm_inline_callee_limit());
                target_job->ram_size = GC_MAIN_RAM_SIZE;
                target_job->mem2_size =
                    cpu == DOLRECOMP_CPU_GEKKO ? 0u : WII_MEM2_SIZE;
                target_job->hash = llvm_job_hash(target_job);
                if (cache_dir[0]) {
                    char cache_name[64];
                    snprintf(cache_name, sizeof(cache_name), "%016llx.o",
                             (unsigned long long)target_job->hash);
                    if (!join_path(target_job->cache_path,
                                   sizeof(target_job->cache_path), cache_dir,
                                   cache_name))
                        target_job->cache_path[0] = '\0';
                    if (target_job->cache_path[0] &&
                        snprintf(target_job->cache_bitcode_path,
                                 sizeof(target_job->cache_bitcode_path),
                                 "%s.bc", target_job->cache_path) >=
                            (int)sizeof(target_job->cache_bitcode_path))
                        target_job->cache_path[0] = '\0';
                }
            }
            DolIRModule audit;
            dolir_module_init(&audit);
            if (!dolir_build_chunk(&audit, job->insts, job->count,
                                   job->function_address)) {
                dolir_module_free(&audit);
                free(chunk_jobs);
                free(insts);
                goto fail;
            }
            const DolIRFunction* audit_function = &audit.functions[0];
            for (u32 i = 0; i < audit_function->block_count; i++) {
                if (audit_function->blocks[i].terminator.kind !=
                    DOLIR_TERM_FALLBACK)
                    continue;
                const PPCInst* fallback = &job->insts[i];
                const char* reason =
                    fallback->embedded_data          ? "embedded-data"
                    : fallback->op == PPC_OP_UNKNOWN ? "unknown"
                    : (fallback->op == PPC_OP_SC || fallback->op == PPC_OP_RFI)
                        ? "exception-boundary"
                        : "unsupported";
                char detail[32] = "";
                switch (fallback->op) {
                case PPC_OP_MFSPR:
                case PPC_OP_MTSPR:
                case PPC_OP_MFTB:
                    snprintf(detail, sizeof(detail), "spr=%u", fallback->spr);
                    break;
                case PPC_OP_MFSR:
                case PPC_OP_MTSR:
                    snprintf(detail, sizeof(detail), "sr=%u", fallback->sr);
                    break;
                case PPC_OP_TW:
                case PPC_OP_TWI:
                    snprintf(detail, sizeof(detail), "to=%u", fallback->to);
                    break;
                default:
                    break;
                }
                fprintf(fallback_report, "%08X,%08X,%s,%s,%s\n",
                        fallback->address, fallback->raw,
                        ppc_op_name(fallback->op), reason, detail);
            }
            dolir_module_free(&audit);
            if (modern_runtime) {
                if (range->abi_flags & DOLLLVM_FUNCTION_ABI_NATIVE) {
                    fprintf(header,
                            "MGNativeExit func_%08X(const MGNativeRuntime*, "
                            "const MGNativeState*, uint32_t, uint32_t);\n",
                            function_address);
                    u64 native_hash = hash_bytes(UINT64_C(0xCBF29CE484222325),
                                                 section->data + start * 4u,
                                                 (size_t)chunk_count * 4u);
                    if (!function_list_add_hashed(&funcs, function_address,
                                                  range->end, native_hash)) {
                        free(chunk_jobs);
                        free(insts);
                        goto fail;
                    }
                }
            } else {
                emit_chunk_prototype(header, function_address);
                for (u32 variant = 1; variant < profile_count; variant++)
                    fprintf(header, "void func_%08X%s(CPUState* ctx);\n",
                            function_address, job[variant].symbol_suffix);
                if (!function_list_add(&funcs, function_address, range->end)) {
                    free(chunk_jobs);
                    free(insts);
                    goto fail;
                }
            }
            for (u32 variant = 0; variant < profile_count; variant++) {
                fprintf(manifest, "// object: chunks/%s\n", job[variant].name);
                fprintf(manifest, "// object[%s]: chunks/%s\n",
                        dolllvm_target_profile_name(profiles[variant]),
                        job[variant].name);
            }
            fprintf(manifest, "// ThinLTO summaries: chunks/*.o.bc\n");
            file_count += profile_count;
        }
        u32 active_jobs = effective_chunk_jobs(job_total, requested_jobs);
        printf("  writing %u LLVM objects with %u job%s\n", job_total,
               active_jobs, active_jobs == 1 ? "" : "s");
        if (!run_llvm_chunk_jobs(chunk_jobs, job_total, requested_jobs)) {
            free(chunk_jobs);
            free(insts);
            goto fail;
        }
        free(chunk_jobs);
        free(insts);
    }
    if (emitted_range_count != range_count)
        goto fail;

    {
        char report[1100];
        if (snprintf(report, sizeof(report), "%s_smc.txt", stem) >=
                (int)sizeof(report) ||
            !write_smc_report(&smc, report))
            goto fail;
        if (smc.possible)
            printf("warning: executable memory writes detected; report: %s\n",
                   report);
    }
    if (modern_runtime) {
        emit_native_module(header, &funcs, options->game_id);
    } else {
    emit_dispatch_helpers(header, &funcs, entry_point);
    emit_llvm_variant_table(header, &funcs, profiles, profile_count,
                            options->fast_semantics);
    }
    emit_footer(header);
    fprintf(manifest, "\n// %u native objects\n", file_count);
    fclose(header);
    fclose(manifest);
    fclose(fallback_report);
    smc_analysis_free(&smc);
    function_list_free(&funcs);
    free(ranges);
    free(entry_points);
    free(partition_points);
    printf("done!\n  header: %s\n  objects: %s (%u files)\n", header_path,
           chunks_dir, file_count);
    return 1;

fail:
    smc_analysis_free(&smc);
    function_list_free(&funcs);
    free(ranges);
    free(entry_points);
    free(partition_points);
    fclose(header);
    fclose(manifest);
    fclose(fallback_report);
    return 0;
}
#endif

int emit_code_sections_split(const LoadedCodeSection* sections,
                             u32 section_count, const char* output_path,
                                    DolRecompCPU cpu, u32 entry_point, u32 jobs,
                                    int local_chunks_dir,
                                    const DolRecompSymbolMap* symbols,
                                    const CliOptions* options) {
#ifdef DOLRECOMP_ENABLE_LLVM
    if (options->backend == DOLRECOMP_BACKEND_LLVM)
        return emit_code_sections_llvm(sections, section_count, output_path,
                                       cpu, entry_point, jobs, local_chunks_dir,
                                       symbols, options);
#else
    if (options->backend == DOLRECOMP_BACKEND_LLVM) {
        fprintf(stderr, "error: LLVM backend is unavailable in this build\n");
        return 0;
    }
#endif
    char stem[1024];
    char header_path[1100];
    char symbol_header_path[1100];
    char chunks_dir[1100];
    char chunks_label[512];
    char include_name[512];

    if (!make_output_stem(output_path, stem, sizeof(stem)))
        return 0;
    if (!split_include_name(stem, include_name, sizeof(include_name))) {
        fprintf(stderr, "error: output include name is too long\n");
        return 0;
    }

    if (snprintf(header_path, sizeof(header_path), "%s.h", stem) >=
        (int)sizeof(header_path)) {
        fprintf(stderr, "error: output path is too long\n");
        return 0;
    }
    if (snprintf(symbol_header_path, sizeof(symbol_header_path), "%s_symbols.h",
                 stem) >= (int)sizeof(symbol_header_path)) {
        fprintf(stderr, "error: output path is too long\n");
        return 0;
    }

    if (local_chunks_dir) {
        char output_dir[1024];
        if (!path_dirname(output_path, output_dir, sizeof(output_dir)) ||
            !join_path(chunks_dir, sizeof(chunks_dir), output_dir, "chunks")) {
            fprintf(stderr, "error: output path is too long\n");
            return 0;
        }
        snprintf(chunks_label, sizeof(chunks_label), "chunks");
    } else {
        if (snprintf(chunks_dir, sizeof(chunks_dir), "%s_chunks", stem) >=
            (int)sizeof(chunks_dir)) {
            fprintf(stderr, "error: output path is too long\n");
            return 0;
        }
        snprintf(chunks_label, sizeof(chunks_label), "%s",
                 path_basename(chunks_dir));
    }

    if (!make_dir_tree(chunks_dir))
        return 0;

    FILE* manifest = fopen(output_path, "w");
    if (!manifest) {
        fprintf(stderr, "error: can't open output '%s'\n", output_path);
        return 0;
    }

    FILE* header = fopen(header_path, "w");
    if (!header) {
        fprintf(stderr, "error: can't open output '%s'\n", header_path);
        fclose(manifest);
        return 0;
    }

    fprintf(manifest, "// DolRecomp split output\n");
    fprintf(manifest, "#include \"%s\"\n\n", include_name);
    fprintf(manifest, "// Build these C files too:\n");

    emit_header_for_cpu(header, cpu);
    if (symbols) {
        u32 symbol_count = count_code_symbols(symbols, sections, section_count);
        if (symbol_count == 0) {
            fprintf(
                stderr,
                "error: symbol map has no entries in executable sections\n");
            fclose(header);
            fclose(manifest);
            return 0;
        }
        FILE* symbol_header = fopen(symbol_header_path, "w");
        if (!symbol_header) {
            fprintf(stderr, "error: can't open output '%s'\n",
                    symbol_header_path);
            fclose(header);
            fclose(manifest);
            return 0;
        }
        if (!emit_symbol_definitions(symbol_header, symbols, sections,
                                     section_count)) {
            fprintf(stderr, "error: failed to emit symbol map\n");
            fclose(symbol_header);
            fclose(header);
            fclose(manifest);
            return 0;
        }
        fclose(symbol_header);
        printf("loaded %u executable symbols\n", symbol_count);
    } else {
        remove(symbol_header_path);
    }
    fprintf(header, "\n// Function entry points\n");

    u32 file_count = 0;
    const u32 chunk_instructions = c_chunk_instructions();
    FunctionList funcs = {0};
    SMCAnalysis smc = {0};

    for (u32 s = 0; s < section_count; s++) {
        const LoadedCodeSection* section = &sections[s];
        if (section->size == 0 || !section->data)
            continue;

        const u8* section_data = section->data;
        u32 base_addr = section->address;
        u32 section_sz = section->size;
        u32 num_insts = section_sz / 4;

        if (section->name && section->name[0] != '\0') {
            printf("decoding %s[%u] %s: %u instructions at 0x%08X\n",
                   section->label, section->index, section->name, num_insts,
                   base_addr);
        } else {
            printf("decoding %s[%u]: %u instructions at 0x%08X\n",
                   section->label, section->index, num_insts, base_addr);
        }

        PPCInst* insts = (PPCInst*)malloc(num_insts * sizeof(PPCInst));
        if (!insts) {
            fprintf(stderr, "error: out of memory\n");
            smc_analysis_free(&smc);
            function_list_free(&funcs);
            fclose(header);
            fclose(manifest);
            return 0;
        }

        u32 decoded = 0, embedded = 0, unknown = 0;
        for (u32 i = 0; i < num_insts; i++) {
            u32 raw = read_be32(section_data + i * 4);
            u32 addr = base_addr + i * 4;
            insts[i] = ppc_decode(raw, addr);
            if (insts[i].op == PPC_OP_UNKNOWN &&
                embedded_data_word(section->embedded_data_mode, raw)) {
                insts[i].embedded_data = true;
            }
            decoded++;
            if (insts[i].embedded_data) {
                embedded++;
            } else if (insts[i].op == PPC_OP_UNKNOWN) {
                unknown++;
            }
        }

        if (embedded != 0) {
            printf("  %u decoded, %u known, %u embedded data, %u unknown\n",
                   decoded, decoded - embedded - unknown, embedded, unknown);
        } else {
            printf("  %u decoded, %u known, %u unknown\n", decoded,
                   decoded - unknown, unknown);
        }

        if (section->embedded_data_mode == EMBEDDED_DATA_DOL) {
            analyze_smc_section(sections, section_count, insts, num_insts,
                                &smc);
            if (smc.allocation_failed) {
                fprintf(stderr, "error: out of memory\n");
                smc_analysis_free(&smc);
                function_list_free(&funcs);
                free(insts);
                fclose(header);
                fclose(manifest);
                return 0;
            }
        }

        u32 section_job_count =
            (num_insts + chunk_instructions - 1u) / chunk_instructions;
        ChunkJob* chunk_jobs =
            (ChunkJob*)calloc(section_job_count, sizeof(ChunkJob));
        if (!chunk_jobs) {
            fprintf(stderr, "error: out of memory\n");
            smc_analysis_free(&smc);
            function_list_free(&funcs);
            free(insts);
            fclose(header);
            fclose(manifest);
            return 0;
        }

        for (u32 start = 0; start < num_insts; start += chunk_instructions) {
            u32 chunk_count = num_insts - start;
            u32 func_addr = base_addr + start * 4u;
            char chunk_name[128];
            u32 job_index = start / chunk_instructions;

            if (chunk_count > chunk_instructions)
                chunk_count = chunk_instructions;

            if (snprintf(chunk_name, sizeof(chunk_name),
                         "chunk_%04u_%s%u_%08X.c", file_count, section->label,
                         section->index,
                         func_addr) >= (int)sizeof(chunk_name)) {
                fprintf(stderr, "error: chunk name is too long\n");
                smc_analysis_free(&smc);
                function_list_free(&funcs);
                free(chunk_jobs);
                free(insts);
                fclose(header);
                fclose(manifest);
                return 0;
            }

            ChunkJob* job = &chunk_jobs[job_index];
            job->insts = insts + start;
            job->count = chunk_count;
            job->func_addr = func_addr;

            if (!join_path(job->path, sizeof(job->path), chunks_dir,
                           chunk_name)) {
                fprintf(stderr, "error: chunk path is too long\n");
                smc_analysis_free(&smc);
                function_list_free(&funcs);
                free(chunk_jobs);
                free(insts);
                fclose(header);
                fclose(manifest);
                return 0;
            }

            if (snprintf(job->include_name, sizeof(job->include_name), "%s",
                         include_name) >= (int)sizeof(job->include_name)) {
                fprintf(stderr, "error: output include name is too long\n");
                smc_analysis_free(&smc);
                function_list_free(&funcs);
                free(chunk_jobs);
                free(insts);
                fclose(header);
                fclose(manifest);
                return 0;
            }

            emit_chunk_prototype(header, func_addr);
            if (!function_list_add(&funcs, func_addr,
                                   func_addr + chunk_count * 4u)) {
                smc_analysis_free(&smc);
                function_list_free(&funcs);
                free(chunk_jobs);
                free(insts);
                fclose(header);
                fclose(manifest);
                return 0;
            }
            fprintf(manifest, "// %s/%s\n", chunks_label, chunk_name);
            file_count++;
        }

        u32 active_jobs = effective_chunk_jobs(section_job_count, jobs);
        printf("  writing %u chunks with %u job%s\n", section_job_count,
               active_jobs, active_jobs == 1 ? "" : "s");
        if (!run_chunk_jobs(chunk_jobs, section_job_count, jobs)) {
            smc_analysis_free(&smc);
            function_list_free(&funcs);
            free(chunk_jobs);
            free(insts);
            fclose(header);
            fclose(manifest);
            return 0;
        }

        free(chunk_jobs);
        free(insts);
    }

    {
        char smc_report_path[1100];
        if (snprintf(smc_report_path, sizeof(smc_report_path), "%s_smc.txt",
                     stem) >= (int)sizeof(smc_report_path) ||
            !write_smc_report(&smc, smc_report_path)) {
            smc_analysis_free(&smc);
            function_list_free(&funcs);
            fclose(header);
            fclose(manifest);
            return 0;
        }
    }

    if (smc.possible) {
        u32 display_count = smc.range_count;
        if (display_count > SMC_DISPLAY_RANGE_LIMIT)
            display_count = SMC_DISPLAY_RANGE_LIMIT;

        printf("warning: this DOL may patch executable memory at runtime. "
               "generated code many need additional patches\n");
        printf("  possible patching instructions:\n");
        for (u32 i = 0; i < display_count; i++) {
            printf("    0x%08X-0x%08X\n", smc.ranges[i].start,
                   smc.ranges[i].end);
        }

        if (smc.range_count > SMC_DISPLAY_RANGE_LIMIT) {
            printf("    ...\n");
            printf("  full list: %s_smc.txt\n", stem);
        }
    }

    emit_dispatch_helpers(header, &funcs, entry_point);
    emit_footer(header);
    smc_analysis_free(&smc);
    function_list_free(&funcs);
    fprintf(manifest, "\n// %u C files\n", file_count);

    fclose(header);
    fclose(manifest);

    printf("done!\n");
    printf("  header: %s\n", header_path);
    if (symbols)
        printf("  symbols: %s\n", symbol_header_path);
    printf("  chunks: %s (%u files)\n", chunks_dir, file_count);
    return 1;
}

int emit_dol_split(const DOLFile* dol, const char* output_path,
                          DolRecompCPU cpu, u32 jobs, int local_chunks_dir,
                          const DolRecompSymbolMap* symbols,
                          const CliOptions* options) {
    LoadedCodeSection sections[DOL_NUM_TEXT];
    u32 section_count = 0;

    for (u32 i = 0; i < DOL_NUM_TEXT; i++) {
        if (dol->header.text_sizes[i] == 0)
            continue;

        const u8* data = dol_get_text_section(dol, (int)i);
        if (!data)
            continue;

        LoadedCodeSection* section = &sections[section_count++];
        section->label = "text";
        section->name = NULL;
        section->data = data;
        section->index = i;
        section->file_offset = dol->header.text_offsets[i];
        section->address = dol->header.text_addresses[i];
        section->size = dol->header.text_sizes[i];
        section->embedded_data_mode = EMBEDDED_DATA_DOL;
    }

    return emit_code_sections_split(sections, section_count, output_path, cpu,
                                    dol->header.entry_point, jobs,
                                    local_chunks_dir, symbols, options);
}

int emit_rpx_split(const RPXFile* rpx, const char* output_path,
                          DolRecompCPU cpu, u32 jobs, int local_chunks_dir,
                          const CliOptions* options) {
    LoadedCodeSection sections[RPX_MAX_CODE_SECTIONS];

    for (u32 i = 0; i < rpx->code_section_count; i++) {
        const RPXCodeSection* code = &rpx->code_sections[i];
        LoadedCodeSection* section = &sections[i];
        section->label = "rpx";
        section->name = code->name;
        section->data = code->data;
        section->index = i;
        section->file_offset = code->offset;
        section->address = code->address;
        section->size = code->size;
        section->embedded_data_mode = EMBEDDED_DATA_RPX;
    }

    return emit_code_sections_split(sections, rpx->code_section_count,
                                    output_path, cpu, 0, jobs, local_chunks_dir,
                                    NULL, options);
}

int emit_rel_split(const RELFile* rel, const char* output_path,
                          DolRecompCPU cpu, u32 jobs, int local_chunks_dir,
                          const CliOptions* options) {
    LoadedCodeSection* sections = (LoadedCodeSection*)calloc(
        rel->section_count, sizeof(LoadedCodeSection));
    if (!sections) {
        fprintf(stderr, "error: out of memory\n");
        return 0;
    }

    u32 section_count = 0;
    for (u32 i = 0; i < rel->section_count; i++) {
        const RELSection* rel_section = &rel->sections[i];
        if (!rel_section->executable || rel_section->size == 0 ||
            !rel_section->data)
            continue;

        LoadedCodeSection* section = &sections[section_count++];
        section->label = "rel";
        section->name = NULL;
        section->data = rel_section->data;
        section->index = rel_section->index;
        section->file_offset = rel_section->offset;
        section->address = rel_section->address;
        section->size = rel_section->size;
        section->embedded_data_mode = EMBEDDED_DATA_DOL;
    }

    int ok = emit_code_sections_split(sections, section_count, output_path, cpu,
                                      rel->entry_point, jobs, local_chunks_dir,
                                      NULL, options);
    free(sections);
    return ok;
}

typedef struct {
    RELFile rel;
} RELBatchItem;

void rel_batch_free(RELBatchItem* items, u32 count) {
    if (!items)
        return;
    for (u32 i = 0; i < count; i++)
        rel_free(&items[i].rel);
    free(items);
}

u32 align_up_cli(u32 value, u32 alignment, int* ok) {
    u64 result = ((u64)value + alignment - 1u) / alignment * alignment;
    if (result > 0xFFFFFFFFu) {
        *ok = 0;
        return 0;
    }
    return (u32)result;
}

int next_rel_base(const RELFile* rel, u32* cursor) {
    u32 end;
    int ok = 1;
    if (rel->file_size > 0xFFFFFFFFu - rel->base_address ||
        rel->bss_size > 0xFFFFFFFFu - rel->base_address - rel->file_size) {
        fprintf(stderr, "error: REL auto address range overflow\n");
        return 0;
    }

    end = rel->base_address + rel->file_size + rel->bss_size;
    *cursor = align_up_cli(end, REL_AUTO_ALIGN, &ok);
    if (!ok) {
        fprintf(stderr, "error: REL auto address range overflow\n");
        return 0;
    }
    return 1;
}

int check_duplicate_rel_module(const RELBatchItem* items, u32 count,
                                      u32 module_id) {
    for (u32 i = 0; i < count; i++) {
        if (items[i].rel.module_id == module_id) {
            fprintf(stderr, "error: duplicate REL module id %u\n", module_id);
            return 0;
        }
    }
    return 1;
}

int emit_rel_directory(const char* input_dir, const char* output_root,
                              const char* title_id, int titleless_mode,
                              DolRecompCPU cpu, u32 jobs, u32 start_base,
                              const CliOptions* options) {
    PathList paths = {0};
    RELBatchItem* items = NULL;
    RELModuleMapEntry* map_entries = NULL;
    char generated_root[1200];
    int ok = 0;
    u32 cursor = start_base;

    if (!collect_rel_paths(input_dir, &paths))
        goto done;
    path_list_sort(&paths);

    if (paths.count == 0) {
        fprintf(stderr, "error: no .rel files found in '%s'\n", input_dir);
        goto done;
    }

    items = (RELBatchItem*)calloc(paths.count, sizeof(*items));
    map_entries = (RELModuleMapEntry*)calloc(paths.count, sizeof(*map_entries));
    if (!items || !map_entries) {
        fprintf(stderr, "error: out of memory\n");
        goto done;
    }

    printf("found %u REL module%s\n", paths.count, paths.count == 1 ? "" : "s");
    for (u32 i = 0; i < paths.count; i++) {
        if (!rel_load_image(&items[i].rel, paths.paths[i], cursor))
            goto done;
        if (!check_duplicate_rel_module(items, i, items[i].rel.module_id))
            goto done;

        map_entries[i].module_id = items[i].rel.module_id;
        map_entries[i].rel = &items[i].rel;

        printf("  module %u: %s -> base 0x%08X\n", items[i].rel.module_id,
               paths.paths[i], items[i].rel.base_address);
        if (!next_rel_base(&items[i].rel, &cursor))
            goto done;
    }

    RELModuleMap map = { map_entries, paths.count };
    for (u32 i = 0; i < paths.count; i++) {
        if (!rel_apply_relocations(&items[i].rel, &map))
            goto done;
    }

    if (!build_generated_folder_path(output_root, title_id, titleless_mode,
                                     generated_root, sizeof(generated_root))) {
        goto done;
    }

    for (u32 i = 0; i < paths.count; i++) {
        char rel_output_path[1200];
        printf("\nREL %u/%u: %s\n", i + 1, paths.count, paths.paths[i]);
        rel_print_info(&items[i].rel, NULL);
        if (!build_rel_output_path(generated_root, paths.paths[i],
                                   items[i].rel.module_id, rel_output_path,
                                   sizeof(rel_output_path))) {
            goto done;
        }
        printf("\nwriting output to: %s\n", rel_output_path);
        if (!emit_rel_split(&items[i].rel, rel_output_path, cpu, jobs, 1,
                            options))
            goto done;
    }

    ok = 1;

done:
    free(map_entries);
    rel_batch_free(items, paths.count);
    path_list_free(&paths);
    return ok;
}
