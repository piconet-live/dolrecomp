#!/usr/bin/env python3
"""fastlookup.py generated.h : replace DolRecomp's linear dolrecomp_find_original chain (thousands of range
tests per dispatch in LLVM output) with an O(1) table built by evaluating the same rules in order.

Beside generated.h it writes native_tables.c (the module template compiles it when it exists): the same lookup for
generated code, and the means for the host to keep native code in step with what it has verified:
  - dolrecomp_native_index / dolrecomp_native_entries: an indirect call (DOLRECOMP_INDIRECT_CALLS=native) indexes
    them and calls the body, or leaves for the run loop when the entry is null;
  - dolrecomp_native_valid_<address>: the byte each function body tests at its entry (defined, set, in the function's
    own object; absent with DOLRECOMP_ENTRY_CHECK=off).
Both start out permitting everything, so a host that knows nothing of them runs the module as before.
staticrecomp_set_native_valid(start, end, state) is for a host that verifies guest code: for every function with
code in [start, end), bit 0 of state sets the entry byte (the code is verified) and bit 1 the table entry (it may
also be called through the table). Such a host clears everything when it loads the module.

It reads the build's switches from its environment, in main and nowhere else."""
import os
import re
import sys

HEAD = 'static inline DolRecompFunction dolrecomp_find_original(u32 address) {'
BLOCK = re.compile(r'u32 offset = address - (0x[0-9A-F]+)u;\s*if \(offset < (0x[0-9A-F]+)u && \(offset & 3u\) == 0u\) \{\s*'
                   r'static const DolRecompFunction chunk_functions\[\] = \{([^}]*)\};\s*return chunk_functions\[offset / (0x[0-9A-F]+)u\];')
RANGE = re.compile(r'if \(address >= (0x[0-9A-F]+)u && address < (0x[0-9A-F]+)u && \(\(address - 0x[0-9A-F]+u\) & 3u\) == 0u\) return (func_[0-9A-F]+);')
EXPORT = """#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
"""
# The write-gather pipe for DOLRECOMP_GATHER_PIPE=inline: the address of the host's pipe pointer, the start of the
# pipe buffer and the host's flush function. All null until a host calls staticrecomp_set_gather_pipe, and while
# they are null a pipe store takes the hardware-write hook as before.
GATHER_PIPE = """struct CPUState;
struct dolrecomp_gather_pipe_t {
    unsigned char** position;
    unsigned char* start;
    void (*flush)(struct CPUState*);
} dolrecomp_gather_pipe = {0, 0, 0};
""" + EXPORT + """void staticrecomp_set_gather_pipe(unsigned char** position, unsigned char* start, void (*flush)(struct CPUState*)) {
    dolrecomp_gather_pipe.position = flush ? position : 0;
    dolrecomp_gather_pipe.start = start;
    dolrecomp_gather_pipe.flush = flush;
}
"""
SET_VALID = EXPORT + """void staticrecomp_set_native_valid(unsigned int start, unsigned int end, int state) {
    unsigned int address;
    for (address = start & ~3u; address < end; address += 4u) {
        unsigned int slot = (address - dolrecomp_native_base) >> 2;
        unsigned short number;
        if (address < dolrecomp_native_base || slot >= dolrecomp_native_count) continue;
        number = dolrecomp_native_index[slot];
        if (!number) continue;
        dolrecomp_native_entries[number] = (state & 2) ? all_entries[number] : 0;
"""


def rules_of(body):
    """The chain's rules in its order: ('b', base, span, functions, stride) for a chunk block, ('r', from, to,
    function) for a range."""
    rules = []
    for m in sorted(list(BLOCK.finditer(body)) + list(RANGE.finditer(body)), key=lambda m: m.start()):
        if m.re is BLOCK:
            funcs = [f.strip() for f in m.group(3).split(',') if f.strip()]
            rules.append(('b', int(m.group(1), 16), int(m.group(2), 16), funcs, int(m.group(4), 16)))
        else:
            rules.append(('r', int(m.group(1), 16), int(m.group(2), 16), m.group(3)))
    return rules


def table_of(rules):
    """The lowest address the rules cover, and the function of every fourth address from it (None: none)."""
    lo = min(r[1] for r in rules)
    hi = max(r[1] + r[2] if r[0] == 'b' else r[2] for r in rules)
    table = [None] * ((hi - lo) // 4)
    for r in reversed(rules):  # earlier rules win: apply them last
        if r[0] == 'b':
            _, base, span, funcs, stride = r
            for off in range(0, span, 4):
                table[(base - lo + off) // 4] = funcs[off // stride]
        else:
            _, a, b, f = r
            for addr in range(a, b, 4):
                table[(addr - lo) // 4] = f
    return lo, table


def lookup(lo, n, names, idx):
    """The table and the new dolrecomp_find_original, as text for generated.h."""
    out = [f'static const DolRecompFunction fz_fast_funcs[{len(names) + 1}] = {{0,']
    out += [f'    {f},' for f in names] + ['};']
    out.append(f'static const unsigned short fz_fast_index[{n}] = {{{idx}}};')
    return '\n'.join(out) + f'''
static inline DolRecompFunction dolrecomp_find_original(u32 address) {{
    u32 i = (address - 0x{lo:08X}u) >> 2;
    if ((address & 3u) != 0u || i >= {n}u) return 0;
    return fz_fast_funcs[fz_fast_index[i]];
}}
'''


def native_tables(lo, n, names, idx, flags, closed):
    """native_tables.c as text. `flags`: the functions test an entry byte. `closed`: they have closed bodies, each
    with its dolrecomp_closed_ok_<address>: whether the closed body may be entered. 2 until the host says anything
    (then only while no call hook is installed); with state reported: 1 when the code is verified and the function
    may be called through the table (the host intercepts nothing in it), else 0."""
    count = len(names) + 1
    entries = ''.join(f'    {f}_budget,\n' for f in names)
    c = ['/* entry tables for native calls (fastlookup.py) */\n']
    c.append(''.join(f'extern void {f}_budget(void);\n' for f in names))
    c.append(f'const unsigned int dolrecomp_native_base = 0x{lo:08X}u;\nconst unsigned int dolrecomp_native_count = {n}u;\n')
    c.append(f'const unsigned short dolrecomp_native_index[{n}] = {{{idx}}};\n')
    c.append(f'static void (*const all_entries[{count}])(void) = {{0,\n' + entries + '};\n')
    c.append(f'void (*dolrecomp_native_entries[{count}])(void) = {{0,\n' + entries + '};\n')
    if flags:
        c.append(''.join(f'extern unsigned char dolrecomp_native_valid_{f[5:]};\n' for f in names))
        c.append(f'static unsigned char* const valid_flags[{count}] = {{0,\n'
                 + ''.join(f'    &dolrecomp_native_valid_{f[5:]},\n' for f in names) + '};\n')
    if closed:
        c.append(''.join(f'extern unsigned char dolrecomp_closed_ok_{f[5:]};\n' for f in names))
        c.append(f'static unsigned char* const closed_flags[{count}] = {{0,\n'
                 + ''.join(f'    &dolrecomp_closed_ok_{f[5:]},\n' for f in names) + '};\n')
    c.append(GATHER_PIPE)
    c.append(SET_VALID + ("        *valid_flags[number] = (unsigned char)(state & 1);\n" if flags else "")
             + ("        *closed_flags[number] = (unsigned char)((state & 3) == 3);\n" if closed else "") + """    }
}
""")
    return ''.join(c)


def main():
    path = sys.argv[1]
    flags = os.environ.get('DOLRECOMP_ENTRY_CHECK') != 'off'
    closed = flags and bool(os.environ.get('DOLRECOMP_CLOSED_ROOTS'))
    src = open(path).read()
    start = src.index(HEAD)
    end = src.index('\n}\n', start) + 3
    body = src[start:end]
    if 'fz_fast_index' in body:
        sys.exit('already patched')
    lo, table = table_of(rules_of(body))
    names = sorted({f for f in table if f})
    ids = {f: i + 1 for i, f in enumerate(names)}
    idx = ','.join(str(ids.get(f, 0)) if f else '0' for f in table)
    # keep the original chain (unused) so gen_module_tables.py still finds the coverage ranges
    slow = body.replace('dolrecomp_find_original(', 'fz_find_original_slow(', 1).replace('static inline', 'static inline DOLRECOMP_UNUSED', 1)
    open(path, 'w').write(src[:start] + slow + lookup(lo, len(table), names, idx) + src[end:])
    with open(os.path.join(os.path.dirname(path), 'native_tables.c'), 'w') as c:
        c.write(native_tables(lo, len(table), names, idx, flags, closed))


if __name__ == '__main__':
    main()
