#!/usr/bin/env python3
"""Make a MoEArc build artefact relocatable by shortening ELF path strings.

`crates/moearc-kernels/build.rs` sets the kernel object's `DT_SONAME` to its absolute
`OUT_DIR` path, because that is the only channel that reaches a downstream binary (see
`docs/packaging.md`). `ld` copies that string into every consumer's `DT_NEEDED`, and glibc
treats a `DT_NEEDED` containing a slash as a path rather than a name to search for. Excellent
for a development build; fatal for a tarball, where the build tree does not exist.

This rewrites those two entries to the bare file name so the loader searches for it normally
and finds the copy the launcher put on `LD_LIBRARY_PATH`.

# Why this is not `patchelf`

It does not need to be, and not needing to be is the point. `DT_SONAME` and `DT_NEEDED` hold
*offsets* into `.dynstr`, and the bare name we want is already a suffix of the string that is
there: `.../out/libmoearc_kernels.so` ends with `libmoearc_kernels.so`. So the edit is to add
the length of the directory prefix to the offset. No section is resized, no byte of `.dynstr`
is written, nothing is relocated, and the result differs from the input in exactly the 8 bytes
of one `Elf64_Dyn.d_val` per entry. `patchelf` rewrites program headers to do the same job,
which is more machinery and more ways to be subtly wrong — and it is not installed on the
build host, which is how the cheaper answer got looked for.

Only entries whose string contains `/` are touched, so running it twice is a no-op.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

DT_NULL, DT_NEEDED, DT_SONAME, DT_STRTAB = 0, 1, 14, 5
SHT_DYNAMIC, SHT_STRTAB = 6, 3


class NotElf64(Exception):
    pass


def _sections(buf: bytes):
    """Yield (sh_type, sh_offset, sh_size, sh_link, sh_entsize) for a little-endian ELF64."""
    if buf[:4] != b"\x7fELF" or buf[4] != 2 or buf[5] != 1:
        raise NotElf64("not a little-endian 64-bit ELF")
    e_shoff, = struct.unpack_from("<Q", buf, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", buf, 0x3A)
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_type, = struct.unpack_from("<I", buf, off + 0x04)
        sh_link, = struct.unpack_from("<I", buf, off + 0x28)
        sh_offset, sh_size = struct.unpack_from("<QQ", buf, off + 0x18)
        yield sh_type, sh_offset, sh_size, sh_link, off


def _dynamic(buf: bytes):
    """Return (dyn_offset, dyn_size, dynstr_offset) or None if there is no .dynamic."""
    secs = list(_sections(buf))
    for sh_type, sh_offset, sh_size, sh_link, hdr_off in secs:
        if sh_type == SHT_DYNAMIC:
            # sh_link of .dynamic is the string table it indexes. Trusting that rather than
            # matching on the section name, because section names are stripped far more often
            # than sh_link is wrong.
            str_hdr = secs[sh_link]
            if str_hdr[0] != SHT_STRTAB:
                raise NotElf64(".dynamic sh_link does not point at a string table")
            return sh_offset, sh_size, str_hdr[1]
    return None


def _cstr(buf: bytes, at: int) -> str:
    end = buf.index(b"\0", at)
    return buf[at:end].decode("utf-8", "surrogateescape")


def make_relocatable(path: Path, verbose: bool = True) -> int:
    """Point every path-valued DT_NEEDED/DT_SONAME at the basename inside the same string."""
    buf = bytearray(path.read_bytes())
    found = _dynamic(bytes(buf))
    if found is None:
        return 0
    dyn_off, dyn_size, strtab = found
    changed = 0
    for i in range(dyn_size // 16):
        at = dyn_off + i * 16
        tag, val = struct.unpack_from("<qQ", buf, at)
        if tag == DT_NULL:
            break
        if tag not in (DT_NEEDED, DT_SONAME):
            continue
        s = _cstr(bytes(buf), strtab + val)
        if "/" not in s:
            continue
        base = s.rsplit("/", 1)[1]
        if not base:
            raise SystemExit(f"{path}: {s!r} ends in a slash and has no file name")
        struct.pack_into("<Q", buf, at + 8, val + (len(s) - len(base)))
        changed += 1
        if verbose:
            kind = "SONAME" if tag == DT_SONAME else "NEEDED"
            print(f"  {path.name}: {kind} {s} -> {base}")
    if changed:
        path.write_bytes(bytes(buf))
    return changed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print("usage: elf-relocatable.py FILE [FILE ...]", file=sys.stderr)
        return 2
    total = 0
    for name in argv[1:]:
        p = Path(name)
        try:
            total += make_relocatable(p)
        except NotElf64 as e:
            print(f"{p}: skipped, {e}", file=sys.stderr)
    print(f"elf-relocatable: rewrote {total} entr{'y' if total == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
