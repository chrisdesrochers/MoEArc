# packaging/

How a MoEArc build becomes something a stranger can run.

The reasoning is in [`../docs/packaging.md`](../docs/packaging.md); the licence position is in
[`THIRD-PARTY.md`](THIRD-PARTY.md). This file is the operator's view: what each piece is and
how to drive it.

## The three commands

```sh
packaging/bundle.sh --build                  # build, assemble, tar  -> dist/
packaging/verify-clean.sh                    # run it where oneAPI does not exist
bench/reproduce.sh <model.gguf>              # reproduce the headline number, with provenance
```

Publishing what comes out of them is [`RELEASE.md`](RELEASE.md) — tag name, asset names,
checksums, and the clean-room gate, in the order they have to happen.

`verify-clean.sh` is not optional before publishing a tarball. It is the only step that
actually answers the question the artefact exists to answer.

🔴 **It proves less than it used to, and the gap is the important part.** It used to take a
model — `MOEARC_VERIFY_MODEL` — and run a real forward pass in the container, checking the
output against stored token ids. That gate ran `moearc-bench`, which left the payload with the
retired SYCL engine, along with `moearc-selftest` and `moearc-server`. What survives proves the
Arc card is found on a machine that has never had oneAPI. It does **not** prove a model loads,
and `docs/packaging.md` measured that those are different questions: a driver stack can pass
detection and still fail inference. The script prints `NOT PROVEN` for exactly that, and the
gate returns when llama.cpp is bundled.

`MOEARC_CLEAN_BASE` picks the distro to test against; `MOEARC_RUNTIME_CACHE` points at a
directory to reuse between runs so the 230 MB fetch happens once.

## Files

| | |
| --- | --- |
| `bundle.sh` | assembles `dist/moearc-<version>-linux-x86_64.tar.gz`. `--build` runs cargo first; `--with-runtime` vendors Intel's runtime instead of fetching it (read `THIRD-PARTY.md` before you do). |
| `elf-relocatable.py` | rewrites an absolute `DT_SONAME` and the matching `DT_NEEDED` down to a bare name. ⬜ **Not run today** — the only object that ever had one was the retired kernel build's. Kept because bundling llama.cpp's shared objects will need it again. `bundle.sh` still asserts the property it existed for: no `DT_NEEDED` in any staged binary may contain a slash. |
| `launcher.sh` | sets `LD_LIBRARY_PATH` and execs the real binary in `libexec/`. This is what closes the dlopen gap. Installed under one name now, having been installed under four. |
| `fetch-runtime.py` | downloads Intel's published SYCL runtime, verified against pinned digests. Standard library only; no `pip`. |
| `runtime.lock.json` | the pins. Versions, SHA-256, per-package file allowlist, licences. |
| `install.sh` | the `curl \| sh` entry point: download, unpack, fetch the runtime, link onto `PATH`. Publishes nothing itself — it expects the assets `RELEASE.md` names, and says exactly that when they are not there. |
| `RELEASE.md` | the owner's checklist for cutting a release. One asset name for every tag, and why. |
| `Containerfile.clean` | Ubuntu 24.04 + the Intel GPU driver + nothing else. `DRIVER=distro` reproduces the too-old-driver case deliberately. |
| `verify-clean.sh` | runs a tarball in that container and asserts, by name, that the Arc card is found. |
| `THIRD-PARTY.md` | what is redistributed, what is not, and why. |

## The layout it produces

```
moearc-<version>-linux-x86_64/
  moearc                <- launcher.sh
  libexec/
    moearc              <- the real ELF binary
    fetch-runtime.py
  runtime/            <- Intel's SYCL runtime: fetched at install, or vendored with --with-runtime
  share/moearc/       <- runtime.lock.json, BUILD-INFO.txt
  share/doc/moearc/   <- LICENSE, NOTICE, THIRD-PARTY.md
  bench/              <- reproduce.sh and the reference token ids
```

The launcher dispatches on the name it was invoked as, so `ln -s .../moearc ~/.local/bin/moearc`
works and the bundle stays one directory.

## Two things that will look like bugs and are not

**`moearc` needs no SYCL runtime, and it is now the only binary in the tarball.** The device
report talks to Level Zero directly, which is deliberate — the first thing a new user runs has
to work before anything has been downloaded, and it has to be able to explain a machine where
the GPU stack is broken. So `moearc` runs immediately after unpacking.

⬜ **Which leaves `runtime/` with no consumer inside the bundle today.** It was fetched for
`libmoearc_kernels.so`, which is retired. `install.sh` still fetches it, and a user's own
llama.cpp SYCL build can resolve against it through the launcher, but that is a side effect
rather than a design. This resolves either way once llama.cpp is bundled — decide it then,
not by deleting the machinery now.

**Passing a container all of `/dev/dri` makes Intel's driver abort at teardown.** The workload
succeeds, prints its result, and *then* dies with
`Abort was called at 433 line in file: ./shared/source/os_interface/linux/drm_neo.cpp`.
Pass the render node — `--device /dev/dri/renderD129` — not the directory. `verify-clean.sh`
does this for you.

## The tarball is byte-repeatable, and only that

`bundle.sh` pins the `built:` stamp and every tar member's mtime, uid, gid and order to
`SOURCE_DATE_EPOCH`, which defaults to the commit's own timestamp; `gzip -n` keeps the name and
time out of the gzip header. Two runs of the same commit on the same machine produce the same
`sha256` — measured, twice, before the claim was written down.

⚠️ **That is repeatability on one machine, not reproducibility from source by a third party.**
Nobody has built this on a second host and compared, so a different `rustc` or `icpx` may well
produce different bytes. `BUILD-INFO.txt` records the toolchain so the build is auditable; do
not upgrade that into a reproducible-build claim.

`bundle.sh` also warns when `BUILD-INFO.txt` comes out with an `unknown` field or a dirty tree.
Both are undramatic ways to publish an artefact that cannot say what produced it: running it in
a shell where `rustc` is not on `PATH` is enough.

## Rebuilding the pins

`runtime.lock.json` is written by hand and verified by use. To move to a newer Intel runtime,
change the versions, take the SHA-256s from the index, and then run `verify-clean.sh` — the
digest check will catch a typo, and only the clean run will catch an ABI break.

🔴 The adapter list is not decoration. `libur_adapter_level_zero.so.0` alone, without `_v2` and
`_opencl` beside it, fails with `UR_RESULT_ERROR_UNSUPPORTED_VERSION` — which reads as a version
mismatch and is not one. A partial adapter set does not degrade; it fails.
