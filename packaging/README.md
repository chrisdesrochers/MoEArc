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

`verify-clean.sh` is not optional before publishing a tarball. It is the only step that
actually answers the question the artefact exists to answer.

🔴 **Give it a model.** Without one it stops at "SYCL found the card", and a driver stack can
pass that and still be unable to load a model — measured, see `docs/packaging.md`. With one it
also runs a real forward pass and checks the output against a stored llama.cpp reference:

```sh
MOEARC_VERIFY_MODEL=/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf \
MOEARC_VERIFY_MODEL_IDS="510 5347 273 6181 310" \
MOEARC_VERIFY_MODEL_REF=/opt/m/bench/references/olmoe-1b-7b.capital.ids \
  packaging/verify-clean.sh
```

`MOEARC_CLEAN_BASE` picks the distro to test against; `MOEARC_RUNTIME_CACHE` points at a
directory to reuse between runs so the 230 MB fetch happens once.

## Files

| | |
| --- | --- |
| `bundle.sh` | assembles `dist/moearc-<version>-linux-x86_64.tar.gz`. `--build` runs cargo first; `--with-runtime` vendors Intel's runtime instead of fetching it (read `THIRD-PARTY.md` before you do). |
| `elf-relocatable.py` | rewrites the kernel object's `DT_SONAME` and the matching `DT_NEEDED` from this build tree's absolute path down to a bare name. Without it the tarball is a development build that only runs on the machine that produced it. |
| `launcher.sh` | installed under four names; sets `LD_LIBRARY_PATH` and execs the real binary in `libexec/`. This is what closes the dlopen gap. |
| `fetch-runtime.py` | downloads Intel's published SYCL runtime, verified against pinned digests. Standard library only; no `pip`. |
| `runtime.lock.json` | the pins. Versions, SHA-256, per-package file allowlist, licences. |
| `install.sh` | the `curl \| sh` entry point: download, unpack, fetch the runtime, link onto `PATH`. |
| `Containerfile.clean` | Ubuntu 24.04 + the Intel GPU driver + nothing else. `DRIVER=distro` reproduces the too-old-driver case deliberately. |
| `verify-clean.sh` | runs a tarball in that container and asserts, by name, that the Arc card is found. |
| `THIRD-PARTY.md` | what is redistributed, what is not, and why. |

## The layout it produces

```
moearc-<version>-linux-x86_64/
  moearc  moearc-server  moearc-bench  moearc-selftest   <- four copies of launcher.sh
  libexec/
    moearc  moearc-server  moearc-bench  moearc-selftest <- the real ELF binaries
    libmoearc_kernels.so                                 <- our SYCL kernels
    fetch-runtime.py
  runtime/            <- Intel's SYCL runtime: fetched at install, or vendored with --with-runtime
  share/moearc/       <- runtime.lock.json, BUILD-INFO.txt
  share/doc/moearc/   <- LICENSE, NOTICE, THIRD-PARTY.md
  bench/              <- reproduce.sh and the reference token ids
```

The launcher dispatches on the name it was invoked as, so `ln -s .../moearc ~/.local/bin/moearc`
works and the bundle stays one directory.

## Two things that will look like bugs and are not

**`moearc` needs no SYCL runtime; everything else does.** The device report talks to Level Zero
directly, which is deliberate — the first thing a new user runs has to work before anything has
been downloaded, and it has to be able to explain a machine where the GPU stack is broken. So
`moearc` runs immediately after unpacking and `moearc-server` triggers the runtime fetch.

**Passing a container all of `/dev/dri` makes Intel's driver abort at teardown.** The workload
succeeds, prints its result, and *then* dies with
`Abort was called at 433 line in file: ./shared/source/os_interface/linux/drm_neo.cpp`.
Pass the render node — `--device /dev/dri/renderD129` — not the directory. `verify-clean.sh`
does this for you.

## Rebuilding the pins

`runtime.lock.json` is written by hand and verified by use. To move to a newer Intel runtime,
change the versions, take the SHA-256s from the index, and then run `verify-clean.sh` — the
digest check will catch a typo, and only the clean run will catch an ABI break.

🔴 The adapter list is not decoration. `libur_adapter_level_zero.so.0` alone, without `_v2` and
`_opencl` beside it, fails with `UR_RESULT_ERROR_UNSUPPORTED_VERSION` — which reads as a version
mismatch and is not one. A partial adapter set does not degrade; it fails.
