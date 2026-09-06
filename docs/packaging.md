# Packaging: how one binary brings its own dependencies

`ux.md` promises a single binary that installs in one step and never hands the user a list of
things to go install. That promise has to survive contact with three native dependencies. This
records how, and what is actually decided versus still open.

Technique reference: [Vendoring C/C++ dependencies in
Rust](https://blog.veeso.dev/blog/en/vendoring-c-cpp-dependencies-in-rust/), whose `-src` /
`-sys` / public three-crate split and `include_bytes!` + `libloading` fallback are the two
patterns we lean on.

## The three problems, which are not the same problem

| | what it is | can it be statically linked? | approach |
| --- | --- | --- | --- |
| **SYCL kernels** | our own C++/SYCL, compiled with DPC++ | no — needs the SYCL runtime | build on our machine, embed the `.so` |
| **Level Zero loader** | `libze_loader.so`, the ICD loader | no by design — it *is* a loader | embed, extract, `dlopen` |
| **TLS trust roots** | CA certificates for Hub downloads | n/a — data, not code | see below |

### SYCL kernels — build here, ship the artifact

The kernels cannot be Rust. They are compiled with `icpx -fsycl` on a machine with oneAPI
installed, which is **our** machine, not the user's. That distinction matters and was nearly
got wrong: "requires the oneAPI toolkit to build" sounds disqualifying and is not, because the
cost lands on the build box and the user receives a compiled artifact.

Three-crate split, following the guide:

- **`moearc-kernels-src`** — owns the SYCL source and a `build.rs` that drives `icpx`. Only
  built when the `vendored` feature is on.
- **`moearc-kernels-sys`** — hand-written `extern "C"` declarations. **No bindgen**: SYCL
  headers transitively pull `<sycl/sycl.hpp>`, and requiring those to build the Rust side would
  reintroduce the toolkit dependency we just moved off the user. This is the same reasoning
  that made the tch-rs XPU patch forward-declare its symbols rather than include the headers.
- **`moearc-kernels`** — the safe API the engine consumes.

✅ **Settled by building it: a shared object, linked by `icpx` itself.**

The first attempt produced a static archive with `ar` and let cargo link it with `cc`. It failed
on `undefined symbol: _intel_fast_memcpy` — a symbol from `libintlc`, one of several Intel
runtime libraries (`intlc`, `irc`, `imf`, `svml`, `irng`, …) that `icpx` links automatically and
`cc` knows nothing about. That list is a property of the compiler version, not of our code, so
chasing it is the exact "transitive dependencies must be explicitly linked" pitfall the guide
warns about.

Letting `icpx` perform the link makes it the compiler's problem: the `.so` records its own
dependencies in `DT_NEEDED` and cargo links one library. It is also the shape we ship regardless
— the SYCL runtime cannot be statically linked — so a packaged build embeds this object and
extracts it, exactly the `include_bytes!` + `libloading` route described below.

**Proven on hardware.** `moearc-kernels` compiles `kernels.cpp` with `icpx` from `build.rs` and
the tests run on a real Arc B580: queue creation, a host↔device round trip, and a device-side
expert gather returning correct data for a scattered, out-of-order, repeating index list.
Mutation-tested — an off-by-one in the gather index fails the suite with
`expected expert 63, got 0`.

### Level Zero loader — embed, extract, dlopen

`moearc-device` already `dlopen`s `libze_loader.so.1` via `libloading` rather than linking it,
with `MOEARC_ZE_LOADER` overriding the soname. That was done so the binary still *starts* on a
machine with no Level Zero runtime and can explain itself instead of dying in the dynamic
loader.

That override is also the vendoring seam, which is why it exists. A packaged build embeds a
known-good loader with `include_bytes!`, writes it to a cache directory on first run, and points
`MOEARC_ZE_LOADER` at it. The guide calls this a fallback for when static linking is
unavailable; for an ICD loader it is not a fallback, it is the only correct answer — the whole
job of that library is to find *other* libraries at runtime.

✅ **Decided 2026-09-05, and the answer is neither: the loader is not embedded at all.** See
*What is bundled and what is the system’s* below. Preferring ours guarantees a known-good
version and risks disagreeing with the installed compute runtime it has to talk to — and that
runtime is chosen by the user’s distribution to match the kernel module it ships beside.
`MOEARC_ZE_LOADER` remains, as an override and as the way the loader-is-missing path gets
tested; it is not the packaging default.

### TLS trust roots — the one that is data

`hf-hub` brings `reqwest`; we enable only `rustls-tls`.

✅ **Correcting an earlier claim in `dependencies.md`**: that note said `native-tls` "stays
enabled underneath", so a static binary could not be claimed. That was reasoned from `hf-hub`'s
manifest rather than from the resolved graph, and it is **wrong**. `openssl-sys` and
`native-tls` are **zero nodes workspace-wide**. The only match on `openssl` is `openssl-probe`,
a small pure-Rust crate reached through `rustls-native-certs → rustls-platform-verifier`, which
*locates* the OS trust store and links nothing. Today `moearc` dynamically links only
`libc`, `libm`, `libgcc_s` and the dynamic loader.

The real residual is narrower and worth stating precisely: **we depend on the host having a CA
certificate bundle.** A minimal container without one will fail TLS at download time. `hf-hub`
1.0.0 exposes no `webpki-roots` feature (`rustls-tls = ["reqwest/rustls"]` is all there is), so
bundling roots means either an upstream feature or depending on `reqwest` directly.

⬜ Open: bundle roots for self-containment, or use the system store and fail with a legible
message naming the missing bundle. The second is less work and arguably more correct — a system
CA store is administered, and overriding it silently is its own bad behaviour.

## ✅ Closed: the kernel `.so` now follows the binary

**Status: fixed 2026-09-05. Nothing has to be set to start a binary that links the kernels —
not `LD_LIBRARY_PATH`, not anything else.**

The history is worth keeping, because the obvious fix is the one that does not work.

`moearc-kernels`' `build.rs` used to emit an rpath via `cargo:rustc-link-arg`. **That directive
applies only to the crate that emits it and does not propagate to downstream crates.** So any
binary depending on the kernels — `moearc-server`, and eventually `moearc` itself — linked
fine and then died at startup:

```
error while loading shared libraries: libmoearc_kernels.so: cannot open shared object file
```

📌 **Worth noting how this was missed.** Every crate's tests were green and the workspace built
clean, because the kernels tests run *inside* the crate that carries the rpath — and test
binaries are exactly the target class a `rustc-link-arg` does reach. Nothing in 309 passing
tests executed a binary that depends on the kernels crate. **A suite can be entirely healthy
and still not cover the first thing a user does.**

### The fix: the soname carries the path

Emitting *more* link args cannot help. `-bins`, `-tests`, `-examples` are all scoped to the
emitting crate too, so the `$ORIGIN` arrangement — copy the object next to the binary and
rpath it — has no way to reach a downstream binary from here. Of the build script's outputs
only `rustc-link-search` and `rustc-link-lib` propagate, and `-L` is a *link*-time path that
leaves no trace in the executable.

So the path has to ride inside something that does propagate, and there is exactly one such
thing: the shared object itself. `ld` copies a library's `DT_SONAME` verbatim into the
`DT_NEEDED` entry of everything that links it, and glibc's loader treats a `DT_NEEDED` string
containing a slash as a **path** rather than a name to search for. `build.rs` therefore links
the object with `-Wl,-soname,$OUT_DIR/libmoearc_kernels.so`, and every consumer — binaries,
tests, examples, benches, in this crate and in any other — gets the absolute path with no
cooperation and no environment variable. The crate-local rpaths are gone; there is nothing
left that only works one crate deep.

The object also now carries its own `DT_RUNPATH` into the oneAPI runtime directories, so
`libsycl`, `libsvml`, `libimf`, `libintlc` and `libirng` resolve for every consumer as well.
Those used to be rpaths on this crate's targets and so had the identical non-propagation bug
one level down — it was simply masked by the first failure.

Two properties, stated because they are the questions this shape usually raises:

- **Staleness and concurrency do not apply.** Nothing is copied and nothing is cached. `OUT_DIR`
  is stable for a given crate, profile and feature set and cargo rewrites the object in place;
  anything that moves it to a new `OUT_DIR` relinks its consumers in the same build. There is
  no second copy that could go stale and no temporary file two processes could race on.
- **It is not relocatable, and is not meant to be.** The recorded path is this build tree's
  `OUT_DIR`, so the artifact is a development build. Distribution is still the `include_bytes!`
  + extract + `dlopen` route described above, and that route is unaffected by this: a `dlopen`d
  library is opened by path and never consults `DT_NEEDED` at all.

### The test that would have caught it

`crates/moearc-kernels/tests/clean_env_binary.rs`, against
`crates/moearc-kernels/src/bin/moearc-kernels-smoke.rs` — a binary whose only job is to be
started. It calls one real symbol through the C ABI (an unused `DT_NEEDED` is dropped under
`--as-needed`, so a binary that does not use the library is not evidence about one that does),
prints a marker and exits 0. It needs no GPU: `moearc_ctx_create` returns null rather than
throwing when there is no device.

Three tests. The first runs it under `env_clear()` — not `env -u LD_LIBRARY_PATH` but an
*empty* environment — and fails loudly on `cannot open shared object file`. The second adds a
deliberately wrong `LD_LIBRARY_PATH`, which passes because a `DT_NEEDED` with a slash in it is
never searched for and so cannot be shadowed.

The third is the one that keeps the other two honest. A binary inside `moearc-kernels` is a
fair stand-in for `moearc-server` **only while the build script emits no link args**, because
a crate-local rpath would reach it and not reach `moearc-server` — which is precisely the trap
that hid the original bug. So it reads the ELF and asserts both halves of the mechanism: the
kernel object must be named by an absolute path in `DT_NEEDED`, and no `RPATH`/`RUNPATH` on
the binary may point into the kernels build directory. Reintroduce the old
`cargo:rustc-link-arg` and the first two tests start passing vacuously while this one fails
and says why.

## ✅ Closed: the SYCL runtime no longer needs `setvars.sh` to find a GPU

**Status: fixed 2026-09-05, and verified where it could not have been faked** — Ubuntu 24.04
in a container with no `/opt/intel`, an empty environment, and glibc 2.39 rather than the build
host's 2.43:

```
$ podman run --rm --device /dev/dri/renderD129 --group-add keep-groups moearc-clean:noble ...
oneAPI:            ABSENT
LD_LIBRARY_PATH:   [unset]

  ▸ Intel(R) Arc(TM) B580 Graphics   level_zero   xe / L0 build 33578   11.3 GiB / 11.3 GiB
  ✓ Intel(R) Arc(TM) B580 Graphics is ready — 11.3 GiB free right now.

$ env -i /opt/m/moearc-selftest
moearc-kernels-smoke: ok device=Intel(R) Arc(TM) B580 Graphics
```

The answer is the unglamorous one, and the section above is why: **`LD_LIBRARY_PATH`, set by a
launcher, pointing at a directory that holds the runtime's whole closure.**

That is not a shrug. It is the *only* mechanism that works, and the reason is structural. The
failing lookup is `libur_adapter_level_zero.so.0` → `libumf.so.1`, where the adapter was
`dlopen`ed by `libsycl` with no loader chain back to anything we control. `DT_RUNPATH` is not
inherited across that boundary and `DT_RPATH` cannot be reached from it — both were tried and
both failed, which is recorded above. `LD_LIBRARY_PATH` is the one search path that *is*
consulted for a `dlopen`ed module's own dependencies, because it belongs to the process rather
than to any object in it. So the launcher is not a workaround for not having done the rpath
work; the rpath work has no solution and the process-wide path does.

### The set, and why each member is in it

Eleven libraries, 78 MiB installed. Discovered by `ldd` on our own object and on each adapter, plus
the adapters themselves, which nothing links and `SYCL_UR_TRACE=1` names:

| | why |
| --- | --- |
| `libsycl.so.9` | the SYCL runtime our kernels are linked against |
| `libur_loader.so.0` | `libsycl`'s `DT_NEEDED`; loads the adapters |
| `libur_adapter_level_zero.so.0`, `…_v2.so.0`, `libur_adapter_opencl.so.0` | `dlopen`ed by the loader |
| `libumf.so.1`, `libhwloc.so.15` | the adapters' `DT_NEEDED` — **the original failure** |
| `libimf.so`, `libsvml.so`, `libintlc.so.5`, `libirng.so` | Intel's compiler runtime; in the `DT_NEEDED` of both our kernel object and Intel's adapter |
| Intel's EULA text | ships beside the binaries it covers |

🔴 **The adapter list is load-bearing and a partial set fails misleadingly.** Installing only
`libur_adapter_level_zero.so.0` — the one that is actually selected — produces
`UR adapter initialization failed: 43 (UR_RESULT_ERROR_UNSUPPORTED_VERSION)` and no device.
That reads as an ABI mismatch between the runtime and the adapter, and it is not one: it is a
missing sibling — it cost one wrong hypothesis here before the full set was tried.
`SYCL_UR_TRACE=1` also shows the
**V2** adapter winning device selection, so the one you would have guessed was optional is the
one in use.

Nothing on `libstdc++` is bundled, which was checked rather than assumed: the kernel object
needs `GLIBCXX_3.4.21`, i.e. GCC 5.1 from 2015. Shadowing a user's `libstdc++` from
`LD_LIBRARY_PATH` would also shadow it for `libze_intel_gpu.so.1`, which is theirs and newer.

## ✅ Decided: what is bundled and what is the system's — and it is not one decision

The open question above ("whether the packaged runtime is preferred over a system one") assumed
a single answer. **There are two questions and they go opposite ways**, split on what each
library has to agree with:

- **The SYCL/oneAPI runtime is version-coupled to our compiler.** It has to agree with
  `libmoearc_kernels.so`, which we built. We ship the pin. It goes first on `LD_LIBRARY_PATH`
  and wins over a system oneAPI if one exists.
- **The Level Zero loader and GPU driver are version-coupled to the user's kernel.**
  `libze_loader.so.1` and `libze_intel_gpu.so.1` have to agree with the `xe`/`i915` module
  running on that machine, and a distribution ships them together for that reason. We use
  theirs, and `moearc-device` already had the seam for it — `DEFAULT_LOADER_SONAME` looks the
  loader up by soname so the system's rules find the system's copy.

Both are MIT and could be bundled. Overriding the half of the stack that has to match a kernel
we know nothing about is the place where "we know better" is most likely to be wrong, so
`MOEARC_ZE_LOADER` stays an override rather than becoming the default.

🔴 **That system half has a floor, and on Ubuntu 24.04 it is below Battlemage.** The stock
`libze-intel-gpu1` is Level Zero build 27642 and predates the B580. Measured, because the same
clean-room run was done twice:

```
                            distro driver (27642)        Intel's repo (33578)
moearc --no-tui             Intel(R) Graphics  i915      Intel(R) Arc(TM) B580  xe
                            85.6 GiB "free"               11.3 GiB free
qwen3-235b-a22b 132.2 GiB   "✓ 70/128 experts resident"   "· will not fit"
```

⚠️ **The failure is not that it stops — it is that it does not.** With the old driver `moearc`
enumerates the Arrow Lake iGPU, reports 85.6 GiB of "VRAM", and cheerfully declares a 132 GiB
model will fit. `bench/README.md` already warns that a wrong-device Vulkan run "does not fail,
it succeeds and lies"; this is the same failure reached a different way, and it is exactly the
science-experiment experience `docs/ux.md` exists to prevent. The Arc card is not reported as
present-but-unusable, because Level Zero never exposes it and the `unusable_hardware` field
has nothing to correlate against inside a container.

⬜ **Open, and it belongs to `moearc-device`, not to packaging:** an integrated device offered
as *the* choice on a machine that also has a discrete Arc card is a wrong answer even when it
is the only one Level Zero returned, and a Level Zero build number old enough to predate the
installed hardware is a known-bad configuration `docs/ux.md` says the tool should recognise
and name.

## 🔴 The GPU driver floor is higher for inference than for detection

Found by running an actual model in the clean container rather than stopping at the selftest,
and it is the most useful thing this packaging work turned up.

| driver stack (all on a B580, no oneAPI) | device report | SYCL queue | model load + decode |
| --- | --- | --- | --- |
| Ubuntu 24.04 stock, `libze-intel-gpu1` build 27642 | ❌ enumerates the **iGPU** | — | — |
| Intel client repo for noble, 25.18.33578.15 + gmm 22.7.2 | ✅ B580 | ✅ B580 | ❌ `host-to-device copy failed on the device`, **then SIGSEGV** |
| Ubuntu 26.04, 26.05.37020.3 + gmm 22.9.0 | ✅ B580 | ✅ B580 | ✅ 32.44 tok/s, token ids **16/16** vs llama.cpp |

⚠️ **Each row fails one step later than the one above, and every step before the failure looks
healthy.** A packaging check that stops at "the selftest found the card" declares the middle row
working. It is not: it cannot load a model. The clean-room procedure therefore has to run a real
forward pass, not just create a queue — the same lesson as the 309-green-tests one, applied to
the layer below.

📌 **Not isolated: whether the middle row fails on the Level Zero driver version or on
`libigdgmm12` 22.7.2.** Both differ between the two working and non-working stacks, and
separating them needs a mixed install that was not built. Recorded as unknown rather than
guessed.

🔴 **A load failure segfaults.** `moearc-bench` prints a clean `LOAD FAILED: unsupported model:
device: host-to-device copy failed on the device` row and *then* dies with SIGSEGV — so the
harness reports the failure correctly and the process still crashes. That is a robustness bug in
the engine's teardown path, not in packaging, and it belongs to whoever owns
`crates/moearc-engine`.

📌 **`libigdgmm12` is a `Recommends` of the driver, not a `Depends`.** With
`--no-install-recommends` it is absent, and the failure is
`Abort was called at 15 line in file: ./shared/source/gmm_helper/resource_info.cpp` — *after*
detection and the SYCL queue have both succeeded. `packaging/Containerfile.clean` names it
explicitly and there is a comment there saying why it must stay named.

## The licence position: nothing of Intel's is redistributed

Full detail in [`packaging/THIRD-PARTY.md`](../packaging/THIRD-PARTY.md). The short version,
because it determined the shape of everything above:

`libimf`, `libsvml`, `libintlc` and `libirng` are Intel's proprietary compiler runtime, they
have no open-source counterpart, and they are in the `DT_NEEDED` of both our kernel object and
*Intel's own* Level Zero adapter — so they cannot be dropped. Intel's EULA grants redistribution
of "Redistributables", defined as the files listed in a `redist.txt` — **and no such file exists
anywhere in the oneAPI 2026.1 installation we build against.** The grant is real and we cannot
show it covers any particular file. Two of its conditions would also propagate to our users:
a no-reverse-engineering clause, and a prohibition on SaaS use — which is one of the things an
inference server is for.

So the default tarball contains **no third-party binaries at all**. `packaging/fetch-runtime.py`
downloads Intel's runtime, on the user's machine, from Intel's own channel — packages Intel
publishes precisely so that "executables can be deployed to hosts without the oneAPI
development toolkits" — pinned by SHA-256 in `packaging/runtime.lock.json`. The user accepts
Intel's terms from Intel. The tarball stays Apache-2.0. An honest dependency beats a licence
violation.

`bundle.sh --with-runtime` vendors it anyway, for air-gapped installs — a 29 MB tarball against
the default 4.8 MB, verified under `podman --network none` with `MOEARC_NO_FETCH=1`. That
archive is **not** Apache-2.0 and must not be published as though it were.

## Installing without a network

The fetch is the one step that needs the internet. On a machine that has none:

```sh
# on a machine that does, with the same tarball unpacked:
python3 libexec/fetch-runtime.py --dest ./runtime --lock share/moearc/runtime.lock.json
# then copy ./runtime across, or set MOEARC_RUNTIME_DIR at it
```

`MOEARC_NO_FETCH=1` makes the launcher refuse to download and say so, rather than hanging on a
firewalled resolver.

## The kernel object is relocatable now, without `patchelf`

The section above records that the soname carries an absolute `OUT_DIR` path, that this is the
only channel reaching a downstream binary, and that the artefact is therefore **not
relocatable**. That is still true of a `cargo build`. `packaging/elf-relocatable.py` makes the
*packaged* copy relocatable, and the mechanism is worth stating because it is smaller than it
sounds:

`DT_SONAME` and `DT_NEEDED` hold **offsets into `.dynstr`**, and the string we want is already a
suffix of the string that is there — `…/out/libmoearc_kernels.so` ends with
`libmoearc_kernels.so`. So the edit is to add the length of the directory prefix to the offset.
No section is resized, no byte of `.dynstr` changes, nothing is relocated, and each rewritten
entry differs from the original in exactly the eight bytes of one `Elf64_Dyn.d_val`. It is
idempotent, because it only touches strings containing a slash.

This is what `patchelf` would have been used for. It is not installed on the build host, and
looking for the cheaper answer first found one that is easier to audit than a program that
rewrites program headers. `bundle.sh` then asserts the result — any remaining `DT_NEEDED` with
a slash in it fails the build — because that failure is silent until someone unpacks the
tarball somewhere else, which is precisely how the original bug shipped.

📌 The `include_bytes!` + extract + `dlopen` route described earlier is therefore **not needed
and not implemented.** Embedding the object would mean writing it to a cache directory on first
run and managing that cache's staleness; shipping it in `libexec/` next to the binary that
names it achieves the same thing with a file copy. The `MOEARC_ZE_LOADER` seam stays, because
its purpose was never vendoring — it is how the loader-is-missing path gets tested.

## What is still not true

- 🔴 **glibc 2.39 or newer.** `moearc` requires `GLIBC_2.39`, inherited from building on Ubuntu
  26.04; the other three binaries need only 2.34. That is Ubuntu 24.04, Fedora 40, Debian 13 or
  newer — and it silently excludes Debian 12 and RHEL 9. `share/moearc/BUILD-INFO.txt` in every
  tarball states the measured floor per binary. Lowering it means building on an older host or
  in a manylinux container; nothing else here needs to change.
- 🔴 **x86-64 Linux only.** No aarch64, no Windows.
- ⬜ `moearc serve` is still a fixture — the CLI's device report is real and its model list,
  downloads and serving stats are not. The tarball ships `moearc-server` (real, links the
  kernels) and `moearc-bench` beside it, so nothing in the bundle is a fixture *only*, but the
  four-command journey in `docs/ux.md` is not yet walkable end to end.
- ⬜ `install.sh` points at a GitHub release that does not exist yet, and `moearc.dev` is still
  unregistered. `MOEARC_TARBALL=/path/to/tarball` installs a local build in the meantime, and
  that is the path that was actually exercised end to end in the clean container.
- 📌 **Two installer bugs were found by running it rather than reading it**, and both would have
  hit the first stranger. It probed for `/dev/dri/renderD128` specifically — on any box with an
  iGPU the discrete card is `renderD129`, so it refused to install on exactly the hardware the
  packaging was proved on. And `find "$tmp" -maxdepth 1 -name 'moearc-*'` matched the staging
  directory itself, which is called `moearc-install.XXXXXX`, so the whole tree landed one level
  too deep. Neither is visible by inspection; both are obvious the first time it runs.
- ⬜ The TLS trust-root question above is untouched: `hf-hub` downloads still need the host's CA
  bundle. The runtime fetcher has the same dependency, and the clean-room image installs
  `ca-certificates` for that reason.

## Standing rules

- **The user's machine never compiles C++.** If a step needs a toolchain, it happens on ours.
- **A build-time dependency on our machine is not a user-facing dependency.** Confusing the two
  rejects good options for bad reasons.
- **The kernel GPU driver stays the one exception.** It ships with the kernel and cannot be
  vendored, so it is the only thing we may ask for — named exactly, with nothing beside it.
- **Claims about linkage get verified against the resolved graph**, never against a manifest.
  `cargo tree -i <crate>` and `ldd` on the built binary. The correction above is exactly what
  reading a manifest gets you.

- **A launcher is not a defeat.** `LD_LIBRARY_PATH` is the only search path a `dlopen`ed
  module’s dependencies inherit. Where that is the failing edge, a process-wide path is the
  correct mechanism and an rpath is not an option that was skipped.
- **Publish nothing that has not run where the toolkit cannot be reached.**
  `packaging/verify-clean.sh` is the gate, and it is a container rather than a test because the
  bug it exists to catch is *the environment*.
