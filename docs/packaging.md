# Packaging: how one binary brings its own dependencies

`ux.md` promises a single binary that installs in one step and never hands the user a list of
things to go install. That promise has to survive contact with three native dependencies. This
records how, and what is actually decided versus still open.

🔴 **Read this before the rest — most of this document is history now, and each section says
which part of it is.** It was written while MoEArc's engine was our own SYCL kernel object, and
it is the record of packaging *that*. The engine is llama.cpp now. `packaging/bundle.sh` was
rewritten on 2026-09-08 and the payload is **exactly one binary**:

```sh
declare -a payload=(
    "moearc:$rel/moearc"
)
```

`moearc-server`, `moearc-bench` and `moearc-selftest` all left that list — each for its own
reason, which `bundle.sh` writes out at the point it stopped shipping them — and
`libmoearc_kernels.so` left with them. The assertion that used to demand the kernel object be
present has been **kept and turned round**: staging any binary that still links it now *fails
the build*, because a stale `target/release` from before the pivot is exactly how a retired
engine reaches a release unnoticed.

The sections below are kept rather than deleted, for two reasons. The findings about *this
hardware* — the driver floor, the adapter set, the licence position — did not become false when
our engine changed; and the reasoning behind a retired decision is the only thing that stops the
next person rediscovering it the expensive way. Every section describing something that is no
longer shipped now opens with a status line. What is true today, collected in one place, is
**What is still not true** at the bottom.

Technique reference: [Vendoring C/C++ dependencies in
Rust](https://blog.veeso.dev/blog/en/vendoring-c-cpp-dependencies-in-rust/), whose `-src` /
`-sys` / public three-crate split and `include_bytes!` + `libloading` fallback are the two
patterns we lean on.

## The three problems, which are not the same problem

| | what it is | can it be statically linked? | approach |
| --- | --- | --- | --- |
| **SYCL kernels** — 💀 *retired 2026-09-08* | our own C++/SYCL, compiled with DPC++ | no — needs the SYCL runtime | was: build on our machine, ship the `.so` beside the binary. **No longer in the payload at all** |
| **Level Zero loader** | `libze_loader.so`, the ICD loader | no by design — it *is* a loader | embed, extract, `dlopen` |
| **TLS trust roots** | CA certificates for Hub downloads | n/a — data, not code | see below |

### SYCL kernels — build here, ship the artifact

💀 **Retired. Nothing in this subsection is shipped any more**, and it is kept because the
three-crate split and the `icpx`-does-the-link finding below are what any future native
dependency here will have to repeat. `moearc-kernels{,-src,-sys}` are still in the tree;
`packaging/bundle.sh` builds none of them (`--build` runs `cargo build --release -p moearc-cli`
with **no feature flags** — every flag that used to be there pulled this engine in) and refuses
to stage anything that links the resulting object.

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

💀 **Moot since 2026-09-08: no shipped binary links the kernels.** Two things outlived it and
are worth carrying forward. The first is the finding itself — `cargo:rustc-link-arg` does not
propagate downstream, and a suite can be entirely green while covering none of it. The second
is concrete and still executed: `bundle.sh` still asserts that **no `DT_NEEDED` in any staged
binary contains a slash**. That guard was written for this object and it now guards the payload
against a class of mistake rather than against one file, which is why it was kept when the
step that motivated it was dropped.

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

⚠️ **Half of the transcript below can no longer be re-run.** `moearc-selftest` is not in the
tarball — it existed only to `dlopen` the retired kernel object — so the second command is a
record, not a check anybody can repeat today. The *first* half is still exercised on every
release: `packaging/verify-clean.sh` runs `moearc --no-tui` in this same container, once
normally and once under `env -i`, and fails if the Arc card is not named. What it can no longer
do is prove that anything **computes** there; it prints `NOT PROVEN` rather than a clean PASS
that covers less than it used to.

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

🔴 **Nothing in today's payload links any of these, and that is not the same as saying they are
unnecessary.** `moearc` reaches Level Zero through `moearc-device`'s `dlopen` and links no SYCL
at all — which is precisely the property that lets the tarball work on first unpack, before
anything has been fetched. `launcher.sh` encodes it directly: `case $self in moearc)
needs_runtime=0`, so the shipped binary never triggers the fetch. What still needs this set is
the thing that computes, and that is now **llama.cpp's SYCL build, which the user supplies**.
The launcher still prepends `runtime/` to `LD_LIBRARY_PATH`, and `serve` spawns `llama-server`
without clearing the environment, so a fetched runtime *is* inherited by the engine — read out
of `launcher.sh` and `serve.rs`, **not tested**, and version-coupling to somebody else's build
of llama.cpp is exactly the question this document warns about two sections down.

Eleven libraries, 78 MiB installed. Discovered by `ldd` on our own object *(as it then was)* and
on each adapter, plus the adapters themselves, which nothing links and `SYCL_UR_TRACE=1` names:

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
  🔴 **The premise of that bullet is gone and the conclusion has not been re-derived.** There is
  no `libmoearc_kernels.so` any more, so the runtime is no longer coupled to a compiler *we*
  ran; it is coupled to whatever compiled the user's `llama-server`. Winning over a system
  oneAPI was the right answer when the object it had to match was ours. Whether a pinned runtime
  should still go first when the binary that consumes it is somebody else's build is an **open
  question nobody has measured**, and it is recorded as open rather than answered by leaving the
  old sentence standing.
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

⚠️ **The finding stands; the instrument that produced it does not ship, and nothing has
replaced it.** The third column below came from `moearc-bench` loading a model in the container,
and `moearc-bench` left the payload with the retired engine. `packaging/verify-clean.sh` says so
in its own verdict — a PASS there now means *the card is found*, not that it computes — so the
middle row of this table, the stack that passes every detection check and then cannot load a
model, is **currently unguarded by the release gate**. It comes back the day llama.cpp is
bundled and a forward pass can be run in the clean room again.

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
have no open-source counterpart, and they are in the `DT_NEEDED` of *Intel's own* Level Zero
adapter — so they cannot be dropped. ✅ **Half of that sentence has expired and the conclusion
survives it.** They were in the `DT_NEEDED` of our kernel object *as well*, and that object is
no longer built or shipped; the adapter alone is enough to require them. What changed is who
does the requiring — today it is a llama.cpp the user installed, not anything in this tarball. Intel's EULA grants redistribution
of "Redistributables", defined as the files listed in a `redist.txt` — **and no such file exists
anywhere in the oneAPI 2026.1 installation we build against.** The grant is real and we cannot
show it covers any particular file. Two of its conditions would also propagate to our users:
a no-reverse-engineering clause, and a prohibition on SaaS use — which is one of the things an
inference server is for.

So the default tarball contains **no third-party binaries at all** — and since the payload
became one binary it contains no third-party *code* at all either, which is a stronger and
simpler statement than the one this section was written to defend. `packaging/fetch-runtime.py`
downloads Intel's runtime, on the user's machine, from Intel's own channel — packages Intel
publishes precisely so that "executables can be deployed to hosts without the oneAPI
development toolkits" — pinned by SHA-256 in `packaging/runtime.lock.json`. The user accepts
Intel's terms from Intel. The tarball stays Apache-2.0. An honest dependency beats a licence
violation.

🔴 **Since 2026-09-09 that download does not happen unless it is asked for.** `install.sh` ran
it on every install, and stopped: nothing in the payload opens the directory it produces. It is
`MOEARC_FETCH_RUNTIME=1` now, and `launcher.sh` keeps its lazy fetch for the day a
`needs_runtime=1` name is back in the bundle. The licence position above is unchanged by that
and is what makes the opt-in safe to keep offering — what changed is that a default install no
longer spends somebody's bandwidth on a runtime nothing in the bundle loads.

`bundle.sh --with-runtime` vendors it anyway, for air-gapped installs, verified under
`podman --network none` with `MOEARC_NO_FETCH=1`. That archive is **not** Apache-2.0 and must
not be published as though it were.

⚠️ **Two tarball sizes used to be quoted here — 29 MB vendored against 4.8 MB default — and
both were wrong from the day the payload became one binary.** They were measured when it was
four executables and a shared object. Neither is restated here: the default has since been
re-measured and `packaging/RELEASE.md` carries that figure beside the artefact it describes,
which is the right place for a number that changes with every build; the vendored figure has
**not** been re-taken and is simply withdrawn. `bundle.sh` prints the size of what it just
produced, and that is the only one true by construction. `bench/PROTOCOL.md` §9 is the rule
underneath both halves — when a figure stops describing what ran, withdraw it; a replacement
nobody measured is the worse error.

## Installing without a network

The runtime fetch used to be the one step that needed the internet, and it is now opt-in
(`MOEARC_FETCH_RUNTIME=1`) precisely because no binary in today's payload consumes what it
fetches — `moearc` runs on a machine that has never seen either. What follows therefore matters
for staging a runtime a *user's own* `llama-server` will resolve against, which the launcher
makes possible by putting `runtime/` on `LD_LIBRARY_PATH` that the child inherits. On a machine
with no network:

```sh
# on a machine that does, with the same tarball unpacked:
python3 libexec/fetch-runtime.py --dest ./runtime --lock share/moearc/runtime.lock.json
# then copy ./runtime across, or set MOEARC_RUNTIME_DIR at it
```

`MOEARC_NO_FETCH=1` makes the launcher refuse to download and say so, rather than hanging on a
firewalled resolver.

## The kernel object is relocatable now, without `patchelf`

💀 **`packaging/elf-relocatable.py` is no longer run.** It shortened exactly one absolute
`DT_SONAME` — the kernel object's — and no staged file has one, so `bundle.sh` dropped the step
and says so in its header. The script stays in the tree; the technique is the interesting part
and it is written down below. **The guard it existed to satisfy was kept**: any `DT_NEEDED` with
a slash in it still fails the build. That is deliberate — the step was the mechanism, the
assertion is the requirement, and a requirement should outlive the mechanism that happened to
satisfy it.

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

**Rewritten 2026-09-09**, against `packaging/bundle.sh`, `packaging/launcher.sh` and
`crates/moearc-cli/src/serve.rs` as they stand. Two entries here had gone false in the direction
that flatters nobody — they understated the product and overstated the payload — and they are
corrected in place rather than deleted, because a limitations list that quietly loses a wrong
entry teaches the next reader nothing.

- 🔴 **glibc 2.39 or newer.** `moearc` requires `GLIBC_2.39`, inherited from building on Ubuntu
  26.04. That is Ubuntu 24.04, Fedora 40, Debian 13 or newer — and it silently excludes Debian 12
  and RHEL 9. `share/moearc/BUILD-INFO.txt` in every tarball states the measured floor, computed
  from the shipped binary rather than asserted.
  ✅ *Corrected:* this used to read "…the other three binaries need only 2.34", which made the
  tarball's floor sound like a property of one component among several. There are no other three
  binaries. The floor of the payload is the floor of `moearc`, and nothing softens it.
- 🔴 **x86-64 Linux only.** No aarch64, no Windows.
- 🔴 **The payload is one binary, and it is not an inference engine.** `moearc serve` supervises
  llama.cpp's own `llama-server` as a child process with the argv the tuning resolver produced,
  so **a machine that unpacks this tarball still needs a llama.cpp with the SYCL backend on it.**
  `serve` looks for one beside its own executable first, then `$MOEARC_LLAMA_SERVER`, then the
  usual build directories, and `$PATH` last — deliberately last, because a `llama-server` earlier
  in someone's `PATH` may be a Vulkan build that runs, answers correctly and is **4.8× slower**,
  with nothing in the file name to say so.
  ⬜ Bundling llama.cpp's `llama-server`, `libllama.so.0` and the `libggml*.so.0` family would
  close this. It is real work with a real licence obligation — `packaging/THIRD-PARTY.md` does
  not yet cover redistributing MIT-licensed llama.cpp binaries — and it is tracked in
  `docs/pivot-inventory.md`.
- ✅ **`moearc serve` is not a fixture, and the entry that said so was the stalest line in this
  document.** It read: *"`moearc serve` is still a fixture … the tarball ships `moearc-server`
  (real, links the kernels) and `moearc-bench` beside it."* Every clause of that is now false.
  `serve` starts the engine, pins the card by name and confirms the pin against the child's own
  device block *before* 59 GiB is read, and it was checked by running 64 tokens through it
  against the same argv hand-launched with no `moearc` in the picture — identical token ids. And
  the tarball ships none of those binaries: `moearc-bench` and `moearc-selftest` existed
  only to exercise the retired SYCL engine, and `moearc-server` was dropped because **without
  llama.cpp linked in it is an echo stub** that answers OpenAI-format requests with the prompt
  read back, over the real routing and SSE path — indistinguishable from a model to anyone not
  reading `/health`. The right thing to have in the tree, the wrong thing to have in a release.
  📌 The general lesson is the one this file keeps relearning: **the payload is a list, and a
  list that shrinks silently is how a release ships something nobody meant to ship.** `bundle.sh`
  now writes out, at the point of the list, why each departed name departed.
- ⬜ **The serving *screen* is still a fixture, which is the true half of the old entry.**
  `source.rs`'s `ServeStats` and the TUI's serving view still draw from fabricated numbers;
  `kv_utilisation` and `expert_hit_rate` have no llama.cpp source at all. `moearc serve <model>`
  on the command line is the wired path — `docs/serve.md` §5 is the standing list.
- ⬜ **The clean-room gate proves less than it did, and prints `NOT PROVEN` rather than absorbing
  it.** Three of its four gates ran binaries that have left the payload, so
  `packaging/verify-clean.sh` can no longer show that a model loads or that tokens come out
  right on a machine with no oneAPI. It still shows the thing it was written for — that the Arc
  card is found from an empty environment on a distro that has never had the toolkit. A PASS
  today means **the card is found, not that it computes.**
- ⬜ **Nothing in the payload consumes the Intel SYCL runtime, and as of 2026-09-09 a default
  install no longer downloads one.** `moearc` links no SYCL, and `launcher.sh` says so by name:
  `case $self in moearc) needs_runtime=0`. The three names that set `needs_runtime=1` all left
  with the retired engine, so the fetch was serving a payload that no longer exists — it is
  `MOEARC_FETCH_RUNTIME=1` now, and `launcher.sh`'s lazy fetch still fires by itself the day a
  `needs_runtime=1` name comes back. **The machinery was made opt-in rather than deleted**, and
  the distinction is the point: the question of what `runtime/` is for gets settled when
  llama.cpp's shared objects are bundled, not by ripping it out first.
  ⬜ There is one real use for the opt-in and it is **untested**: a user whose own llama.cpp
  SYCL build has no oneAPI beside it can resolve against ours, because the launcher puts
  `runtime/` on `LD_LIBRARY_PATH` and the supervised `llama-server` inherits it. Nobody has run
  that across an ABI skew. It is a side effect that happens to work, not a design — which is
  exactly why it is a flag somebody chooses rather than a download everybody pays for.
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
  bug it exists to catch is *the environment*. 🔴 **A gate that loses coverage says so in its own
  verdict.** When three of its four checks left with the binaries they ran, the answer was to
  print `NOT PROVEN` beside the PASS, not to narrow the definition of passing until the script
  looked healthy again.
- **A payload is a list, and it is written down with its reasons.** Every name that leaves it
  leaves for a stated reason, at the point in `bundle.sh` where it stopped being staged. A build
  that silently produces a smaller tarball is indistinguishable from one that produces the right
  one.
