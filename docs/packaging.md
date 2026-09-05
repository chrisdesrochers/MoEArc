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

🔴 Still open: whether the embedded loader is preferred over a system one, or only used when the
system has none. Preferring ours guarantees a known-good version and risks disagreeing with the
installed compute runtime it has to talk to. **Not yet decided; do not assume either.**

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

## 🔴 Open gap: the SYCL runtime still needs `setvars.sh` to find a GPU

Found while proving the above, and it is a *different* problem that the same clean-environment
run exposed. With no environment set, a MoEArc binary now starts, links and calls into the
kernels — and then reports **no usable GPU**, on a machine with a working Arc B580.

`libsycl` dlopens its Unified Runtime adapters, and `libur_adapter_level_zero.so.0` needs
`libumf.so.1` and `libhwloc.so.15`. Those live in `umf/…/lib` and `tcm/…/lib`, *different*
oneAPI component directories from the compiler's, which `setvars.sh` adds to
`LD_LIBRARY_PATH`. `SYCL_UR_TRACE=1` names them exactly.

⚠️ **No rpath on our object can fix this, and it was measured rather than assumed.** Both
`DT_RUNPATH` and `DT_RPATH` were tried on `libmoearc_kernels.so` with those directories added;
neither works. The failing lookup is a dependency of a `dlopen`d module several links away,
and the loader consults neither our runpath (not inherited) nor our rpath (the dlopened
adapter has no loader chain back to us).

So today MoEArc sees a GPU only if `setvars.sh` has been sourced — an environment dependency
that `ux.md` does not permit and that nothing had written down, because every previous test
run inherited it. It is the same shape as the `MOEARC_ZE_LOADER` question above and probably
has the same answer: what gets embedded and extracted is not one library but the runtime's
closure, and the extraction directory is a single place the process can point the loader at.

⬜ Open: decide whether the packaged runtime is preferred over a system one (see the Level Zero
loader note above — it is the same decision), and whether the extracted set is discovered at
package time by walking `DT_NEEDED` plus the adapters' dlopen list, or pinned by hand.

## Standing rules

- **The user's machine never compiles C++.** If a step needs a toolchain, it happens on ours.
- **A build-time dependency on our machine is not a user-facing dependency.** Confusing the two
  rejects good options for bad reasons.
- **The kernel GPU driver stays the one exception.** It ships with the kernel and cannot be
  vendored, so it is the only thing we may ask for — named exactly, with nothing beside it.
- **Claims about linkage get verified against the resolved graph**, never against a manifest.
  `cargo tree -i <crate>` and `ldd` on the built binary. The correction above is exactly what
  reading a manifest gets you.
