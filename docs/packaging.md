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

⚠️ The guide's warning about extracting object files from `make` output applies directly:
DPC++ emits a shared object, and building a `.a` from it means driving `ar` over the objects by
hand. The author's verdict on parsing build output for those paths — *"many times it won't
work"* — is worth believing before spending a day on it. The `include_bytes!` route below may
simply be better.

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

## Standing rules

- **The user's machine never compiles C++.** If a step needs a toolchain, it happens on ours.
- **A build-time dependency on our machine is not a user-facing dependency.** Confusing the two
  rejects good options for bad reasons.
- **The kernel GPU driver stays the one exception.** It ships with the kernel and cannot be
  vendored, so it is the only thing we may ask for — named exactly, with nothing beside it.
- **Claims about linkage get verified against the resolved graph**, never against a manifest.
  `cargo tree -i <crate>` and `ldd` on the built binary. The correction above is exactly what
  reading a manifest gets you.
