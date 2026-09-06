# Dependency evaluations

Why MoEArc does or does not use a given crate. Kept because "why didn't you just use X?" is
the most common question an open-source project gets, and answering it once in writing is
cheaper than answering it repeatedly from memory.

Being listed as *not used* is not a criticism of the project. Most entries here are good
software that solves a different problem than ours.

## Adopted

| Crate | Role | Why |
| --- | --- | --- |
| `ratatui` + `crossterm` | TUI | The de facto Rust TUI; `gitui`, `atuin`, `yazi` and `bottom` build on it. Expresses the Charm/lipgloss aesthetic we want. See `ux.md`. |
| `clap` | CLI | Standard, derive API, and good `--help` output — which is part of the product, not decoration. |
| `libloading` | Level Zero at runtime | Lets the binary `dlopen` `libze_loader` instead of linking it, so it still starts and can explain itself on a machine with no Level Zero runtime. Directly serves the "brings its own dependencies" rule. |
| `serde` / `serde_json` | Config, `--json` output, test fixtures | Unavoidable and universal. |
| `sysinfo` | Host RAM and free space | Read for the host memory budget: `MemAvailable`, fitted RAM, and free space on the filesystem the models live on. **Already in the CLI's dependency graph** — `hf-hub` reaches it through `hf-xet` → `xet-runtime` — so naming it as a direct dependency adds a manifest line and no crates. Only `system` and `disk` are enabled. The alternative was reading `/proc/meminfo` and calling `statvfs` ourselves, which is ~30 lines and Linux-only; `sysinfo` is already paid for and is not. 📌 It is a **CLI** dependency, never an engine one: `moearc-engine` has no dependencies at all, which is what keeps the planners testable on a machine with no card and no oneAPI. The engine decides; the CLI measures. |

### `hf-hub` — Hugging Face's own Hub client · adopted, with a cost worth stating

Adopted for `moearc pull`. It knows the Hub's repo/revision/blob semantics, and reimplementing
those correctly is not a weekend.

**It costs 213 transitive crates, up from 2.** `reqwest`, `tokio`, `hf-xet` and `rustls` come
with it, and that is by a wide margin the largest dependency footprint in this workspace. Worth
it for Hub semantics; not worth it for anything we could write ourselves.

Two things had to be worked around rather than used, both found by running it:

- **It does not resume.** `download_file` opens its destination with `File::create`, which
  truncates, and issues a plain `GET` with no `Range` header. A 20 GiB interruption would
  restart from zero. We use `download_file_stream` and own the file handling.
- **`get_file_metadata` fails on every real GGUF repo**: it HEADs the resolve URL, the HEAD
  redirects to the CDN, and the CDN response carries no `X-Repo-Commit`, which `hf-hub` treats
  as malformed. We use `paths-info` instead, which also returns the true blob size rather than
  the ~134-byte LFS *pointer* length.

✅ **Correction (verified against the resolved graph).** This entry previously said
`native-tls` "stays enabled underneath", so a static binary could not be claimed. That was
reasoned from `hf-hub`'s manifest rather than from what cargo actually resolves, and it is
wrong. `openssl-sys` and `native-tls` are **zero nodes workspace-wide**; the only `openssl`
match is `openssl-probe`, a pure-Rust crate reached via `rustls-native-certs →
rustls-platform-verifier` that *locates* the OS trust store and links nothing. The built binary
dynamically links only `libc`, `libm`, `libgcc_s` and the dynamic loader.

The real residual is narrower: **we depend on the host having a CA certificate bundle**, so a
minimal container without one fails TLS at download time. `hf-hub` 1.0.0 offers no
`webpki-roots` feature. See `packaging.md`.

📌 The lesson is the reusable part: **a manifest states what a crate can enable, not what your
build resolved.** Check `cargo tree -i` and `ldd`, not `Cargo.toml`.

## Evaluated, not adopted

### `oneapi-rs` — official Rust SYCL bindings · Apache-2.0/MIT · **watching**

The closest thing to what MoEArc's kernel seam eventually needs, from the `oneapi-src`
organisation. Not adopted **yet**, on maturity grounds only: it is explicitly pre-0.1.0 and
states its API may change without notice.

Worth correcting one objection that looks decisive and is not: it requires the oneAPI toolkit
to build. That cost lands on *our* build machine, not the user's — we compile SYCL kernels
with DPC++ and ship the resulting shared library either way. So the toolkit requirement does
not conflict with the one-binary rule. **Re-evaluate at 0.1.0.**

### `RLX` — Rust ML compiler and runtime · Apache-2.0/MIT · **reference, and a real one**

The most relevant project surveyed so far, and the claims were checked rather than taken from
the README — this project has been burned once already by a repository whose advertised SYCL
kernels existed only in its documentation.

What is actually there: a genuine history (30 distinct commit days, May–Aug 2026, not a code
dump), **712k lines of Rust**, and near-zero stub density (0 `todo!()`, 2 `unimplemented!()`
across the whole tree). It is real software. It also contains our exact problem domain —
`hot_expert_cache.rs`, `expert_pool.rs`, `moe_expert_store.rs`, `moe_residency.rs`,
`moe_split.rs` — which makes it worth **reading** before we design expert residency.

Why it is not a foundation for MoEArc, all measured:

- **The Intel backend is its weakest limb**: 4,893 lines of Rust against the CUDA backend's
  38,589, roughly an eight-to-one gap.
- **Its oneAPI kernels are OpenCL C** (53 `.cl` files), not SYCL.
- **Zero references to XMX, DPAS or `joint_matrix`** anywhere in the oneAPI backend — so no
  matrix-engine path, which is where Battlemage's throughput actually lives.
- **The MoE hot/cold parity tests exist for CUDA and ROCm, not for oneAPI.** The expert
  machinery that interests us most is the part unvalidated on Intel.

A general-purpose compiler with a thin Intel backend is the opposite of what this project is:
an Intel-first engine. Read it for design, build our own.

### `llama-cpp-rs` — Rust bindings to llama.cpp · Apache-2.0/MIT · **for benchmarking**

Actively maintained (1,812 commits) with an explicit goal of tracking llama.cpp closely.

**Not for the engine** — building on it would make MoEArc a llama.cpp wrapper, which is not the
project. **Useful for the benchmark harness**: comparison (1) in `bench/README.md` is MoEArc
against llama.cpp SYCL on the same card, and driving that in-process from Rust gives cleaner
measurement than shelling out to `llama-bench` and parsing tables.

⬜ **Unverified**: only `cuda` and `metal` cargo features are documented. llama.cpp's SYCL
backend is a CMake option, so it may be reachable through the same build path — but that is an
assumption, not a finding, and must be tested before the harness depends on it.

### `gpu-probe` — cross-platform VRAM detection · Apache-2.0 · **no**

Its Intel path is a **static PCI device-ID table** rather than a runtime query, and its own
documentation says the Intel path is *not yet confirmed on real hardware*. A table cannot
report VRAM for a card it does not recognise, which is a live risk for a GPU as new as
Battlemage. It also does not distinguish **discrete from integrated** — for us the single
highest-risk field, since a development box with both a discrete Arc card and an Arrow Lake
iGPU will happily benchmark the wrong one (see `bench/README.md`).

It did surface a real gap, which we now fill ourselves: Level Zero tells us what is *usable*,
sysfs/PCI tells us what is *present*, and the difference between them is the error message
worth printing — *"an Arc card is installed but Level Zero cannot see it; the `xe` driver is
not loaded"* — instead of a bare "no devices found."

### `all-smi` — multi-vendor GPU/NPU monitor · Apache-2.0 · **reference**

Has a working Intel Level Zero **Sysman** backend that dynamically loads `libze_loader`, which
is the exact architecture we chose independently. Not taken as a dependency — it is a
monitoring application, and we need a few `zes_*` calls rather than a fleet telemetry stack.
Valuable as a correctness reference for FFI signatures and struct layouts, which is how it is
being used.

### `gpu-descriptor` — descriptor-set allocator · **no**

Solves descriptor *set* allocation for Vulkan-like APIs: binding resources to shaders. Not
memory allocation, not device enumeration, no Level Zero backend. Wrong layer and wrong API.
Its sibling `gpu-allocator` is a genuine VRAM sub-allocator but is likewise Vulkan/DX12-only.

### `decuda` — CUDA to HIP/SYCL/OpenCL translator · Apache-2.0 · **no**

Solves a problem MoEArc deliberately does not have. We are writing our own kernels rather than
translating anyone's CUDA. Also very early (4 commits, no releases) and self-described as "a
starting point for migration, not a finished translator." If mechanical CUDA→SYCL translation
were ever needed, Intel's own SYCLomatic is far more mature.

### `RGM` — GPU monitoring GUI · Apache-2.0/MIT · **no**

An egui desktop monitor for NVIDIA and AMD; Intel support is listed as planned, not
implemented. Unrelated to inference. Noted only because if we build Level Zero telemetry for
our serving-stats panel, that is precisely the gap they have — a possible upstream
contribution rather than a dependency.

## Standing rules

- **A dependency that forces work onto the user is not a shortcut.** Anything that would make
  a user install a toolkit, match a driver version, or read an error we could have explained is
  measured against `ux.md` first.
- **Prefer a small amount of our own code over a large dependency that nearly fits.** The
  Level Zero surface we need is a handful of entry points; a monitoring framework is not.
- **Record rejections with reasons.** A rejection with no reason gets re-litigated.
