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

## Evaluated, not adopted

### `oneapi-rs` — official Rust SYCL bindings · Apache-2.0/MIT · **watching**

The closest thing to what MoEArc's kernel seam eventually needs, from the `oneapi-src`
organisation. Not adopted **yet**, on maturity grounds only: it is explicitly pre-0.1.0 and
states its API may change without notice.

Worth correcting one objection that looks decisive and is not: it requires the oneAPI toolkit
to build. That cost lands on *our* build machine, not the user's — we compile SYCL kernels
with DPC++ and ship the resulting shared library either way. So the toolkit requirement does
not conflict with the one-binary rule. **Re-evaluate at 0.1.0.**

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
