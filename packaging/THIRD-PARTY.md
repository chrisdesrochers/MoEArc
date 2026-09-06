# Third-party software MoEArc runs against

MoEArc itself is Apache-2.0 (`LICENSE`, `NOTICE`). This file is about the things it needs at
run time that it did not write, what they are licensed under, and — the part that decided the
shape of the packaging — **which of them MoEArc does and does not redistribute.**

## What is in the tarball

Everything under `libexec/` is MoEArc's own work, Apache-2.0:

| file | what |
| --- | --- |
| `moearc`, `moearc-server`, `moearc-bench`, `moearc-selftest` | Rust binaries |
| `libmoearc_kernels.so` | our SYCL kernels, compiled with Intel DPC++ on the build machine |

`libmoearc_kernels.so` is compiled *by* Intel's compiler and links its runtime, but it contains
no Intel source. The Rust dependency tree is Apache-2.0/MIT throughout; `cargo tree` is the
authority and `NOTICE` carries the attributions.

**No third-party binary ships in the default tarball.**

## What is fetched at install time, and why it is fetched rather than shipped

`runtime/` is populated by `packaging/fetch-runtime.py` from Intel's own published packages,
pinned by version and SHA-256 in `packaging/runtime.lock.json`.

| library | project | licence | redistributable by us? |
| --- | --- | --- | --- |
| `libsycl.so.9` | [intel/llvm](https://github.com/intel/llvm) | Apache-2.0 WITH LLVM-exception | source licence says yes |
| `libur_loader.so.0`, `libur_adapter_level_zero{,_v2}.so.0`, `libur_adapter_opencl.so.0` | intel/llvm (Unified Runtime) | Apache-2.0 WITH LLVM-exception | source licence says yes |
| `libumf.so.1` | [oneapi-src/unified-memory-framework](https://github.com/oneapi-src/unified-memory-framework) | Apache-2.0 WITH LLVM-exception | yes |
| `libhwloc.so.15` | [open-mpi/hwloc](https://github.com/open-mpi/hwloc), shipped in Intel's `tcmlib` | BSD-3-Clause (package: Intel Simplified Software License) | yes, with notice |
| `libimf.so`, `libsvml.so`, `libintlc.so.5`, `libirng.so` | Intel compiler runtime | **proprietary** — Intel End User License Agreement for Developer Tools | 🔴 **not established** |

### 🔴 The licence position, stated plainly

The last row is why nothing is vendored.

`libimf`, `libsvml`, `libintlc` and `libirng` are Intel's C/C++ math and compiler runtime.
They have no open-source counterpart, and both our kernel object *and Intel's own Level Zero
adapter* have them in `DT_NEEDED`, so they cannot be dropped. Intel's EULA does grant a
redistribution right — but only for "**Redistributables**", which §1.I defines as *"the files
(if any) listed in the `redist.txt`, `redist-rt.txt` or similarly-named text files that may be
included in the Materials."*

**There is no such file anywhere in the oneAPI 2026.1 installation we build against.** We
looked. So the grant exists and we cannot show that it covers any particular file, which is
not the same as being told no, and is not good enough to publish binaries on.

Two further terms would follow us downstream even if it did apply, and both sit badly with an
Apache-2.0 project: §2.1.D(2) requires that redistributed Intel executables travel *"subject to
a license agreement that prohibits reverse engineering, decompiling or disassembly"*, and
§3.1(xi) forbids using the Materials *"directly or indirectly for SaaS services or service
bureau purposes."* Shipping those files inside our tarball would quietly impose both on
everyone who downloads MoEArc — including a restriction on running it as a service, which is
one of the things an inference server is for.

**So MoEArc does not redistribute them.** It downloads them, on the user's machine, from
Intel's own channel, into a directory that carries Intel's EULA text beside them. Intel
publishes these packages for exactly this purpose — their description is *"shared common
libraries required to deploy executables on systems without the Intel oneAPI development
toolkits installed"* — and it is the same arrangement PyTorch's XPU builds use. The user
accepts Intel's terms from Intel, as they would have if they had installed the runtime
themselves, and MoEArc's tarball stays wholly Apache-2.0.

The cost is honest and small: an install-time download, and a machine with no network needs
one manual step (see `docs/packaging.md`, *Installing without a network*). An honest
dependency beats a licence violation.

### The escape hatch, and its condition

`packaging/bundle.sh --with-runtime` vendors the runtime into the tarball, for air-gapped or
enterprise installs. **The resulting archive is not wholly Apache-2.0 and must not be published
as if it were.** Anyone building or distributing one is accepting Intel's terms on their own
account, not on ours; read the EULA that lands in `runtime/` first.

## What must already be on the machine

Not shipped, not fetched, and named exactly — `docs/ux.md` allows one dependency and this is
it, in two halves that live in the same package on every distribution:

| | what | licence |
| --- | --- | --- |
| kernel | `xe` (Arc B-series / Battlemage) or `i915` (A-series) | GPL-2.0, ships with your kernel |
| user space | `libze_loader.so.1` + `libze_intel_gpu.so.1` — the Level Zero loader and the Intel Graphics Compute Runtime | MIT ([oneapi-src/level-zero](https://github.com/oneapi-src/level-zero), [intel/compute-runtime](https://github.com/intel/compute-runtime)) |

Both are MIT and could in principle be bundled. They are not, on purpose: they are the half of
the stack that has to agree with the *kernel* on this machine, and a distribution ships them
together for that reason. Pinning our own copy would mean overriding a version the user's
distribution chose to match their kernel, which is the one place where "we know better" is
most likely to be wrong.

🔴 **It has a floor, and it bites.** Ubuntu 24.04's own `libze-intel-gpu1` is Level Zero build
27642 and predates Battlemage: on a box with a B580 it enumerates the Arrow Lake iGPU and
nothing else. `docs/packaging.md` records what that looks like and what to install instead.
