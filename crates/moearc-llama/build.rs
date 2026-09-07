//! Build the MoEArc <-> llama.cpp shim and link it against a **pre-built** llama.cpp.
//!
//! # Why we link a pre-built llama.cpp instead of building one
//!
//! MoEArc already has a llama.cpp SYCL build, produced by
//! `/zfs/swift/projects/build-llamacpp-sycl.sh`, pinned to an exact commit in
//! `llama.cpp-COMMIT`, and that pin is load-bearing for the benchmark protocol:
//! llama.cpp is the number MoEArc is measured against, and a reference that moves
//! on every rebuild is not a reference. Building a *second*, differently-pinned
//! copy of llama.cpp inside `target/` would mean the engine and the baseline are
//! no longer the same code -- the exact class of error that has already cost this
//! project three retracted results.
//!
//! So this crate consumes the reference build rather than duplicating it.
//!
//! # Why this does not hit the `_intel_fast_memcpy` wall
//!
//! `crates/moearc-kernels/build.rs` documents the failure: a static `.a` linked by
//! `cc` died on `_intel_fast_memcpy` from `libintlc`, because `icpx` links a set of
//! Intel runtime libraries that belongs to the compiler version, not to our code.
//!
//! That failure mode is specific to **static** linking. llama.cpp's CMake build
//! produces **shared** objects, and `libllama.so` already records `libsvml`,
//! `libirng`, `libimf` and `libintlc.so.5` in its own `DT_NEEDED`, with a
//! `DT_RUNPATH` covering the oneAPI directories. The Intel runtime is therefore
//! resolved by the dynamic loader on `libllama.so`'s behalf and never becomes our
//! link problem. A consequence worth stating plainly: **this shim compiles with a
//! plain `g++` and needs no oneAPI toolchain at all.** `icpx` is required to build
//! llama.cpp, once, out of tree -- not to build MoEArc.
//!
//! # How a downstream binary finds the shim
//!
//! Same mechanism as `moearc-kernels`, for the same reason, and the reasoning
//! there is the canonical write-up: `cargo:rustc-link-arg` does not propagate to
//! downstream crates, so an rpath emitted here would cover this crate's own tests
//! and nothing else. The path has to travel inside something that *does*
//! propagate, and that is the shared object itself -- `ld` copies `DT_SONAME`
//! verbatim into every consumer's `DT_NEEDED`, and glibc treats a `DT_NEEDED`
//! string containing a slash as a path. So the soname is set to the object's
//! absolute location, and the object carries its own `DT_RUNPATH` to llama.cpp's
//! `build/bin` so that `libllama.so.0` and the `libggml-*.so.0` family resolve for
//! every consumer with no cooperation from any of them.
//!
//! As with `moearc-kernels`, this makes the artifact **non-relocatable** -- it is a
//! development build. Packaging is a separate path (`packaging/`).

fn main() {
    println!("cargo:rerun-if-changed=shim/moearc_llama_shim.cpp");
    println!("cargo:rerun-if-env-changed=MOEARC_LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=MOEARC_LLAMA_CPP_BUILD");
    println!("cargo:rerun-if-env-changed=CXX");

    // The whole FFI surface is behind the `runtime` feature so that a default
    // `cargo test --workspace` needs neither llama.cpp nor a GPU. Cargo compiles
    // build scripts with the crate's feature cfgs, so this gate is resolved at
    // build-script compile time.
    runtime::build();
}

#[cfg(not(feature = "runtime"))]
mod runtime {
    pub fn build() {}
}

#[cfg(feature = "runtime")]
mod runtime {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub fn build() {
        let src = locate_source();
        let lib = locate_build(&src);

        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let obj = out.join("libmoearc_llama_shim.so");

        let cxx = std::env::var("CXX").unwrap_or_else(|_| "g++".into());

        let status = Command::new(&cxx)
            .args(["-std=c++17", "-O2", "-fPIC", "-shared", "-Wall"])
            .arg("shim/moearc_llama_shim.cpp")
            .arg("-o")
            .arg(&obj)
            .arg(format!("-I{}", src.join("include").display()))
            .arg(format!("-I{}", src.join("ggml/include").display()))
            .arg(format!("-L{}", lib.display()))
            .args(["-lllama", "-lggml-base"])
            // Resolve libllama.so.0 + the libggml-*.so.0 family for every consumer.
            .arg(format!("-Wl,-rpath,{}", lib.display()))
            // See the module docs: the soname is the propagation channel.
            .arg(format!("-Wl,-soname,{}", obj.display()))
            .status()
            .unwrap_or_else(|e| panic!("failed to run C++ compiler `{cxx}`: {e}"));

        assert!(status.success(), "compiling moearc_llama_shim.cpp failed ({status})");

        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=dylib=moearc_llama_shim");
        println!("cargo:llama_cpp_dir={}", src.display());
        println!("cargo:llama_cpp_build={}", lib.display());
    }

    /// Find the llama.cpp source checkout (for its headers).
    fn locate_source() -> PathBuf {
        if let Some(dir) = std::env::var_os("MOEARC_LLAMA_CPP_DIR") {
            let p = PathBuf::from(dir);
            assert!(
                p.join("include/llama.h").is_file(),
                "MOEARC_LLAMA_CPP_DIR={} does not contain include/llama.h",
                p.display()
            );
            return p;
        }

        for cand in ["/zfs/swift/projects/llama.cpp", "../../third_party/llama.cpp"] {
            let p = PathBuf::from(cand);
            if p.join("include/llama.h").is_file() {
                return p;
            }
        }

        panic!(
            "could not find a llama.cpp checkout. MoEArc links the *reference* llama.cpp \
             build rather than compiling its own, so that the engine and the benchmark \
             baseline are provably the same commit. Set MOEARC_LLAMA_CPP_DIR to a checkout \
             whose build/ was produced by build-llamacpp-sycl.sh."
        );
    }

    /// Find the directory holding the built shared objects.
    ///
    /// 🔴 This must never be resolved by globbing. A sibling `build-vulkan/` exists
    /// on the reference machine and is 4.8x slower; selecting a backend by whichever
    /// directory a glob happened to yield first is precisely how a benchmark ends up
    /// measuring something other than what it reports. The default is the single
    /// explicit path `build/bin`, and the presence of `libggml-sycl.so` is asserted
    /// rather than assumed -- a build directory that is not the SYCL build is a hard
    /// error here, not a silent fallback to CPU.
    fn locate_build(src: &Path) -> PathBuf {
        let dir = match std::env::var_os("MOEARC_LLAMA_CPP_BUILD") {
            Some(d) => PathBuf::from(d),
            None => src.join("build/bin"),
        };

        assert!(
            dir.join("libllama.so").is_file(),
            "no libllama.so in {} -- llama.cpp has not been built there. \
             Run build-llamacpp-sycl.sh, or point MOEARC_LLAMA_CPP_BUILD at the build output.",
            dir.display()
        );

        assert!(
            dir.join("libggml-sycl.so").is_file(),
            "{} contains libllama.so but no libggml-sycl.so, so it is not a SYCL build. \
             MoEArc targets Intel Arc via SYCL; linking a CPU-only or Vulkan llama.cpp here \
             would produce a binary that runs and silently reports the wrong backend.",
            dir.display()
        );

        dir
    }
}
