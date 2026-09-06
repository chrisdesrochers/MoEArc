#!/bin/sh
# MoEArc launcher. Installed at the bundle root under three names -- `moearc`,
# `moearc-server`, `moearc-bench` -- and dispatches on which one it was invoked as.
#
# Its whole job is `docs/packaging.md`'s open gap #2: libsycl dlopens its Unified Runtime
# adapters, and those adapters need libumf and libhwloc from directories that no rpath of
# ours can reach -- the failing lookup is a dependency of a dlopened module with no loader
# chain back to us. LD_LIBRARY_PATH is the one search path that *is* inherited by a dlopened
# module's own dependency resolution, which is why this script exists and why it is a script
# rather than an rpath.
#
# It is deliberately POSIX sh with no non-coreutils commands, because it runs before anything
# else on a machine we know nothing about.

set -eu

self=${0##*/}

# Resolve through symlinks so `ln -s .../moearc /usr/local/bin/moearc` works. `readlink -f`
# is in GNU coreutils and busybox both; the fallback covers neither having it.
target=$0
if command -v readlink >/dev/null 2>&1 && readlink -f "$0" >/dev/null 2>&1; then
    target=$(readlink -f "$0")
fi
root=$(CDPATH= cd -- "$(dirname -- "$target")" && pwd -P)

real="$root/libexec/$self"
if [ ! -x "$real" ]; then
    echo "moearc: $real is missing -- this bundle is incomplete." >&2
    exit 1
fi

runtime=${MOEARC_RUNTIME_DIR:-$root/runtime}

# `moearc` itself talks to Level Zero directly and needs none of this; the device report has
# to work on a machine where the SYCL runtime was never installed, because telling the user
# what their card is is the first thing it does. Everything that runs a kernel needs it.
case $self in
    moearc) needs_runtime=0 ;;
    *) needs_runtime=1 ;;
esac

if [ "$needs_runtime" = 1 ] && [ ! -e "$runtime/libsycl.so.9" ]; then
    if [ "${MOEARC_NO_FETCH:-0}" = 1 ]; then
        echo "moearc: the SYCL runtime is not installed in $runtime and MOEARC_NO_FETCH=1." >&2
        exit 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "moearc: need python3 once, to fetch the Intel SYCL runtime into $runtime." >&2
        echo "        Install python3, or copy a populated runtime/ directory here." >&2
        exit 1
    fi
    python3 "$root/libexec/fetch-runtime.py" --dest "$runtime" \
        --lock "$root/share/moearc/runtime.lock.json"
fi

# Prepend, never replace: a user who set LD_LIBRARY_PATH did it for a reason, and ours only
# has to win over the system copies of libraries the system usually does not have at all.
if [ -d "$runtime" ]; then
    LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi
# The kernel object sits beside the binaries and is found by name once
# packaging/elf-relocatable.py has shortened its DT_SONAME.
LD_LIBRARY_PATH="$root/libexec${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

export MOEARC_BUNDLE_ROOT="$root"

exec "$real" "$@"
