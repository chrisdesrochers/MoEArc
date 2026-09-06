#!/bin/sh
# MoEArc installer.
#
#   curl -fsSL https://raw.githubusercontent.com/chrisdesrochers/MoEArc/main/packaging/install.sh | sh
#
# Unpacks a release into a prefix, fetches the Intel SYCL runtime once, and links the
# commands onto PATH. `docs/ux.md` allows exactly one thing to be asked of the user, and it
# is the kernel GPU driver; if that is missing this says so by name and stops.

set -eu

REPO=chrisdesrochers/MoEArc
PREFIX=${MOEARC_PREFIX:-$HOME/.local/share/moearc}
BINDIR=${MOEARC_BINDIR:-$HOME/.local/bin}
TARBALL=${MOEARC_TARBALL:-}
VERSION=${MOEARC_VERSION:-latest}

say() { printf 'moearc: %s\n' "$*" >&2; }
die() { printf 'moearc: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "MoEArc is Linux-only today (found $(uname -s))."
[ "$(uname -m)" = x86_64 ] || die "MoEArc ships x86_64 only (found $(uname -m))."

# The one dependency we are allowed to have. Named exactly, with the reason.
#
# Any render node, not renderD128. On a box with an iGPU the discrete card is renderD129, and
# a container is often given only the node it needs -- the first version of this check named
# renderD128 and refused to install on exactly the machine the packaging was proved on.
if ! ls /dev/dri/renderD* >/dev/null 2>&1; then
    say "no /dev/dri/render* device."
    say "MoEArc needs the kernel GPU driver -- 'xe' for Arc B-series (Battlemage),"
    say "'i915' for A-series. It ships with the kernel; check 'lsmod | grep -E \"^xe|^i915\"'."
    die "stopping: nothing else here can work without it."
fi

fetch() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then wget -qO "$2" "$1"
    else die "need curl or wget."; fi
}

tmp=$(mktemp -d "${TMPDIR:-/tmp}/moearc-install.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

if [ -n "$TARBALL" ]; then
    say "installing from $TARBALL"
    cp "$TARBALL" "$tmp/moearc.tar.gz"
else
    if [ "$VERSION" = latest ]; then
        url="https://github.com/$REPO/releases/latest/download/moearc-linux-x86_64.tar.gz"
    else
        url="https://github.com/$REPO/releases/download/$VERSION/moearc-$VERSION-linux-x86_64.tar.gz"
    fi
    say "downloading $url"
    fetch "$url" "$tmp/moearc.tar.gz" || die "download failed. Set MOEARC_TARBALL=/path/to/tarball to install a local one."
fi

tar -C "$tmp" -xzf "$tmp/moearc.tar.gz"
# -mindepth 1: `find` lists its own starting directory first, and the staging directory is
# itself called moearc-install.XXXXXX -- which matches `moearc-*`. Without this the installer
# moves the whole staging directory into the prefix and the tree ends up one level too deep.
src=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'moearc-*' | head -1)
[ -n "$src" ] || die "tarball did not contain a moearc-* directory."

rm -rf "$PREFIX"
mkdir -p "$(dirname "$PREFIX")"
mv "$src" "$PREFIX"

# Runtime up front rather than on first serve: an install that finishes is a better promise
# than one that downloads 230 MB the first time someone is trying to run a model.
if command -v python3 >/dev/null 2>&1; then
    python3 "$PREFIX/libexec/fetch-runtime.py" --dest "$PREFIX/runtime" \
        --lock "$PREFIX/share/moearc/runtime.lock.json"
else
    say "python3 not found -- skipping the SYCL runtime for now."
    say "'moearc' (device report) works without it; running a model does not."
fi

mkdir -p "$BINDIR"
for cmd in moearc moearc-server moearc-bench moearc-selftest; do
    ln -sf "$PREFIX/$cmd" "$BINDIR/$cmd"
done

say "installed to $PREFIX, linked into $BINDIR"
case ":${PATH}:" in
    *":$BINDIR:"*) ;;
    *) say "note: $BINDIR is not on PATH. Add it, or run $BINDIR/moearc directly." ;;
esac
echo
"$PREFIX/moearc" --no-tui || true
