# Cutting a MoEArc release

The checklist for publishing a tarball people can `curl | sh`. Everything here is the owner's
to run: it tags a public repo and publishes artefacts under the project's name, so no agent
does any of it.

`packaging/install.sh` and this file are one contract. If they disagree, the installer is
wrong — it is the half a stranger runs.

---

## The contract, in three lines

| | |
| --- | --- |
| **Tag** | `v<MAJOR>.<MINOR>.<PATCH>` — e.g. `v0.1.0` |
| **Assets** | `moearc-linux-x86_64.tar.gz` and `moearc-linux-x86_64.tar.gz.sha256` — **these exact names, under every tag** |
| **Version** | inside the tarball: the top-level directory is `moearc-<version>-linux-x86_64/`, and `share/moearc/BUILD-INFO.txt` records commit and toolchain |

🔴 **The asset name never carries the version, and that is deliberate.** GitHub's
`releases/latest/download/<name>` endpoint can only resolve a name known in advance, so a
versioned filename cannot be fetched through it — `install.sh` would have to ask the API which
release is newest before it could ask for a file. One fixed name makes `latest` and a pinned
`MOEARC_VERSION=v0.1.0` the same code path, and the version still travels, inside the archive
where it cannot be renamed away from its bytes.

⚠️ `bundle.sh` writes `dist/moearc-<version>-linux-x86_64.tar.gz`. **Upload it under the fixed
asset name** (step 6). Renaming a file does not change its bytes, so the checksum still holds —
but regenerate the `.sha256` against the uploaded name anyway, so the published file is
self-consistent. `install.sh` compares the digest field only and tolerates either.

---

## Before you start

- A build host with **oneAPI** installed (`icpx`), an **Arc card**, and the Intel GPU runtime.
  The user's machine never compiles anything; this one compiles everything.
- 🔴 **Build on the oldest glibc you are willing to support.** `moearc` inherits its
  `GLIBC_2.39` floor from the build host, which today is Ubuntu 26.04 — that is Ubuntu 24.04,
  Fedora 40 and Debian 13 or newer, and it silently excludes Debian 12 and RHEL 9.
  `share/moearc/BUILD-INFO.txt` states the measured floor per binary; read it before publishing
  and put it in the release notes.
- `podman` or `docker`, for the clean-room step, which is not optional.
- A small MoE GGUF to verify against. `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` (4.2 GB) is the
  one this project uses.

---

## 1 — Pick the version and commit it

```sh
# Cargo.toml, [workspace.package]
version = "0.1.0"
```

The workspace is at `0.0.0` until someone decides otherwise. Commit that bump on `main` first;
the tag must point at a commit whose `Cargo.toml` already says what the tag says.

## 2 — Gates, on a clean tree

```sh
git status --porcelain          # must be empty
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features moearc-engine/gpu -- -D warnings
cargo test --workspace
```

A dirty tree is recorded in `BUILD-INFO.txt` as `dirty: YES` and makes the build
unreproducible. Do not publish one.

## 3 — Build the tarball

```sh
source /opt/intel/oneapi/setvars.sh
packaging/bundle.sh --build --version 0.1.0
```

`--version` matters: without it the name picks up a `+g<sha>` suffix from git, and `+` is not a
character you want in a URL or a filename people type.

This produces:

```
dist/moearc-0.1.0-linux-x86_64.tar.gz         ~3.8 MB
dist/moearc-0.1.0-linux-x86_64.tar.gz.sha256
```

⚠️ **That figure was `~4.8 MB` until 2026-09-09 and had been wrong since the pivot.** The
payload became a single binary when `moearc-server`, `moearc-bench` and `moearc-selftest` left
it; built from the worktree at commit `da645a6`, with the traces staged, the archive measures
**3,889,176 bytes**. Read the size `bundle.sh` prints beside the artefact path rather than this
line — that one is measured and this one is copied, and the exact byte count moves under any
packaging change, because `launcher.sh` is itself installed into the archive as `moearc`.

**It contains no third-party binaries.** Intel's SYCL runtime is not in it and is no longer
downloaded at install time either: nothing in the payload loads it, so `install.sh` fetches it
only when asked, with `MOEARC_FETCH_RUNTIME=1`, pinned by SHA-256 in `runtime.lock.json`. The
tarball is Apache-2.0. `packaging/THIRD-PARTY.md` is why; read it before you consider
`--with-runtime`, which vendors the runtime into the archive and produces something that is
**not** Apache-2.0 and must not be published as one. (⚠️ `--with-runtime`'s "29 MB" was
measured before the payload shrank and nobody has re-measured it; the fetched runtime is
78.6 MiB on disk, so expect it to dominate whatever the number is now.)

### Reproducibility — what is and is not claimed

`bundle.sh` pins the two things that used to make the artefact differ from itself: the `built:`
stamp and every tar member's mtime, uid, gid and order. Both come from `SOURCE_DATE_EPOCH`,
which defaults to the commit's own timestamp, and the gzip header carries no name or time
(`gzip -n`). So **the same commit, on the same machine, with the same toolchain, produces the
same `sha256`** — check it before you publish:

```sh
packaging/bundle.sh --version 0.1.0 --out /tmp/r1
packaging/bundle.sh --version 0.1.0 --out /tmp/r2
cmp /tmp/r1/moearc-0.1.0-linux-x86_64.tar.gz /tmp/r2/moearc-0.1.0-linux-x86_64.tar.gz && echo same
```

⚠️ **That is repeatability, not reproducibility across machines.** Nobody has built this on a
second host and compared, so a different rustc or a different absolute build path may well
produce different bytes. (`icpx` used to be on that list and is not any more: nothing in the
payload is compiled by it.) The claim to make in release notes is the honest
one: *these bytes came from this commit with the toolchain recorded in `BUILD-INFO.txt`*, and
the sha256 lets anyone check they got what was published. It is auditable, not
bit-for-bit reproducible from source by a third party.

## 4 — 🔴 Verify in a clean room. This is the gate.

🔴 **This gate got weaker and it is now TWO steps, because one of them can no longer be done
in the container.** `verify-clean.sh` used to take a model and run a real forward pass against
stored token ids. That ran `moearc-bench`, which left the payload with the retired SYCL engine
along with `moearc-selftest` and `moearc-server`. The container can still prove the hard part
of packaging — that the card is found on a machine that has never had oneAPI — and it can no
longer prove that a model computes. `docs/packaging.md` measured that those are different
questions: a driver stack can pass detection and then fail inference, each layer failing one
step later than the one above it, everything before the failure looking healthy.

**4a — the clean room, for detection.**

```sh
MOEARC_RUNTIME_CACHE=/tmp/moearc-runtime-cache \
  packaging/verify-clean.sh dist/moearc-0.1.0-linux-x86_64.tar.gz
```

It must print **`clean-environment verification PASSED`**. Anything else is not a release. It
will also print a `NOT PROVEN` line; that is expected, and 4b is what covers it.

⚠️ `MOEARC_RUNTIME_CACHE` is inert here and is left in the command only because it costs
nothing. `verify-clean.sh` unpacks the tarball directly — it never runs `install.sh` — and the
one binary in the tarball is the one that needs no SYCL runtime, so there is no fetch for the
cache to save. It becomes load-bearing again the day something in the payload needs one.

**4b — a real model, on a machine with llama.cpp.** Not a clean room — the point is the
inference, not the environment — so run it from the unpacked tarball on a box with an Arc card
and a llama.cpp build:

```sh
/tmp/unpacked/moearc serve olmoe-1b-7b-0924-instruct --no-tui --port 8099 &
curl -s localhost:8099/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"olmoe","messages":[{"role":"user","content":"Capital of France?"}],
       "max_tokens":16,"temperature":0}'
```

The completion has to be text, from the discrete card, with the device line in the startup
banner naming the Arc and not the iGPU. ⬜ This step returns to 4a the day llama.cpp is
bundled; until then a release is gated on a human running it and reading the output.

## 5 — Tag

```sh
git tag -a v0.1.0 -m "MoEArc v0.1.0"
git push origin v0.1.0
```

## 6 — Publish

```sh
cd dist
cp moearc-0.1.0-linux-x86_64.tar.gz moearc-linux-x86_64.tar.gz
sha256sum moearc-linux-x86_64.tar.gz > moearc-linux-x86_64.tar.gz.sha256

gh release create v0.1.0 \
  --title "MoEArc v0.1.0" \
  --notes-file ../packaging/release-notes-0.1.0.md \
  moearc-linux-x86_64.tar.gz \
  moearc-linux-x86_64.tar.gz.sha256
```

Release notes should carry, at minimum: the glibc floor per binary from `BUILD-INFO.txt`, the
`x86-64 Linux only` constraint, the one dependency (`xe` or `i915`, which ships with the
kernel), and the sha256.

## 7 — Install it the way a stranger will

From a machine that is **not** the build host, and with nothing set:

```sh
curl -fsSL https://raw.githubusercontent.com/chrisdesrochers/MoEArc/main/packaging/install.sh | sh
```

Expect, in order: the download, `sha256 verified against the published checksum`, the two-line
note that **no SYCL runtime was fetched**, `installed to …`, and then the device report.

🔴 **That note is not a warning and the release does not wait on it.** Until 2026-09-09 this
step expected a runtime fetch in that slot, and the installer performed one — 199.5 MiB pulled
from the index on every install, into a `runtime/` directory that nothing in the payload opens
(`packaging/install.sh` carries the measurement). What a stranger needs in order to run a model
is their own llama.cpp `llama-server`, which links its own runtime; step 4b is where that is
checked. `MOEARC_FETCH_RUNTIME=1` puts the old behaviour back for one install.

If `install.sh` says
**"MoEArc has no published release yet"** the assets did not attach under the expected names —
re-check step 6 rather than editing the installer.

Then pin-check the other path:

```sh
MOEARC_VERSION=v0.1.0 sh install.sh
```

---

## What `install.sh` does when this has not been done

It fails on purpose and says so, because until step 6 the URL 404s. Each case was run:

| situation | what the user sees |
| --- | --- |
| no release published | *"MoEArc has no published release yet, or the latest one has no moearc-linux-x86_64.tar.gz attached"*, the releases URL, and the two commands to build one locally |
| `MOEARC_VERSION` names a tag with no asset | the same, naming the tag |
| network down / DNS | *"Nothing answered — check the network, a proxy, or DNS"* |
| HTTP 403 | named as a rate limit or network policy, **not** as a missing release |
| a short or non-gzip download | refused before `tar` sees it, with the byte count |
| checksum mismatch | refused, both digests printed |

It never hands a truncated file to `tar`, and curl's own `(22) The requested URL returned
error: 404` is captured and replayed only when there is nothing better to say — a raw 404 above
an explanation buries the explanation.

The escape hatch, which is the path this project has actually exercised end to end:

```sh
packaging/bundle.sh --build
MOEARC_TARBALL=dist/moearc-*-linux-x86_64.tar.gz sh packaging/install.sh
```

---

## Environment `install.sh` reads

| | |
| --- | --- |
| `MOEARC_VERSION` | a tag, e.g. `v0.1.0`. Default `latest`. |
| `MOEARC_TARBALL` | install this local file instead of downloading. Skips the checksum step. |
| `MOEARC_PREFIX` | where the bundle lands. Default `~/.local/share/moearc`. |
| `MOEARC_BINDIR` | where the command is linked. Default `~/.local/bin`. One name, `moearc`; it was four before the pivot, and linking a name with nothing behind it leaves a dangling symlink on `PATH`. |
| `MOEARC_FETCH_RUNTIME` | `1` fetches Intel's SYCL runtime into `$MOEARC_PREFIX/runtime` (199.5 MiB down, 78.6 MiB kept). **Off by default** — nothing in the payload loads it. Set it if your own llama.cpp SYCL build has no oneAPI beside it, or to stage the runtime for a machine that will be offline. |

`MOEARC_PREFIX` is `rm -rf`'d before the new tree is moved in; the installer refuses `/`, `$HOME`
and the empty string, and nothing else. Point it at a directory MoEArc owns.
