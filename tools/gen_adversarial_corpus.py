#!/usr/bin/env python3
"""Generate a differential-test oracle for the cache_budget Rust port.

Runs FreeToken's reference plan_cache_budget over a deterministic sweep of inputs and
records what it returns -- including which inputs it REJECTS. The Rust port replays this
fixture, so correctness is proven against the original rather than asserted by hand, and
the test stays hermetic (contributors need neither Python nor FreeToken checked out).
"""
import itertools, json, random, sys

# Load cache_budget.py DIRECTLY by path rather than as `freetoken.engine.cache_budget`.
# Its LOGIC is torch-free, but its import path is not: freetoken/utils/__init__.py
# re-exports .torch_utils, so a normal import drags in torch for the sake of one helper.
# That helper is div_ceil -- three lines of integer math in utils/misc.py -- so we load
# misc.py standalone and stub the package around it. No torch, no GPU, no FreeToken
# package init.
import importlib.util, types

FT = "/zfs/swift/projects/FreeToken-study/python/freetoken"

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_misc = _load("ft_misc", f"{FT}/utils/misc.py")
_pkg = types.ModuleType("freetoken"); _pkg.__path__ = []
_utils = types.ModuleType("freetoken.utils")
_utils.div_ceil = _misc.div_ceil          # the real implementation, not a reimplementation
sys.modules["freetoken"] = _pkg
sys.modules["freetoken.utils"] = _utils

_mod = _load("ft_cache_budget", f"{FT}/engine/cache_budget.py")
plan_cache_budget = _mod.plan_cache_budget
net_cache_budget_bytes = _mod.net_cache_budget_bytes
required_bytes = _mod.required_bytes
resolve_moe_cache_auto = _mod.resolve_moe_cache_auto

cases = []

def record(kind, args):
    entry = {"kind": kind, "args": args}
    try:
        moe, pages, overlap = plan_cache_budget(**args)
        entry["ok"] = {"moe_cache_size": moe, "num_pages": pages, "prefill_overlap": overlap}
    except AssertionError as e:
        entry["err"] = str(e)
    except ZeroDivisionError:
        entry["err"] = "ZeroDivisionError"
    cases.append(entry)

# --- 1. structured sweep: every corner of the clamp/overlap logic ---------------
GRID = dict(
    budget_bytes=[0, 1, 100, 10_000, 1 << 20, 6 * (1 << 30), 11 * (1 << 30)],
    per_expert_bytes=[1, 100, 4 * (1 << 20), 22 * (1 << 20)],
    cache_per_page=[1, 10, 32 * (1 << 10), 1 << 20],
    num_experts=[1, 4, 8],
    total_experts=[4, 64, 128],
    prefill_overlap=[True, False],
    kv_reserve_pages=[0, 2, 64],
    max_slots=[6, 64, 512],
)
keys = list(GRID)
# Full cross-product is ~1.2M; take a deterministic stride through it instead.
combos = list(itertools.product(*(GRID[k] for k in keys)))
for combo in combos[::137]:
    record("grid", dict(zip(keys, combo)))

# --- 2. randomised fuzz, including hostile values ------------------------------
rng = random.Random(20260904)
for _ in range(3000):
    record("fuzz", dict(
        budget_bytes=rng.choice([0, rng.randint(-(1 << 20), 1 << 34)]),
        per_expert_bytes=rng.choice([1, rng.randint(1, 1 << 26)]),
        cache_per_page=rng.choice([1, rng.randint(1, 1 << 22)]),
        num_experts=rng.randint(1, 16),
        total_experts=rng.randint(1, 256),
        prefill_overlap=rng.random() < 0.5,
        kv_reserve_pages=rng.randint(0, 256),
        max_slots=rng.randint(1, 1024),
    ))

# --- 3. the pure helpers -------------------------------------------------------
helpers = []
for _ in range(400):
    a = dict(memory_ratio=rng.choice([0.0, 0.5, 0.9, 1.0, rng.random()]),
             baseline_free=rng.randint(0, 1 << 34),
             weights_bytes=rng.randint(0, 1 << 33),
             fixed_cache_size=rng.randint(0, 1 << 30))
    b = dict(moe_cache_size=rng.randint(0, 512), num_pages=rng.randint(0, 1 << 20),
             per_expert_bytes=rng.randint(1, 1 << 26), cache_per_page=rng.randint(1, 1 << 22))
    helpers.append({"net": {"args": a, "want": net_cache_budget_bytes(**a)},
                    "req": {"args": b, "want": required_bytes(**b)}})

# --- 4. resolve_moe_cache_auto: the real entry point ---------------------------
def _resolve_case(a):
    e = {"args": a}
    try:
        moe, pages, overlap = resolve_moe_cache_auto(**a)
        e["ok"] = {"moe_cache_size": moe, "num_pages": pages, "prefill_overlap": overlap}
    except AssertionError as ex:
        e["err"] = str(ex)
    return e

resolve = []
for _ in range(600):
    a = dict(
        baseline_free=rng.choice([12 * (1 << 30), 16 * (1 << 30), rng.randint(0, 1 << 35)]),
        weights_bytes=rng.randint(0, 1 << 34),
        memory_ratio=rng.choice([0.9, 0.75, 1.0, rng.random()]),
        cache_per_page=rng.choice([1 << 16, rng.randint(1, 1 << 22)]),
        fixed_cache_size=rng.randint(0, 1 << 28),
        per_expert_bytes=rng.choice([4 << 20, rng.randint(1, 1 << 26)]),
        num_experts=rng.randint(1, 16),
        total_experts=rng.choice([64, 128, 256]),
        prefill_overlap=rng.random() < 0.5,
        kv_reserve_tokens=rng.randint(0, 1 << 16),
        page_size=rng.choice([16, 64, 128]),
        quant_format=rng.choice(["nvfp4_marlin", "mxfp4", "bf16"]),
    )
    resolve.append(_resolve_case(a))

# --- 5. targeted: make the Marlin slot cap observable ---------------------------
# A negative-control run showed the branch was dead in the fixture above: every
# total_experts there is <= 256, so hi = min(total_experts, 992) is total_experts whether
# or not the 992 cap applies. Only total_experts > 992 distinguishes them. No shipping
# model has that many experts -- which is itself the finding -- but MoEArc must not
# inherit the constant silently, so it gets pinned here.
for te in [993, 1024, 2048]:
    for qf in ["nvfp4_marlin", "mxfp4"]:
        for pe in [1 << 20, 1 << 24]:
            resolve.append(_resolve_case(dict(
                baseline_free=64 * (1 << 30), weights_bytes=1 << 30, memory_ratio=0.9,
                cache_per_page=1 << 16, fixed_cache_size=0, per_expert_bytes=pe,
                num_experts=8, total_experts=te, prefill_overlap=False,
                kv_reserve_tokens=1024, page_size=64, quant_format=qf)))

out = {"source": "freetoken/engine/cache_budget.py",
       "generated_by": "tools/gen_cache_budget_oracle.py",
       "plan": cases, "helpers": helpers, "resolve": resolve}
json.dump(out, open(sys.argv[1], "w"), indent=0)
ok = sum(1 for c in cases if "ok" in c)
rok = sum(1 for c in resolve if "ok" in c)
print(f"{len(cases)} plan ({ok} ok / {len(cases)-ok} rejected), {len(helpers)} helper, {len(resolve)} resolve ({rok} ok)")
