//! Host-side expert execution, overlapped with the device.
//!
//! The engine's other answer to an expert that is not resident is [`crate::moe::stage`]: copy it
//! over PCIe and let the card compute it. This module is the second answer — **compute it on the
//! CPU, out of the memory-mapped file, while the GPU is busy with something else** — and the
//! whole point is the word *while*. A host path that merely replaces device work is
//! substitution, which `bench/baselines/qwen3-30b-a3b.md` already measures on llama.cpp
//! (`-ncmoe`) and which gets monotonically slower the more of it there is. Overlap is a
//! different claim and it needs a different mechanism: [`HostExecutor::submit`] returns
//! immediately, the caller issues its GPU work, and [`HostExecutor::sync`] collects.
//!
//! # What one job is
//!
//! One block, some subset of the experts that block's router named, and the block's post-norm
//! activation. For each expert the job computes what the device computes — `silu(gate . x) *
//! (up . x)`, then `down . that` — and `sync` hands back the router-weighted sum of those
//! vectors. The caller adds it to whatever the GPU produced for the experts it kept.
//!
//! # Why the weights are raw pointers
//!
//! The executor holds an `Arc<MappedModel>`, so the mapping outlives every worker it spawns and
//! [`HostExecutor::drop`] joins them before the `Arc` is released. A slot in `Shared::table` is a
//! `(pointer, length)` into those pages, resolved once at construction through
//! `MappedModel::expert` — the validated accessor — rather than by arithmetic on a base address.
//! That is the only reason a lifetime is erased here, and it is why the erasure is sound.
//!
//! # The fork-join, and why it is a full handshake
//!
//! Workers spin on an epoch counter, then park. `submit` writes the job, bumps the epoch and
//! unparks; `sync` waits for **every** worker to acknowledge that epoch, not merely for the task
//! counters to drain. The weaker condition is not enough: a worker that had read the epoch but
//! not yet claimed a task would still be inside the job when the next `submit` overwrote it, and
//! the symptom would be an expert computed against the wrong block's activation — finite,
//! fluent, wrong, which is the failure mode this engine keeps meeting.
//!
//! # Determinism
//!
//! Every output element is written by exactly one task, and the final weighted sum runs over the
//! experts in router order on the calling thread, so a job's result does not depend on how the
//! work was scheduled. Two runs with the same policy produce the same token ids.
//!
//! 🔴 It is **not** bit-identical to the device path and is not meant to be. The device
//! dequantises and reduces in a f32 tree; this reduces a row in order and folds the K-quant
//! block minimum out of the inner loop (`sum((d*sc*q - dmin*m) * x) == d*sc*sum(q*x) -
//! dmin*m*sum(x)`), which is the same value in exact arithmetic and a different rounding.
//! `tests/host_experts_gpu.rs` measures the disagreement against the device on real weights.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use moearc_kernels::QuantType;
use moearc_kernels::reference::{KVALUES_MXFP4, e8m0_half, f16_to_f32};
use moearc_model::tensors::{ExpertBank, MappedModel, names};

use crate::host_budget::{self, HostResidency, HostRouting};
use crate::moe::Activation;

/// Why the host executor could not start or run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostError {
    /// A bank uses a quantisation, or a shape, this module has no host kernel for. Refused at
    /// construction rather than at the first token, so a run cannot get half way and stop.
    Unsupported(String),
    /// The caller asked for more experts in one job than the executor was built for.
    TooManyExperts { asked: usize, capacity: usize },
    /// A worker thread died. Every subsequent call fails rather than hanging on a handshake
    /// nobody will complete.
    WorkerDied,
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(m) => write!(f, "host expert execution: {m}"),
            Self::TooManyExperts { asked, capacity } => {
                write!(f, "host job of {asked} experts, executor built for {capacity}")
            }
            Self::WorkerDied => write!(f, "a host expert worker thread died"),
        }
    }
}

impl std::error::Error for HostError {}

// =======================================================================================
// Policy
// =======================================================================================

/// Which of a block's cache misses go to the CPU.
///
/// Deliberately three shapes and no adaptivity. The question this module exists to answer is
/// whether overlapped host execution pays *at all*; an adaptive scheme built before that is
/// settled would only make the answer harder to read.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum HostPolicy {
    /// Every miss is streamed. The engine as it was.
    #[default]
    Off,
    /// This fraction of a block's misses, rounded up. `1.0` sends every miss to the CPU, which
    /// freezes the resident pool: nothing new is ever admitted.
    Fraction(f32),
    /// Stream the first `n` misses of a block and send the rest to the CPU.
    Over(usize),
}

impl HostPolicy {
    /// How many of `misses` this policy routes host-side.
    pub fn host_count(self, misses: usize) -> usize {
        let n = match self {
            Self::Off => 0,
            Self::Fraction(f) => ((misses as f32) * f.clamp(0.0, 1.0)).ceil() as usize,
            Self::Over(n) => misses.saturating_sub(n),
        };
        n.min(misses)
    }

    /// Whether this policy can ever route anything host-side.
    pub fn is_off(self) -> bool {
        match self {
            Self::Off => true,
            Self::Fraction(f) => f <= 0.0,
            Self::Over(_) => false,
        }
    }

    /// Whether the host RAM budget can back what this policy routes to the CPU.
    ///
    /// 🔴 This policy places **compute**; [`crate::host_budget::HostBudget`] places **data**;
    /// and the executor below reads expert weights straight out of the mapping. So the two are
    /// coupled and the coupling has no other expression in the code:
    /// [`HostPolicy::Fraction`]`(1.0)` against a budget that backs nothing sends **every**
    /// host-executed expert to the drive. See [`HostRouting`] — this reports that state and
    /// deliberately does not act on it.
    pub fn backing(self, residency: HostResidency) -> HostRouting {
        host_budget::routing_backing(!self.is_off(), residency)
    }
}

/// One spelling for every tool: `off`, `frac:<0..1>`, `over:<n>`, `all`.
impl std::str::FromStr for HostPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "off" {
            return Ok(Self::Off);
        }
        if s == "all" {
            return Ok(Self::Fraction(1.0));
        }
        if let Some(r) = s.strip_prefix("frac:") {
            let f: f32 = r.parse().map_err(|_| format!("`{r}` is not a fraction"))?;
            if !(0.0..=1.0).contains(&f) {
                return Err(format!("`{r}` is outside 0..1"));
            }
            return Ok(Self::Fraction(f));
        }
        if let Some(r) = s.strip_prefix("over:") {
            let n: usize = r.parse().map_err(|_| format!("`{r}` is not a count"))?;
            return Ok(Self::Over(n));
        }
        Err(format!("`{s}` is not a host policy: expected `off`, `all`, `frac:<f>` or `over:<n>`"))
    }
}

impl std::fmt::Display for HostPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Fraction(v) => write!(f, "frac:{v}"),
            Self::Over(n) => write!(f, "over:{n}"),
        }
    }
}

// =======================================================================================
// Geometry
// =======================================================================================

/// One bank of one block: what shape its experts are and how they are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankSpec {
    pub ty: QuantType,
    /// Outputs — the number of rows the matvec produces.
    pub n_rows: usize,
    /// Reduction length, the contiguous axis.
    pub n_cols: usize,
}

impl BankSpec {
    fn row_bytes(&self) -> usize {
        (self.n_cols / self.ty.block_elems()) * self.ty.block_bytes()
    }

    fn check(&self, what: &str) -> Result<(), HostError> {
        if !matches!(self.ty, QuantType::Q4K | QuantType::Q6K | QuantType::Mxfp4) {
            return Err(HostError::Unsupported(format!(
                "{what} is {:?}; only Q4_K, Q6_K and MXFP4 have host kernels",
                self.ty
            )));
        }
        if self.n_cols % self.ty.block_elems() != 0 {
            return Err(HostError::Unsupported(format!(
                "{what} has {} columns, not a whole number of {:?} blocks",
                self.n_cols, self.ty
            )));
        }
        Ok(())
    }
}

/// The three banks of one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSpec {
    pub gate: BankSpec,
    pub up: BankSpec,
    pub down: BankSpec,
}

/// What the executor needs to know about the model.
///
/// 🔴 The last two fields are graph facts, not shapes, and they are here because the host path
/// must compute the **same function** as the device path. An executor that ran plain SwiGLU
/// where the device ran `swiglu_oai`, or skipped a bias the device applied, would produce a
/// model whose output changed with the host policy — and the policy is a performance knob.
/// `tests/host_experts_gpu.rs` asserts that it does not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub n_block: usize,
    pub n_expert: usize,
    /// The most experts one step can name, and so the most one job can hold.
    pub n_expert_used: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    /// Whether each expert bank carries a per-expert f32 bias.
    pub expert_bias: bool,
    /// The activation the gate and up projections feed.
    pub act: Activation,
}

// =======================================================================================
// Shared state
// =======================================================================================

/// A borrow of the memory-mapped model with its lifetime erased.
///
/// 🔴 Sound only because `Shared` owns the `Arc<MappedModel>` these point into and
/// [`HostExecutor::drop`] joins every worker before that `Arc` is released. Nothing else in this
/// module may construct one from anything but a `TensorView` of that mapping.
#[derive(Clone, Copy)]
struct Weights {
    ptr: *const u8,
    len: usize,
}

// SAFETY: see the note on `Weights`. The pages are read-only for the executor's whole life.
unsafe impl Send for Weights {}
unsafe impl Sync for Weights {}

impl Weights {
    /// # Safety
    /// The mapping must outlive every use of the returned slice.
    unsafe fn as_slice<'a>(self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// A fixed-size float buffer many threads write disjoint ranges of.
///
/// The cached pointer is what makes that sound: handing out `&mut [f32]` for a sub-range never
/// forms a `&mut` over the whole allocation, so two tasks writing different rows are not
/// aliasing references to one object. The buffer is allocated once and never resized, so the
/// pointer stays valid.
struct Scratch {
    _own: UnsafeCell<Box<[f32]>>,
    ptr: *mut f32,
    len: usize,
}

impl Scratch {
    fn new(n: usize) -> Self {
        let mut b = vec![0.0f32; n].into_boxed_slice();
        let ptr = b.as_mut_ptr();
        Self { _own: UnsafeCell::new(b), ptr, len: n }
    }

    /// # Safety
    /// No other live borrow may overlap `off..off + len`.
    unsafe fn range_mut<'a>(&self, off: usize, len: usize) -> &'a mut [f32] {
        debug_assert!(off + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), len) }
    }

    /// # Safety
    /// No `range_mut` borrow overlapping `off..off + len` may be live.
    unsafe fn range<'a>(&self, off: usize, len: usize) -> &'a [f32] {
        debug_assert!(off + len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(off), len) }
    }
}

/// One block's work, written by `submit` and read by every worker of that epoch.
struct Job {
    spec: BlockSpec,
    /// `[gate, up, down]` per routed expert, in the order the router named them.
    banks: Vec<[Weights; 3]>,
    /// `[gate, up, down]` biases for the same experts, or empty when the model has none.
    bias: Vec<[Weights; 3]>,
    /// The block's post-norm activation, `n_embd` long.
    x: Vec<f32>,
    /// `x` summed in groups of 32 — what the Q4_K kernel needs to hoist the block minimum out
    /// of its inner loop. Depends only on `x`, so it is computed once per job, not per row.
    xsum: Vec<f32>,
    n_experts: usize,
    chunk_a: usize,
    chunk_b: usize,
    chunks_a: usize,
    chunks_b: usize,
    started: Instant,
}

const NO_BANK: BankSpec = BankSpec { ty: QuantType::Q4K, n_rows: 0, n_cols: 0 };

impl Job {
    fn empty() -> Self {
        Self {
            spec: BlockSpec { gate: NO_BANK, up: NO_BANK, down: NO_BANK },
            banks: Vec::new(),
            bias: Vec::new(),
            x: Vec::new(),
            xsum: Vec::new(),
            n_experts: 0,
            chunk_a: 1,
            chunk_b: 1,
            chunks_a: 0,
            chunks_b: 0,
            started: Instant::now(),
        }
    }
}

/// Counters the caller can read without disturbing anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HostStats {
    /// Jobs submitted.
    pub jobs: u64,
    /// Experts computed host-side.
    pub experts: u64,
    /// Wall time from `submit` to the last task finishing, summed. What the CPU was busy for,
    /// and the number that has to be *hidden* for overlap to be worth anything.
    pub busy_nanos: u64,
    /// Wall time the calling thread spent blocked inside `sync`, summed. The gap between this
    /// and `busy_nanos` **is** the overlap: if they are equal, nothing was hidden.
    pub wait_nanos: u64,
}

struct Shared {
    /// Kept alive because `Weights` points into it.
    _mapped: Arc<MappedModel>,
    geom: Geometry,
    n_threads: usize,
    /// `[(block * 3 + bank) * n_expert + expert]`, resolved once through `MappedModel::expert`.
    table: Vec<Weights>,
    /// The same index over the expert biases, or empty. Unlike `table` these **are** derived by
    /// arithmetic, and legitimately so: a bias is a plain contiguous f32 matrix whose shape the
    /// header states outright (`[n_ff, n_expert]`, `[n_embd, n_expert]`), and that shape is
    /// checked against the tensor's own dimensions before a single offset is taken.
    bias_table: Vec<Weights>,

    stop: AtomicBool,
    /// Bumped by `submit`. Workers wait for it to move.
    epoch: AtomicU64,
    /// Per worker, the last epoch it finished. `sync` waits for all of them.
    acks: Vec<AtomicU64>,

    cursor_a: AtomicUsize,
    cursor_b: AtomicUsize,
    done_a: AtomicUsize,
    /// Set by whichever worker completes the last phase-A task, after it prepares `xsum_act`.
    gate_b: AtomicBool,
    done_b: AtomicUsize,

    job: UnsafeCell<Job>,
    act: Scratch,
    out: Scratch,
    xsum_act: Scratch,

    jobs: AtomicU64,
    experts: AtomicU64,
    busy_nanos: AtomicU64,
    wait_nanos: AtomicU64,
}

// SAFETY: the `UnsafeCell` and the `Scratch` buffers are governed by the epoch handshake
// documented on the module. No worker touches them outside a job it was woken for, and `sync`
// does not return until every worker has acknowledged that job.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// A submitted job, redeemed by [`HostExecutor::sync`].
///
/// Not `Clone`: syncing one epoch twice would return before the second job had finished.
#[derive(Debug)]
pub struct HostJob {
    epoch: u64,
    n_experts: usize,
}

impl HostJob {
    /// Experts this job put on the CPU.
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }
}

// =======================================================================================
// The executor
// =======================================================================================

/// A pool of pinned worker threads that computes experts out of the mapping.
pub struct HostExecutor {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Handles for `unpark`, kept separately because `JoinHandle::thread` borrows.
    threads: Vec<thread::Thread>,
    /// The router weights of the job in flight, for the combine in `sync`.
    weights: UnsafeCell<Vec<f32>>,
}

// SAFETY: `weights` is touched only by `submit` and `sync`, which are called in that order from
// the one device thread. It is an `UnsafeCell` so that both can take `&self`, which is what lets
// an `Option<HostExecutor>` field be borrowed beside `&mut self.state` in `Model::decode`.
unsafe impl Sync for HostExecutor {}

/// How long a worker spins before parking, in `spin_loop()` iterations.
///
/// During generation a block arrives every few hundred microseconds, so a spinning worker sees
/// the next job in tens of nanoseconds and never pays a futex round trip; between requests it
/// parks. Tunable because the right value is a property of the machine and the model.
fn spin_budget() -> u32 {
    std::env::var("MOEARC_HOST_SPIN").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000_000)
}

/// Threads to run experts on.
///
/// One fewer than the machine has cores, by default, and not out of politeness: the device
/// thread submits every kernel and every staging copy, and a host executor that descheduled it
/// would buy CPU throughput by delaying the GPU work it is supposed to be hiding behind.
pub fn default_threads() -> usize {
    if let Some(n) = std::env::var("MOEARC_HOST_THREADS").ok().and_then(|v| v.parse().ok()) {
        return n;
    }
    thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(1).max(1)
}

impl HostExecutor {
    /// Build the pool and resolve every expert's bytes.
    ///
    /// `blocks` must have one entry per block, in block order.
    pub fn new(
        mapped: Arc<MappedModel>,
        geom: Geometry,
        blocks: &[BlockSpec],
        n_threads: usize,
    ) -> Result<Self, HostError> {
        if blocks.len() != geom.n_block {
            return Err(HostError::Unsupported(format!(
                "{} block specs for a {}-block model",
                blocks.len(),
                geom.n_block
            )));
        }
        for (i, b) in blocks.iter().enumerate() {
            b.gate.check(&format!("block {i} gate"))?;
            b.up.check(&format!("block {i} up"))?;
            b.down.check(&format!("block {i} down"))?;
            // The SwiGLU pairs gate and up element for element and the down projection reduces
            // over their output, so these are not stylistic checks: a mismatch would index off
            // the end of one buffer or silently truncate the other.
            if b.gate.n_rows != b.up.n_rows || b.gate.n_cols != b.up.n_cols {
                return Err(HostError::Unsupported(format!(
                    "block {i}: gate is {}x{} and up is {}x{}",
                    b.gate.n_rows, b.gate.n_cols, b.up.n_rows, b.up.n_cols
                )));
            }
            if b.gate.n_rows != geom.n_ff
                || b.gate.n_cols != geom.n_embd
                || b.down.n_cols != geom.n_ff
                || b.down.n_rows != geom.n_embd
            {
                return Err(HostError::Unsupported(format!(
                    "block {i} does not match n_embd={} n_ff={}",
                    geom.n_embd, geom.n_ff
                )));
            }
        }

        // Resolved through the validated accessor, once, rather than by arithmetic on a base
        // address. 🔴 An expert bank is a slice of a stacked tensor and its stride is not stated
        // anywhere in the file; deriving one would be a guess that produces fluent nonsense when
        // it is wrong.
        let mut table = Vec::with_capacity(geom.n_block * 3 * geom.n_expert);
        for b in 0..geom.n_block {
            for bank in [ExpertBank::Gate, ExpertBank::Up, ExpertBank::Down] {
                for e in 0..geom.n_expert {
                    let v = mapped
                        .expert(b as u32, bank, e as u32)
                        .map_err(|e| HostError::Unsupported(e.to_string()))?;
                    table.push(Weights { ptr: v.data.as_ptr(), len: v.data.len() });
                }
            }
        }

        // The per-expert biases, sliced out of three plain f32 matrices per block.
        let mut bias_table = Vec::new();
        if geom.expert_bias {
            for b in 0..geom.n_block {
                for (bank, (suffix, rows)) in [
                    (names::FFN_GATE_EXPS_BIAS, geom.n_ff),
                    (names::FFN_UP_EXPS_BIAS, geom.n_ff),
                    (names::FFN_DOWN_EXPS_BIAS, geom.n_embd),
                ]
                .into_iter()
                .enumerate()
                {
                    let v = mapped
                        .block_tensor(b as u32, suffix)
                        .map_err(|e| HostError::Unsupported(e.to_string()))?;
                    // 🔴 Checked, not assumed. The offsets below are arithmetic, and arithmetic
                    // on a shape read wrongly indexes into a neighbouring expert's bias — which
                    // is finite, small, and invisible in the output.
                    if v.dims != [rows as u64, geom.n_expert as u64] {
                        return Err(HostError::Unsupported(format!(
                            "`{}` is {:?}; expected [{rows}, {}]",
                            v.name, v.dims, geom.n_expert
                        )));
                    }
                    if v.data.len() != rows * geom.n_expert * 4 {
                        return Err(HostError::Unsupported(format!(
                            "`{}` holds {} B for {rows} x {} f32",
                            v.name,
                            v.data.len(),
                            geom.n_expert
                        )));
                    }
                    debug_assert_eq!(bias_table.len(), (b * 3 + bank) * geom.n_expert);
                    for e in 0..geom.n_expert {
                        bias_table
                            .push(Weights { ptr: v.data[e * rows * 4..].as_ptr(), len: rows * 4 });
                    }
                }
            }
        }

        let cap = geom.n_expert_used;
        let n_threads = n_threads.max(1);
        let shared = Arc::new(Shared {
            _mapped: mapped,
            geom,
            n_threads,
            table,
            bias_table,
            stop: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            acks: (0..n_threads).map(|_| AtomicU64::new(0)).collect(),
            cursor_a: AtomicUsize::new(0),
            cursor_b: AtomicUsize::new(0),
            done_a: AtomicUsize::new(0),
            gate_b: AtomicBool::new(false),
            done_b: AtomicUsize::new(0),
            job: UnsafeCell::new(Job::empty()),
            act: Scratch::new(cap * geom.n_ff),
            out: Scratch::new(cap * geom.n_embd),
            xsum_act: Scratch::new(cap * geom.n_ff.div_ceil(32)),
            jobs: AtomicU64::new(0),
            experts: AtomicU64::new(0),
            busy_nanos: AtomicU64::new(0),
            wait_nanos: AtomicU64::new(0),
        });

        let spin = spin_budget();
        let mut workers = Vec::with_capacity(n_threads);
        let mut threads = Vec::with_capacity(n_threads);
        for id in 0..n_threads {
            let s = Arc::clone(&shared);
            let h = thread::Builder::new()
                .name(format!("moearc-expert-{id}"))
                .stack_size(256 * 1024)
                .spawn(move || worker(&s, id, spin))
                .map_err(|_| HostError::WorkerDied)?;
            threads.push(h.thread().clone());
            workers.push(h);
        }

        Ok(Self { shared, workers, threads, weights: UnsafeCell::new(Vec::with_capacity(cap)) })
    }

    pub fn n_threads(&self) -> usize {
        self.shared.n_threads
    }

    pub fn stats(&self) -> HostStats {
        HostStats {
            jobs: self.shared.jobs.load(Ordering::Relaxed),
            experts: self.shared.experts.load(Ordering::Relaxed),
            busy_nanos: self.shared.busy_nanos.load(Ordering::Relaxed),
            wait_nanos: self.shared.wait_nanos.load(Ordering::Relaxed),
        }
    }

    pub fn reset_stats(&self) {
        self.shared.jobs.store(0, Ordering::Relaxed);
        self.shared.experts.store(0, Ordering::Relaxed);
        self.shared.busy_nanos.store(0, Ordering::Relaxed);
        self.shared.wait_nanos.store(0, Ordering::Relaxed);
    }

    /// Hand one block's host-side experts to the pool and return without waiting.
    ///
    /// `experts` is `(expert id, router weight)` in router order; `x` is the block's post-norm
    /// activation, `n_embd` long.
    pub fn submit(
        &self,
        block: usize,
        spec: BlockSpec,
        experts: &[(u16, f32)],
        x: &[f32],
    ) -> Result<HostJob, HostError> {
        let g = self.shared.geom;
        if experts.len() > g.n_expert_used {
            return Err(HostError::TooManyExperts {
                asked: experts.len(),
                capacity: g.n_expert_used,
            });
        }
        if x.len() != g.n_embd {
            return Err(HostError::Unsupported(format!(
                "activation is {} long, expected {}",
                x.len(),
                g.n_embd
            )));
        }

        // SAFETY: the previous epoch was acknowledged by every worker inside `sync`, so no
        // worker is reading the job while this writes it.
        let job = unsafe { &mut *self.shared.job.get() };
        job.spec = spec;
        job.n_experts = experts.len();
        job.banks.clear();
        job.bias.clear();
        let banks = [spec.gate, spec.up, spec.down];
        for (e, _) in experts {
            let at = |bank: usize| self.shared.table[(block * 3 + bank) * g.n_expert + *e as usize];
            let w = [at(0), at(1), at(2)];
            // 🔴 The spec is checked against the bytes it claims to describe, per bank, per
            // block. `moe.rs` always passes the block's own spec so this cannot fire there — but
            // this model quantises `ffn_down_exps` at Q6_K in half its blocks and Q4_K in the
            // rest, and a caller that reused one block's spec for another would index off the
            // end of a row inside a worker thread. A panic there is worse than a wrong answer:
            // the worker never acknowledges the epoch and `sync` waits for it forever.
            for (i, b) in banks.iter().enumerate() {
                let need = b.n_rows * b.row_bytes();
                if w[i].len != need {
                    return Err(HostError::Unsupported(format!(
                        "block {block} expert {e} bank {i} holds {} B, but the spec describes \
                         {} rows of {:?} over {} columns = {need} B",
                        w[i].len, b.n_rows, b.ty, b.n_cols
                    )));
                }
            }
            job.banks.push(w);
            if g.expert_bias {
                let bat = |bank: usize| {
                    self.shared.bias_table[(block * 3 + bank) * g.n_expert + *e as usize]
                };
                job.bias.push([bat(0), bat(1), bat(2)]);
            }
        }
        job.x.clear();
        job.x.extend_from_slice(x);
        job.xsum.clear();
        job.xsum.extend(x.chunks(32).map(|c| c.iter().sum::<f32>()));

        // A couple of chunks per thread per phase, so a straggler costs a fraction of a chunk
        // rather than a whole one.
        let target = (self.shared.n_threads * 2).max(1);
        job.chunk_a = chunk_for(experts.len() * spec.gate.n_rows, target, spec.gate.n_rows);
        job.chunk_b = chunk_for(experts.len() * spec.down.n_rows, target, spec.down.n_rows);
        job.chunks_a = spec.gate.n_rows.div_ceil(job.chunk_a);
        job.chunks_b = spec.down.n_rows.div_ceil(job.chunk_b);
        job.started = Instant::now();

        // SAFETY: single-threaded — `submit` and `sync` are the only writers and readers, both
        // on the device thread.
        let w = unsafe { &mut *self.weights.get() };
        w.clear();
        w.extend(experts.iter().map(|(_, wt)| *wt));

        let s = &self.shared;
        s.cursor_a.store(0, Ordering::Relaxed);
        s.cursor_b.store(0, Ordering::Relaxed);
        s.done_a.store(0, Ordering::Relaxed);
        s.done_b.store(0, Ordering::Relaxed);
        s.gate_b.store(false, Ordering::Relaxed);
        let epoch = s.epoch.fetch_add(1, Ordering::Release) + 1;
        for t in &self.threads {
            t.unpark();
        }
        s.jobs.fetch_add(1, Ordering::Relaxed);
        s.experts.fetch_add(experts.len() as u64, Ordering::Relaxed);
        Ok(HostJob { epoch, n_experts: experts.len() })
    }

    /// Wait for a job and write `sum over experts of weight * expert(x)` into `dst`.
    pub fn sync(&self, job: HostJob, dst: &mut [f32]) -> Result<(), HostError> {
        let s = &self.shared;
        let waited = Instant::now();
        let mut spins = 0u32;
        'wait: loop {
            for a in &s.acks {
                if a.load(Ordering::Acquire) < job.epoch {
                    if s.stop.load(Ordering::Relaxed) {
                        return Err(HostError::WorkerDied);
                    }
                    spins = spins.wrapping_add(1);
                    if spins % 8192 == 0 {
                        thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                    continue 'wait;
                }
            }
            break;
        }
        s.wait_nanos.fetch_add(waited.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if s.stop.load(Ordering::Acquire) {
            return Err(HostError::WorkerDied);
        }

        // SAFETY: every worker acknowledged this epoch with a release store that this thread has
        // acquired, so all writes to `out` are visible and no worker holds a borrow of it.
        let n = s.geom.n_embd;
        let w = unsafe { &*self.weights.get() };
        if dst.len() < n {
            return Err(HostError::Unsupported(format!(
                "destination is {} long, expected {n}",
                dst.len()
            )));
        }
        dst[..n].fill(0.0);
        for (i, weight) in w.iter().take(job.n_experts).enumerate() {
            let part = unsafe { s.out.range(i * n, n) };
            for (d, p) in dst[..n].iter_mut().zip(part) {
                *d += weight * p;
            }
        }
        Ok(())
    }
}

impl Drop for HostExecutor {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for t in &self.threads {
            t.unpark();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Rows per task: enough that scheduling is cheap, few enough that `target` tasks exist.
fn chunk_for(total_rows: usize, target: usize, rows_per_expert: usize) -> usize {
    total_rows.div_ceil(target.max(1)).max(16).min(rows_per_expert.max(1))
}

// =======================================================================================
// Workers
// =======================================================================================

fn worker(s: &Arc<Shared>, id: usize, spin: u32) {
    // Core 0 is left for the device thread; see `default_threads`.
    pin_to_core(id + 1);
    let mut seen = 0u64;
    loop {
        let mut idle = 0u32;
        loop {
            let e = s.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            if s.stop.load(Ordering::Relaxed) {
                return;
            }
            idle = idle.saturating_add(1);
            if idle > spin {
                thread::park_timeout(std::time::Duration::from_millis(50));
                idle = 0;
            } else {
                std::hint::spin_loop();
            }
        }
        if s.stop.load(Ordering::Relaxed) {
            return;
        }
        // 🔴 A worker that dies mid-job never acknowledges its epoch, and `sync` would wait for
        // that acknowledgement forever. Catching the unwind turns a hang into a reported error.
        // ⚠️ Only in a build that unwinds: the release profile sets `panic = "abort"`, where the
        // process dies instead — which is loud, and is the outcome this is second-best to.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_job(s))).is_err() {
            s.stop.store(true, Ordering::Release);
            // Acknowledged anyway, so `sync` reaches its `stop` check instead of spinning on an
            // acknowledgement from a thread that is about to disappear.
            s.acks[id].store(seen, Ordering::Release);
            return;
        }
        s.acks[id].store(seen, Ordering::Release);
    }
}

fn run_job(s: &Shared) {
    // SAFETY: `sync` did not return for the previous epoch until every worker acknowledged it,
    // so `submit`'s writes happened before the acquire load of `epoch` that woke this worker,
    // and nothing overwrites them until this worker acknowledges in turn.
    let job = unsafe { &*s.job.get() };
    if job.n_experts == 0 {
        return;
    }
    let g = s.geom;
    let total_a = job.n_experts * job.chunks_a;
    let total_b = job.n_experts * job.chunks_b;
    let per_xsum = g.n_ff.div_ceil(32);

    // ---- phase A: gate and up, fused straight into the activation --------------------------
    loop {
        let i = s.cursor_a.fetch_add(1, Ordering::Relaxed);
        if i >= total_a {
            break;
        }
        let (e, c) = (i / job.chunks_a, i % job.chunks_a);
        let r0 = c * job.chunk_a;
        let r1 = (r0 + job.chunk_a).min(job.spec.gate.n_rows);
        // SAFETY: `(e, r0..r1)` is disjoint across tasks, so no two threads write the same
        // element of `act`.
        let act = unsafe { s.act.range_mut(e * g.n_ff + r0, r1 - r0) };
        let gate = unsafe { job.banks[e][0].as_slice() };
        let up = unsafe { job.banks[e][1].as_slice() };
        // SAFETY: the bias views point into the same mapping `Shared` keeps alive, and are
        // never written.
        let (gb, ub) = match job.bias.get(e) {
            Some(b) => unsafe { (Some(b[0].as_slice()), Some(b[1].as_slice())) },
            None => (None, None),
        };
        swiglu_rows(
            job.spec.gate,
            gate,
            gb,
            job.spec.up,
            up,
            ub,
            &job.x,
            &job.xsum,
            r0,
            r1,
            act,
            g.act,
        );
        if s.done_a.fetch_add(1, Ordering::AcqRel) + 1 == total_a {
            // The last worker out of phase A prepares what phase B needs. Here rather than in
            // every phase-B task, so it stays O(experts) instead of O(tasks).
            for e in 0..job.n_experts {
                // SAFETY: every phase-A write has completed (this is the last one) and no
                // phase-B task can have started, because `gate_b` is still false.
                let a = unsafe { s.act.range(e * g.n_ff, g.n_ff) };
                let xs = unsafe { s.xsum_act.range_mut(e * per_xsum, per_xsum) };
                for (slot, c) in xs.iter_mut().zip(a.chunks(32)) {
                    *slot = c.iter().sum::<f32>();
                }
            }
            s.gate_b.store(true, Ordering::Release);
        }
    }

    // ---- the barrier -----------------------------------------------------------------------
    let mut spins = 0u32;
    while !s.gate_b.load(Ordering::Acquire) {
        spins = spins.wrapping_add(1);
        if spins % 8192 == 0 {
            thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }

    // ---- phase B: the down projection --------------------------------------------------------
    loop {
        let i = s.cursor_b.fetch_add(1, Ordering::Relaxed);
        if i >= total_b {
            break;
        }
        let (e, c) = (i / job.chunks_b, i % job.chunks_b);
        let r0 = c * job.chunk_b;
        let r1 = (r0 + job.chunk_b).min(job.spec.down.n_rows);
        // SAFETY: phase A is complete (the barrier above), so `act` and `xsum_act` are stable;
        // `out`'s `(e, r0..r1)` is disjoint across tasks.
        let a = unsafe { s.act.range(e * g.n_ff, g.n_ff) };
        let axs = unsafe { s.xsum_act.range(e * per_xsum, per_xsum) };
        let o = unsafe { s.out.range_mut(e * g.n_embd + r0, r1 - r0) };
        let down = unsafe { job.banks[e][2].as_slice() };
        // SAFETY: as above.
        let db = job.bias.get(e).map(|b| unsafe { b[2].as_slice() });
        matvec_rows(job.spec.down, down, db, a, axs, r0, r1, o);
        if s.done_b.fetch_add(1, Ordering::AcqRel) + 1 == total_b {
            s.busy_nanos.fetch_add(job.started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

/// Pin this thread to one core. Best effort: a failure costs scheduling quality, not
/// correctness, so it is not reported.
#[cfg(target_os = "linux")]
fn pin_to_core(core: usize) {
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    let n = thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
    let core = core % n;
    let mut mask = [0u64; 16];
    mask[core / 64] |= 1u64 << (core % 64);
    // SAFETY: `mask` is exactly the size passed, and pid 0 names the calling thread.
    unsafe {
        sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr());
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_to_core(_core: usize) {}

// =======================================================================================
// The arithmetic
// =======================================================================================
//
// Both kernels below fold the dequantisation into the reduction rather than expanding a row
// into floats first. The K-quants make that cheap: a Q4_K sub-block is `d*sc*q - dmin*m`, so
// `sum(w*x)` over 32 elements is `d*sc*sum(q*x) - dmin*m*sum(x)`, and `sum(x)` depends only on
// the activation — computed once per job in `submit`, not once per row. Q6_K has no minimum at
// all, so its 16-element groups reduce to `d * sum over groups of sc*sum(q*x)`.
//
// The transcription of each format is `moearc_kernels::reference`'s, term for term; that module
// is the executable specification and is itself checked against llama.cpp's own output.

/// One f16 out of a block header.
#[inline(always)]
fn ld_f16(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// `get_scale_min_k4` from `ggml/src/ggml-quants.c`, returning `(scale, min)`.
#[inline(always)]
fn scale_min_k4(q: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// `dot(row, x)` for a Q4_K row of `nb` super-blocks. `xsum` holds `x` summed in 32s.
#[inline(always)]
fn dot_q4k(w: &[u8], x: &[f32], xsum: &[f32], nb: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..nb {
        let blk = &w[b * 144..b * 144 + 144];
        let d = ld_f16(blk, 0);
        let dmin = ld_f16(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let xb = &x[b * 256..b * 256 + 256];
        let xs = &xsum[b * 8..b * 8 + 8];
        for j in 0..4 {
            let q = &qs[j * 32..j * 32 + 32];
            let xlo = &xb[j * 64..j * 64 + 32];
            let xhi = &xb[j * 64 + 32..j * 64 + 64];
            let (sc1, m1) = scale_min_k4(scales, 2 * j);
            let (sc2, m2) = scale_min_k4(scales, 2 * j + 1);
            // 🔴 Eight independent accumulators, not one. A float sum is not associative, so a
            // single `s1 +=` is a serial dependency chain LLVM is **not allowed** to reorder,
            // and it vectorises to nothing: the first version of this loop ran at 1.96 GB/s a
            // core against the ~22.8 GB/s the core reads memory at. Lane `l` is its own chain,
            // which is a reassociation the source performs rather than one the compiler assumes.
            let mut a1 = [0.0f32; 8];
            let mut a2 = [0.0f32; 8];
            for c in 0..4 {
                let qc = &q[c * 8..c * 8 + 8];
                let xl = &xlo[c * 8..c * 8 + 8];
                let xh = &xhi[c * 8..c * 8 + 8];
                for l in 0..8 {
                    a1[l] += f32::from(qc[l] & 0x0F) * xl[l];
                    a2[l] += f32::from(qc[l] >> 4) * xh[l];
                }
            }
            let s1: f32 = a1.iter().sum();
            let s2: f32 = a2.iter().sum();
            acc += d * f32::from(sc1) * s1 - dmin * f32::from(m1) * xs[2 * j];
            acc += d * f32::from(sc2) * s2 - dmin * f32::from(m2) * xs[2 * j + 1];
        }
    }
    acc
}

/// `dot(row, x)` for a Q6_K row of `nb` super-blocks.
#[inline(always)]
fn dot_q6k(w: &[u8], x: &[f32], nb: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..nb {
        let blk = &w[b * 210..b * 210 + 210];
        let d = ld_f16(blk, 208);
        let xb = &x[b * 256..b * 256 + 256];
        let mut tot = 0.0f32;
        for n in 0..2 {
            let ql = &blk[n * 64..n * 64 + 64];
            let qh = &blk[128 + n * 32..128 + n * 32 + 32];
            let sc = &blk[192 + n * 8..192 + n * 8 + 8];
            let y = &xb[n * 128..n * 128 + 128];
            // `is = l / 16` in the reference: the two halves of the 32-element strip take
            // different scales, so they are two loops rather than a branch in one.
            for half in 0..2 {
                // Eight accumulators apiece, for the reason on the Q4_K loop above.
                let mut a0 = [0.0f32; 8];
                let mut a1 = [0.0f32; 8];
                let mut a2 = [0.0f32; 8];
                let mut a3 = [0.0f32; 8];
                for c in 0..2 {
                    let base = half * 16 + c * 8;
                    for l in 0..8 {
                        let h = qh[base + l];
                        let lo = ql[base + l];
                        let hi = ql[base + l + 32];
                        let q1 = i32::from((lo & 0x0F) | ((h & 3) << 4)) - 32;
                        let q2 = i32::from((hi & 0x0F) | (((h >> 2) & 3) << 4)) - 32;
                        let q3 = i32::from((lo >> 4) | (((h >> 4) & 3) << 4)) - 32;
                        let q4 = i32::from((hi >> 4) | (((h >> 6) & 3) << 4)) - 32;
                        a0[l] += q1 as f32 * y[base + l];
                        a1[l] += q2 as f32 * y[base + l + 32];
                        a2[l] += q3 as f32 * y[base + l + 64];
                        a3[l] += q4 as f32 * y[base + l + 96];
                    }
                }
                tot += f32::from(sc[half] as i8) * a0.iter().sum::<f32>();
                tot += f32::from(sc[half + 2] as i8) * a1.iter().sum::<f32>();
                tot += f32::from(sc[half + 4] as i8) * a2.iter().sum::<f32>();
                tot += f32::from(sc[half + 6] as i8) * a3.iter().sum::<f32>();
            }
        }
        acc += d * tot;
    }
    acc
}

/// One f32 out of a bias row, read byte-wise.
///
/// The mapping is aligned enough in practice; reading it byte-wise makes that irrelevant, and
/// it costs one load per *row*, against a dot product of thousands of elements.
#[inline(always)]
fn ld_f32(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}

/// One row's dot product against MXFP4 weights.
///
/// 🔴 Simpler than the K-quants, and deliberately not written like them. There is no block
/// minimum, so there is nothing for the `xsum` trick to hoist: one block is
/// `d * sum(kvalues[q] * x)` and that is the whole of it. There is also no sub-block structure —
/// the block *is* the 32-element unit.
///
/// The two nibbles of `qs[j]` are elements `j` and `j + 16`, the **halves** of the block. The
/// two accumulators keep the halves apart so the compiler can vectorise each, and they are
/// summed at the end; the order differs from `reference::dequant` followed by a serial dot, and
/// float addition is not associative, so this is a different rounding — the same licence the
/// device kernel's tree reduction already takes.
#[inline(always)]
fn dot_mxfp4(w: &[u8], x: &[f32], nb: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..nb {
        let blk = &w[b * 17..b * 17 + 17];
        let d = e8m0_half(blk[0]);
        let xs = &x[b * 32..b * 32 + 32];
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for l in 0..16 {
            let byte = blk[1 + l];
            lo += f32::from(KVALUES_MXFP4[(byte & 0x0F) as usize]) * xs[l];
            hi += f32::from(KVALUES_MXFP4[(byte >> 4) as usize]) * xs[l + 16];
        }
        acc += d * (lo + hi);
    }
    acc
}

/// One row's dot product, dispatched on the bank's format.
///
/// 🔴 Returns NaN rather than a plausible number for a format with no host kernel.
/// [`BankSpec::check`] refuses those at construction, so this is unreachable — and if it ever
/// becomes reachable, a NaN in the logits is loud where a zero would be silent.
#[inline(always)]
fn dot_row(ty: QuantType, row: &[u8], x: &[f32], xsum: &[f32]) -> f32 {
    match ty {
        QuantType::Q4K => dot_q4k(row, x, xsum, x.len() / 256),
        QuantType::Q6K => dot_q6k(row, x, x.len() / 256),
        QuantType::Mxfp4 => dot_mxfp4(row, x, x.len() / 32),
        _ => f32::NAN,
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn swiglu_rows_body(
    gs: BankSpec,
    g: &[u8],
    gb: Option<&[u8]>,
    us: BankSpec,
    u: &[u8],
    ub: Option<&[u8]>,
    x: &[f32],
    xsum: &[f32],
    r0: usize,
    r1: usize,
    act: &mut [f32],
    activation: Activation,
) {
    let (grb, urb) = (gs.row_bytes(), us.row_bytes());
    for r in r0..r1 {
        let mut gv = dot_row(gs.ty, &g[r * grb..r * grb + grb], x, xsum);
        let mut uv = dot_row(us.ty, &u[r * urb..r * urb + urb], x, xsum);
        // 🔴 The bias goes on before the activation, which is where `ggml_add_id` sits — after
        // each `mul_mat_id` and before the GLU. Adding it afterwards would put it outside a
        // clamp and outside a sigmoid.
        if let Some(b) = gb {
            gv += ld_f32(b, r);
        }
        if let Some(b) = ub {
            uv += ld_f32(b, r);
        }
        act[r - r0] = match activation {
            // `silu(gate) * up`, exactly as `reference::swiglu` writes it.
            Activation::Swiglu => (gv / (1.0 + (-gv).exp())) * uv,
            // `reference::swiglu_oai`, term for term.
            Activation::SwigluOai { alpha, limit } => {
                let xg = gv.min(limit);
                let yu = uv.clamp(-limit, limit);
                (xg / (1.0 + (alpha * -xg).exp())) * (yu + 1.0)
            }
        };
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matvec_rows_body(
    spec: BankSpec,
    w: &[u8],
    bias: Option<&[u8]>,
    x: &[f32],
    xsum: &[f32],
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    let rb = spec.row_bytes();
    for r in r0..r1 {
        let mut v = dot_row(spec.ty, &w[r * rb..r * rb + rb], x, xsum);
        // 🔴 Inside the router's weighting: `sync` multiplies this by the expert's weight, and
        // `build_moe_ffn` adds the down bias before it multiplies too.
        if let Some(b) = bias {
            v += ld_f32(b, r);
        }
        out[r - r0] = v;
    }
}

// The AVX2/FMA wrappers exist because the crate is built for a baseline x86-64 target — SSE2,
// no FMA — and the inner loops above are exactly the shape that a 256-bit FMA doubles. Runtime
// dispatch rather than a build flag, so a release binary runs on any x86-64 and is fast on this
// one. Measured on a Core Ultra 7 265K: the AVX2 path is worth roughly 2x.
#[cfg(target_arch = "x86_64")]
fn have_avx2() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
/// # Safety
/// The target must support AVX2 and FMA; checked by [`have_avx2`].
unsafe fn swiglu_rows_avx2(
    gs: BankSpec,
    g: &[u8],
    gb: Option<&[u8]>,
    us: BankSpec,
    u: &[u8],
    ub: Option<&[u8]>,
    x: &[f32],
    xsum: &[f32],
    r0: usize,
    r1: usize,
    act: &mut [f32],
    activation: Activation,
) {
    swiglu_rows_body(gs, g, gb, us, u, ub, x, xsum, r0, r1, act, activation);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The target must support AVX2 and FMA; checked by [`have_avx2`].
#[allow(clippy::too_many_arguments)]
unsafe fn matvec_rows_avx2(
    spec: BankSpec,
    w: &[u8],
    bias: Option<&[u8]>,
    x: &[f32],
    xsum: &[f32],
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    matvec_rows_body(spec, w, bias, x, xsum, r0, r1, out);
}

/// `act[r - r0] = activation(gate_r . x + gb_r, up_r . x + ub_r)` for `r0..r1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn swiglu_rows(
    gs: BankSpec,
    g: &[u8],
    gb: Option<&[u8]>,
    us: BankSpec,
    u: &[u8],
    ub: Option<&[u8]>,
    x: &[f32],
    xsum: &[f32],
    r0: usize,
    r1: usize,
    act: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check.
        unsafe { swiglu_rows_avx2(gs, g, gb, us, u, ub, x, xsum, r0, r1, act, activation) };
        return;
    }
    swiglu_rows_body(gs, g, gb, us, u, ub, x, xsum, r0, r1, act, activation);
}

/// `out[r - r0] = row_r . x + bias_r` for `r0..r1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_rows(
    spec: BankSpec,
    w: &[u8],
    bias: Option<&[u8]>,
    x: &[f32],
    xsum: &[f32],
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check.
        unsafe { matvec_rows_avx2(spec, w, bias, x, xsum, r0, r1, out) };
        return;
    }
    matvec_rows_body(spec, w, bias, x, xsum, r0, r1, out);
}

/// `x` summed in groups of 32, which is what a Q4_K row needs to hoist its block minimum out of
/// the inner loop.
pub fn group_sums(x: &[f32]) -> Vec<f32> {
    x.chunks(32).map(|c| c.iter().sum::<f32>()).collect()
}

/// One expert, computed host-side on the calling thread: `down . (silu(gate . x) * (up . x))`.
///
/// The pool's workers compute exactly this, split across cores and two phases. It exists as one
/// function so that a test can put the host arithmetic beside the device's — on the same weights
/// and the same activation — without standing up a thread pool, and so that the threaded path
/// has something to be checked against that is not itself threaded.
pub fn expert_ffn(spec: BlockSpec, gate: &[u8], up: &[u8], down: &[u8], x: &[f32]) -> Vec<f32> {
    expert_ffn_ext(spec, gate, up, down, [None; 3], x, Activation::Swiglu)
}

/// [`expert_ffn`] with per-expert biases and a chosen activation — what gpt-oss needs.
#[allow(clippy::too_many_arguments)]
pub fn expert_ffn_ext(
    spec: BlockSpec,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    bias: [Option<&[u8]>; 3],
    x: &[f32],
    activation: Activation,
) -> Vec<f32> {
    let xs = group_sums(x);
    let mut act = vec![0.0f32; spec.gate.n_rows];
    swiglu_rows(
        spec.gate,
        gate,
        bias[0],
        spec.up,
        up,
        bias[1],
        x,
        &xs,
        0,
        spec.gate.n_rows,
        &mut act,
        activation,
    );
    let axs = group_sums(&act);
    let mut out = vec![0.0f32; spec.down.n_rows];
    matvec_rows(spec.down, down, bias[2], &act, &axs, 0, spec.down.n_rows, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use moearc_kernels::reference;

    /// A deterministic pseudo-random byte stream. Not cryptographic and not meant to be — it
    /// only has to produce quantised blocks whose scales and quants are not all alike.
    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    fn floats(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 32) as u32 as f64 / u32::MAX as f64) as f32 - 0.5
            })
            .collect()
    }

    /// A matrix of plausible blocks.
    ///
    /// 🔴 The f16 scale fields cannot be random: one 16-bit pattern in thirty-two is a NaN or an
    /// infinity, so a matrix of pure noise makes the comparison below vacuous rather than
    /// failing. Everything else in a block — the 6-bit scales, the packed quants — is arbitrary
    /// by construction and is left as noise.
    fn matrix(ty: QuantType, n_rows: usize, n_cols: usize, seed: u64) -> Vec<u8> {
        let nb = n_cols / ty.block_elems();
        let bb = ty.block_bytes();
        let mut w = bytes(n_rows * nb * bb, seed);
        let ds = floats(n_rows * nb, seed ^ 0x0d);
        for (i, d) in ds.iter().enumerate() {
            let off = i * bb;
            let mut put = |at: usize, v: f32| {
                let h = reference::f32_to_f16(v).to_le_bytes();
                w[at..at + 2].copy_from_slice(&h);
            };
            match ty {
                QuantType::Q4K => {
                    put(off, 0.01 + d.abs() * 0.05);
                    put(off + 2, 0.02 * d.abs());
                }
                QuantType::Q6K => put(off + 208, 0.001 + d.abs() * 0.01),
                _ => {}
            }
        }
        w
    }

    /// The fused host kernel against the reference's dequantise-then-dot, which is the module
    /// llama.cpp's own output is checked against. Different arithmetic on purpose — the point is
    /// how far apart the two orderings land.
    fn agrees(ty: QuantType, n_rows: usize, n_cols: usize, seed: u64) -> f32 {
        let w = matrix(ty, n_rows, n_cols, seed);
        let x = floats(n_cols, seed ^ 0x9e37);
        let xs = group_sums(&x);
        let want = reference::matvec_q(ty, &w, &x, n_rows, n_cols);
        let mut got = vec![0.0f32; n_rows];
        matvec_rows(BankSpec { ty, n_rows, n_cols }, &w, None, &x, &xs, 0, n_rows, &mut got);
        let mut worst = 0.0f32;
        for (a, b) in got.iter().zip(&want) {
            let scale = b.abs().max(1.0);
            worst = worst.max((a - b).abs() / scale);
        }
        worst
    }

    #[test]
    fn the_q4k_host_kernel_matches_the_reference() {
        for seed in [1u64, 7, 12345] {
            let e = agrees(QuantType::Q4K, 32, 2048, seed);
            assert!(e < 1e-4, "Q4_K host matvec is {e} off the reference");
        }
    }

    #[test]
    fn the_q6k_host_kernel_matches_the_reference() {
        for seed in [1u64, 7, 12345] {
            let e = agrees(QuantType::Q6K, 32, 768, seed);
            assert!(e < 1e-4, "Q6_K host matvec is {e} off the reference");
        }
    }

    #[test]
    fn a_policy_round_trips_through_the_one_parser_every_tool_uses() {
        for s in ["off", "all", "frac:0.5", "over:3"] {
            let p: HostPolicy = s.parse().expect("parses");
            assert_eq!(p.to_string().parse::<HostPolicy>().unwrap(), p);
        }
        assert!("frac:2".parse::<HostPolicy>().is_err());
        assert!("nonsense".parse::<HostPolicy>().is_err());
    }

    #[test]
    fn the_policies_split_a_blocks_misses_as_written() {
        assert_eq!(HostPolicy::Off.host_count(8), 0);
        assert_eq!(HostPolicy::Fraction(1.0).host_count(8), 8);
        assert_eq!(HostPolicy::Fraction(0.5).host_count(8), 4);
        // Rounds up, so a single miss is not silently left to the bus.
        assert_eq!(HostPolicy::Fraction(0.25).host_count(3), 1);
        assert_eq!(HostPolicy::Over(4).host_count(8), 4);
        assert_eq!(HostPolicy::Over(4).host_count(2), 0);
    }

    #[test]
    fn a_policy_knows_whether_the_ram_budget_can_back_what_it_routes() {
        use crate::host_budget::HostResidency;

        let covered =
            HostResidency { slots: 512, bytes: 0, cold_slots: 0, covers_all_misses: true };
        let cold = HostResidency { slots: 0, bytes: 0, cold_slots: 512, covers_all_misses: false };

        // 🔴 The combination this pins: `all` — every miss to the CPU — against a budget that
        // backs none of it. The executor reads expert weights straight out of the mapping, so
        // that is a drive read on the critical path, and before `backing` existed nothing in
        // the engine could tell it from two sensible settings.
        assert!(HostPolicy::Fraction(1.0).backing(cold).is_hazardous());
        // The same routing over a budget that holds the bank is the intended configuration.
        assert!(!HostPolicy::Fraction(1.0).backing(covered).is_hazardous());
        // And routing nothing cannot be the hazard however cold the budget is.
        assert!(!HostPolicy::Off.backing(cold).is_hazardous());
        assert_eq!(HostPolicy::Fraction(0.0).backing(cold), HostRouting::NotRouting);
    }
}
