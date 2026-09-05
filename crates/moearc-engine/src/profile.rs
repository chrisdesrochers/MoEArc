//! Where a decode step's wall time goes, measured rather than argued.
//!
//! Off unless `MOEARC_PROFILE=1` is set, and off in a strong sense: a disabled [`scope`] is one
//! relaxed atomic load and no clock read at all. That matters because the thing being measured
//! is per-launch overhead — an instrument costing a microsecond a call would be a meaningful
//! fraction of what it is trying to see.
//!
//! # Why a process-global
//!
//! The device lives on a thread of its own (see [`crate::session`]) and `Model` is not reachable
//! from outside it. A static accumulator is the only place a caller on another thread can read a
//! breakdown from without threading a handle through the whole command channel — which would be
//! more code, in the hot path, for a debugging aid.
//!
//! # Reading a report
//!
//! Phases do not nest, so the seconds column sums to the measured whole. Any residue between
//! that sum and the wall clock is host work nobody attributed, and is worth knowing about.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: std::sync::Once = std::sync::Once::new();

/// Totals per phase, in first-seen order so a report reads in graph order.
static SLOTS: Mutex<Vec<Bucket>> = Mutex::new(Vec::new());

struct Bucket {
    name: &'static str,
    nanos: u128,
    calls: u64,
}

/// One phase's total.
#[derive(Debug, Clone, Copy)]
pub struct Phase {
    pub name: &'static str,
    pub seconds: f64,
    pub calls: u64,
}

/// Whether profiling is on. Reads the environment once.
pub fn enabled() -> bool {
    INIT.call_once(|| {
        let on = std::env::var("MOEARC_PROFILE").ok().as_deref() == Some("1");
        ENABLED.store(on, Ordering::Relaxed);
    });
    ENABLED.load(Ordering::Relaxed)
}

/// Add `d` to `name`'s total.
pub fn record(name: &'static str, d: Duration) {
    let Ok(mut slots) = SLOTS.lock() else { return };
    if let Some(s) = slots.iter_mut().find(|s| s.name == name) {
        s.nanos += d.as_nanos();
        s.calls += 1;
    } else {
        slots.push(Bucket { name, nanos: d.as_nanos(), calls: 1 });
    }
}

/// A running phase. Charges its elapsed time to `name` when dropped.
///
/// Dropping on an early `?` return is deliberate: the work up to the failure did happen, and a
/// guard that only recorded on success would quietly under-report a path that errors.
pub struct Scope {
    name: &'static str,
    at: Option<Instant>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(t) = self.at {
            record(self.name, t.elapsed());
        }
    }
}

/// Begin a phase. Free when profiling is off.
pub fn scope(name: &'static str) -> Scope {
    Scope { name, at: if enabled() { Some(Instant::now()) } else { None } }
}

/// Every phase seen so far, in graph order.
pub fn report() -> Vec<Phase> {
    let Ok(slots) = SLOTS.lock() else { return Vec::new() };
    slots
        .iter()
        .map(|s| Phase { name: s.name, seconds: s.nanos as f64 / 1e9, calls: s.calls })
        .collect()
}

/// Discard everything measured so far — call it after warm-up, so a report describes the steady
/// state rather than the first token, which pays a cold expert cache.
pub fn reset() {
    if let Ok(mut slots) = SLOTS.lock() {
        slots.clear();
    }
}
