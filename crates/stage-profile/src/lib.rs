//! Flat stage timers for Rust search paths.
//!
//! Call [`begin`] before a profiled search, wrap stages with [`StageGuard`]
//! (or the `stage!` macro), optionally attach numerical [`StageGuard::set`]
//! metadata, then [`take`] to drain samples. When collection is off, guards
//! are no-ops.
//!
//! # Threading
//!
//! The sample buffer is **process-global**, so stages recorded on worker
//! threads are drained by a [`take`] on any thread. This matters because
//! backends routinely move the search off the caller: fast-plaid dispatches
//! through a `ThreadPoolExecutor` even for a single device, and tachiom's
//! `batch_search` fans out over Rayon. A thread-local buffer returned empty or
//! partial profiles in both cases.
//!
//! The trade-off is that a session is global too: concurrent profiled searches
//! from different threads land in one buffer, and stages repeated across
//! queries or workers appear as repeated samples in completion order. Consumers
//! are expected to coalesce by name (PyLate's `spans_from_rust` does, marking
//! the result amortised); use one query per begin→search→take when you want
//! latency-grade percentiles.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);

// Process-global so worker-thread stages survive the drain — see module docs.
// Contention is negligible: stages are coarse and only locked at completion.
static STAGES: Mutex<Vec<StageSample>> = Mutex::new(Vec::new());

/// One completed stage sample (duration + optional numerical metadata).
#[derive(Debug, Clone)]
pub struct StageSample {
    pub name: String,
    pub dur_ns: u64,
    /// Where the stage ran (`"cpu"` / `"cuda"`), for the Span `device` field.
    ///
    /// This crate only *carries* the label — it has no accelerator dependency
    /// and never synchronises. A GPU consumer must bracket its own guards with
    /// a device sync (see the fast-plaid `stage_shim`), otherwise the sample
    /// times kernel *launch* rather than execution.
    pub device: &'static str,
    /// Arbitrary numerical annotations (e.g. `n_candidates`, `k_docs`).
    pub meta: HashMap<String, f64>,
}

/// Start a collection session, clearing any samples left from a previous one.
///
/// The session is process-global: stages recorded on worker threads count.
pub fn begin() {
    ENABLED.store(true, Ordering::Relaxed);
    buffer(|samples| samples.clear());
}

/// Stop collecting and return every sample recorded since [`begin`], from all
/// threads, in completion order.
pub fn take() -> Vec<StageSample> {
    ENABLED.store(false, Ordering::Relaxed);
    let mut drained = Vec::new();
    buffer(|samples| drained = std::mem::take(samples));
    drained
}

/// Run `f` against the global buffer, recovering from a poisoned lock.
///
/// A panic while recording must not disable profiling for the rest of the
/// process, so a poisoned mutex is taken over rather than propagated.
fn buffer(f: impl FnOnce(&mut Vec<StageSample>)) {
    match STAGES.lock() {
        Ok(mut samples) => f(&mut samples),
        Err(poisoned) => f(&mut poisoned.into_inner()),
    }
}

/// Whether a profiling session is currently active.
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// RAII timer: records a [`StageSample`] into the thread-local buffer on drop
/// when a session is active.
pub struct StageGuard {
    name: &'static str,
    device: &'static str,
    start: Option<Instant>,
    meta: HashMap<&'static str, f64>,
}

impl StageGuard {
    /// Begin timing a CPU stage `name` if collection is enabled.
    #[inline]
    pub fn new(name: &'static str) -> Self {
        Self::new_on(name, "cpu")
    }

    /// Begin timing `name`, labelling it as running on `device` (`"cpu"` /
    /// `"cuda"`).
    ///
    /// The label is recorded verbatim; synchronising a GPU device around the
    /// guard is the caller's job.
    #[inline]
    pub fn new_on(name: &'static str, device: &'static str) -> Self {
        let start = ENABLED.load(Ordering::Relaxed).then(Instant::now);
        Self {
            name,
            device,
            start,
            meta: HashMap::new(),
        }
    }

    /// Attach a numerical metadata field (no-op when collection is off).
    #[inline]
    pub fn set(&mut self, key: &'static str, value: f64) -> &mut Self {
        if self.start.is_some() {
            self.meta.insert(key, value);
        }
        self
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        if let Some(start) = self.start.take() {
            let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let meta = self
                .meta
                .drain()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            buffer(|samples| {
                samples.push(StageSample {
                    name: self.name.to_string(),
                    dur_ns: ns,
                    device: self.device,
                    meta,
                })
            });
        }
    }
}

/// Expand to `StageGuard::new($name)` (bind with `let mut _stage = stage!(...);`).
///
/// The two-argument form labels the stage's device: `stage!("rerank", "cuda")`.
#[macro_export]
macro_rules! stage {
    ($name:expr) => {
        $crate::StageGuard::new($name)
    };
    ($name:expr, $device:expr) => {
        $crate::StageGuard::new_on($name, $device)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // The session and buffer are process-global, so tests must not overlap.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn begin_stage_take_records_duration_and_meta() {
        let _lock = serial();
        begin();
        {
            let mut g = StageGuard::new("rerank");
            g.set("n_candidates", 42.0).set("k", 100.0);
            thread::sleep(Duration::from_millis(1));
        }
        let samples = take();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].name, "rerank");
        assert!(samples[0].dur_ns > 0);
        assert_eq!(samples[0].meta.get("n_candidates"), Some(&42.0));
        assert_eq!(samples[0].meta.get("k"), Some(&100.0));
        assert_eq!(samples[0].device, "cpu");
        assert!(!is_enabled());
    }

    #[test]
    fn device_label_is_carried_through() {
        let _lock = serial();
        begin();
        drop(StageGuard::new_on("exact_score", "cuda"));
        let samples = take();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].device, "cuda");
    }

    #[test]
    fn guard_is_noop_when_not_begun() {
        let _lock = serial();
        let _ = take(); // ensure disabled
        {
            let mut g = StageGuard::new("skipped");
            g.set("n_candidates", 99.0);
            thread::sleep(Duration::from_millis(1));
        }
        assert!(take().is_empty());
    }

    #[test]
    fn worker_thread_samples_are_drained_by_the_caller() {
        // The case that motivated a global buffer: backends dispatch the search
        // onto a pool thread (fast-plaid) or fan out over Rayon (tachiom), then
        // drain from the calling thread.
        let _lock = serial();
        begin();
        thread::spawn(|| {
            drop(StageGuard::new("exact_score"));
        })
        .join()
        .expect("worker panicked");
        let samples = take();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].name, "exact_score");
    }
}
