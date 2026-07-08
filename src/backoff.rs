//! Adaptive spin backoff primitives ([`SpinWait`] and [`Backoff`]).
//!
//! Every blocking primitive in this crate does a short bout of optimistic spinning before it falls
//! back to parking (or, for microscopic critical sections, spins outright). That spinning is not a
//! naive busy loop: it emits an *exponentially growing burst* of [`spin_loop`](core::hint::spin_loop)
//! hints, doubling the burst each round up to a cap. A tiny first burst keeps latency low when the
//! contended resource frees almost immediately; the growth throttles bus traffic when it does not,
//! and the cap stops the burst from growing so large that it overshoots the release badly.
//!
//! These two types promote that logic — previously duplicated across [`Lock`](crate::Lock),
//! [`Semaphore`](crate::Semaphore), [`SeqLock`](crate::SeqLock), and the internal waker-queue
//! spinlock — into a public utility, so callers can build their own primitives on the exact same
//! arch-tuned schedule (cf. [`crossbeam_utils::Backoff`](https://docs.rs/crossbeam-utils) and
//! [`parking_lot`]'s `SpinWait`).
//!
//! Pick by whether the wait is bounded:
//!
//! - [`SpinWait`] — **bounded**. Spins a fixed budget of rounds, then reports exhaustion so the
//!   caller can park/block. This is the shape used by the `Lock`/`Semaphore` acquire paths, where a
//!   thread that loses the spin race should stop burning cycles and sleep.
//! - [`Backoff`] — **unbounded**. Keeps spinning with the same growing-then-capped burst forever,
//!   never yielding. This is the shape used by [`SeqLock`](crate::SeqLock)'s writer and the
//!   waker-queue spinlock, whose critical sections are a handful of instructions — parking would
//!   cost far more than spinning through them.
//!
//! Neither type ever calls the OS scheduler (`sched_yield`/`thread::yield_now`): benchmarking on
//! this crate's primitives found yielding consistently *hurt*, so the schedule is pure spin. A
//! caller that wants to yield or sleep should do so itself once [`SpinWait::spin`] returns `false`
//! (or [`Backoff::is_completed`] returns `true`).
//!
//! [`parking_lot`]: https://docs.rs/parking_lot

// The burst doubles 1, 2, 4, … up to its cap, so `spin()` on a saturated backoff emits `CAP`
// `spin_loop()` hints per call. The two caps below match the values these loops were hand-tuned to
// before they were unified here.
//
// The `cfg` split is kept even though both arms are currently equal: non-x86 tuning may diverge
// later (see the note in `lock/mod.rs`), and keeping the seam here means it changes in one place.

/// Burst cap for [`SpinWait`] (the bounded, park-eventually schedule).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SPIN_WAIT_CAP: u32 = 32;
/// Number of spin rounds [`SpinWait`] performs before reporting exhaustion.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SPIN_WAIT_ROUNDS: u32 = 64;
/// Burst cap for [`Backoff`] (the unbounded, spin-forever schedule).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const BACKOFF_CAP: u32 = 64;

// Non-x86 tuning, aligned with the x86 values on the hypothesis that high-performance Arm cores
// behave more like x86 than like weak/in-order cores here. Sweep individually if results diverge.
/// Burst cap for [`SpinWait`] (the bounded, park-eventually schedule).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const SPIN_WAIT_CAP: u32 = 32;
/// Number of spin rounds [`SpinWait`] performs before reporting exhaustion.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const SPIN_WAIT_ROUNDS: u32 = 64;
/// Burst cap for [`Backoff`] (the unbounded, spin-forever schedule).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const BACKOFF_CAP: u32 = 64;

/// A bounded exponential-backoff spinner: spin optimistically for a fixed budget, then park.
///
/// `SpinWait` drives the "try, then spin, then eventually block" loop used by the crate's
/// [`Lock`](crate::Lock) and [`Semaphore`](crate::Semaphore) acquire paths. Call [`spin`] after each
/// failed attempt; it emits a growing burst of [`spin_loop`](core::hint::spin_loop) hints and returns
/// `true` while spinning is still worthwhile, or `false` once the budget is spent — the cue to park
/// the thread (or register a waker) instead of spinning further.
///
/// The burst doubles each round (1, 2, 4, …, capped) and the spinner gives up after a fixed number
/// of rounds. Both are the values the crate's own locks were tuned to.
///
/// [`spin`]: SpinWait::spin
///
/// # Example
///
/// ```
/// use core::sync::atomic::{AtomicBool, Ordering};
/// use z_sync::SpinWait;
///
/// fn try_lock(locked: &AtomicBool) -> bool {
///     locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
/// }
///
/// fn lock(locked: &AtomicBool) {
///     let mut spin = SpinWait::new();
///     loop {
///         if try_lock(locked) {
///             return;
///         }
///         if !spin.spin() {
///             // Spun our whole budget without success: a real implementation would park here.
///             # break;
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SpinWait {
    /// Length of the next `spin_loop` burst (1, 2, 4, … up to [`SPIN_WAIT_CAP`]).
    burst: u32,
    /// Rounds spun so far; the spinner reports exhaustion at [`SPIN_WAIT_ROUNDS`].
    rounds: u32,
}

impl SpinWait {
    /// Creates a fresh spinner with the smallest burst.
    #[inline(always)]
    pub const fn new() -> Self {
        Self { burst: 1, rounds: 0 }
    }

    /// Resets the spinner to its initial state, as if freshly [`new`](SpinWait::new)ed. Call this
    /// after making progress (e.g. re-reading a changed state word) so the next contended wait
    /// starts spinning cheaply again.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Emits one exponentially growing burst of spin hints and advances the schedule.
    ///
    /// Returns `true` if the caller should keep spinning (attempt again after this call), or `false`
    /// once the round budget is exhausted, meaning the caller should stop spinning and park/block.
    ///
    /// Once `false` has been returned, further calls keep spinning at the capped burst but continue
    /// to return `false`; [`reset`](SpinWait::reset) to start a new bounded wait.
    #[inline(always)]
    pub fn spin(&mut self) -> bool {
        for _ in 0..self.burst {
            core::hint::spin_loop();
        }
        // Bitwise shift is microscopically faster than `*= 2`.
        if self.burst < SPIN_WAIT_CAP {
            self.burst <<= 1;
        }
        if self.rounds < SPIN_WAIT_ROUNDS {
            self.rounds += 1;
        }
        self.rounds < SPIN_WAIT_ROUNDS
    }

    /// Returns `true` once the spinner has spun its full budget (the next/previous [`spin`] returned
    /// or would return `false`). Useful when the spin and the give-up test live in different places.
    ///
    /// [`spin`]: SpinWait::spin
    #[inline(always)]
    pub fn is_completed(&self) -> bool {
        self.rounds >= SPIN_WAIT_ROUNDS
    }
}

impl Default for SpinWait {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// An unbounded exponential-backoff spinner for microscopic critical sections.
///
/// `Backoff` is the schedule used by primitives that spin until they win rather than parking —
/// [`SeqLock`](crate::SeqLock)'s contended writer and the crate's internal waker-queue spinlock.
/// Their critical sections are a few instructions long, so the release is always imminent and
/// parking would cost far more than spinning through it. [`spin`](Backoff::spin) emits a growing
/// burst of [`spin_loop`](core::hint::spin_loop) hints that doubles up to a cap and then stays there,
/// keeping the waiting core parked in its own L1 cache line rather than hammering the bus.
///
/// Unlike [`SpinWait`], `Backoff` never reports "give up" — it is meant for loops with no park
/// fallback. [`is_completed`](Backoff::is_completed) is offered only as a hint (the burst has
/// saturated) for callers that *do* want to switch strategies.
///
/// # Example
///
/// ```
/// use core::sync::atomic::{AtomicBool, Ordering};
/// use z_sync::Backoff;
///
/// // A minimal spinlock over an extremely short critical section.
/// fn lock(locked: &AtomicBool) {
///     let mut backoff = Backoff::new();
///     while locked.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
///         // Spin cheaply until the (tiny) critical section releases the flag.
///         while locked.load(Ordering::Relaxed) {
///             backoff.spin();
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Backoff {
    /// Length of the next `spin_loop` burst (1, 2, 4, … up to [`BACKOFF_CAP`]).
    burst: u32,
}

impl Backoff {
    /// Creates a fresh backoff with the smallest burst.
    #[inline(always)]
    pub const fn new() -> Self {
        Self { burst: 1 }
    }

    /// Resets the backoff to its initial state, so the next spin starts cheaply again. Call this
    /// after the loop makes observable progress.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Emits one exponentially growing burst of spin hints, doubling the burst up to the cap.
    ///
    /// Intended to be called on every iteration of a spin loop that has no park fallback.
    #[inline(always)]
    pub fn spin(&mut self) {
        for _ in 0..self.burst {
            core::hint::spin_loop();
        }
        // Bitwise shift is microscopically faster than `*= 2`.
        if self.burst < BACKOFF_CAP {
            self.burst <<= 1;
        }
    }

    /// Returns `true` once the burst has grown to its cap — i.e. further spinning no longer backs
    /// off. It is a hint for callers that want to switch to yielding or parking after the burst
    /// saturates; the core spin loops in this crate ignore it and spin indefinitely.
    #[inline(always)]
    pub fn is_completed(&self) -> bool {
        self.burst >= BACKOFF_CAP
    }
}

impl Default for Backoff {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_wait_reports_exhaustion_after_budget() {
        let mut spin = SpinWait::new();
        assert!(!spin.is_completed());

        let mut rounds = 0;
        while spin.spin() {
            rounds += 1;
            assert!(rounds < 10_000, "SpinWait should terminate");
        }
        // `spin` returns `false` on the round that hits the budget, so it ran `SPIN_WAIT_ROUNDS`
        // times in total (the loop body counted every round that returned `true`).
        assert_eq!(rounds as u32, SPIN_WAIT_ROUNDS - 1);
        assert!(spin.is_completed());
    }

    #[test]
    fn spin_wait_keeps_reporting_false_after_exhaustion() {
        let mut spin = SpinWait::new();
        while spin.spin() {}
        assert!(!spin.spin());
        assert!(!spin.spin());
    }

    #[test]
    fn spin_wait_reset_restarts_budget() {
        let mut spin = SpinWait::new();
        while spin.spin() {}
        assert!(spin.is_completed());

        spin.reset();
        assert!(!spin.is_completed());
        assert!(spin.spin());
    }

    #[test]
    fn backoff_saturates_but_never_stops() {
        let mut backoff = Backoff::new();
        assert!(!backoff.is_completed());
        // Spinning many times must stay sound and eventually saturate the burst.
        for _ in 0..64 {
            backoff.spin();
        }
        assert!(backoff.is_completed());
        // Still usable after saturation.
        backoff.spin();
        assert!(backoff.is_completed());

        backoff.reset();
        assert!(!backoff.is_completed());
    }

    #[test]
    fn constructors_are_const() {
        const _SPIN: SpinWait = SpinWait::new();
        const _BACKOFF: Backoff = Backoff::new();
    }
}
