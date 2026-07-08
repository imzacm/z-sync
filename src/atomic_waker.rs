//! A single-waiter waker cell ([`AtomicWaker`]).
//!
//! `AtomicWaker` stores at most one [`Waker`] and lets one side register it while another side
//! wakes it, lock-free. It is the degenerate one-waiter case of the crate's [`WakerQueueLock`] — a
//! lighter building block for single-consumer async structures (e.g. a one-shot channel) where a
//! full waker queue would be overkill.
//!
//! The registering side calls [`register`](AtomicWaker::register) from its `poll`; the waking side
//! calls [`wake`](AtomicWaker::wake) (or [`take`](AtomicWaker::take)). A wake that races a
//! registration is never lost: whichever side observes the conflict delivers the wake.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::task::Waker;

/// Idle: no registration or wake in progress.
const WAITING: usize = 0;
/// A `register` call is writing the waker.
const REGISTERING: usize = 0b01;
/// A `wake`/`take` call is taking the waker.
const WAKING: usize = 0b10;

/// A cell holding at most one [`Waker`], with lock-free registration and waking.
///
/// See the [module documentation](self). Analogous to `futures::task::AtomicWaker`.
pub struct AtomicWaker {
    state: AtomicUsize,
    waker: UnsafeCell<Option<Waker>>,
}

// SAFETY: the `waker` cell is only accessed by the thread that holds the `REGISTERING` lock or the
// `WAKING` claim, and the state machine makes those mutually exclusive, so there is never
// concurrent access to the `Waker`.
unsafe impl Send for AtomicWaker {}
unsafe impl Sync for AtomicWaker {}

impl AtomicWaker {
    /// Creates an empty cell.
    pub const fn new() -> Self {
        Self { state: AtomicUsize::new(WAITING), waker: UnsafeCell::new(None) }
    }

    /// Registers `waker` to be woken by the next [`wake`](AtomicWaker::wake).
    ///
    /// Call this from the registering task's `poll` (typically after a failed readiness check, then
    /// re-check). If a wake arrives while registering, the passed waker is woken immediately so the
    /// task re-polls — no wakeup is lost. Re-registering the same waker avoids a clone.
    pub fn register(&self, waker: &Waker) {
        match self
            .state
            .compare_exchange(WAITING, REGISTERING, Ordering::Acquire, Ordering::Acquire)
            .unwrap_or_else(|actual| actual)
        {
            WAITING => {
                unsafe {
                    // We hold the REGISTERING lock: exclusive access to the waker slot.
                    let slot = &mut *self.waker.get();
                    if slot.as_ref().is_none_or(|w| !w.will_wake(waker)) {
                        *slot = Some(waker.clone());
                    }

                    // Release the lock. If a `wake` set the WAKING bit while we registered, it left
                    // delivery to us — take the waker and wake it now.
                    match self.state.compare_exchange(
                        REGISTERING,
                        WAITING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {}
                        Err(actual) => {
                            debug_assert_eq!(actual, REGISTERING | WAKING);
                            let woken = (*self.waker.get()).take();
                            self.state.swap(WAITING, Ordering::AcqRel);
                            if let Some(woken) = woken {
                                woken.wake();
                            }
                        }
                    }
                }
            }
            // A wake is in progress (or, defensively, another registration): wake the passed waker
            // so the task polls again rather than sleeping through the pending
            // readiness.
            _ => waker.wake_by_ref(),
        }
    }

    /// Takes and wakes the registered waker, if any.
    #[inline]
    pub fn wake(&self) {
        if let Some(waker) = self.take() {
            waker.wake();
        }
    }

    /// Takes the registered waker out of the cell, if one is present and no registration is racing.
    /// Returns `None` if the cell is empty or a `register`/`wake` is concurrently in progress (that
    /// party will deliver the wake).
    pub fn take(&self) -> Option<Waker> {
        match self.state.fetch_or(WAKING, Ordering::AcqRel) {
            WAITING => {
                // We won the WAKING claim with no registration in flight: the slot is ours.
                let waker = unsafe { (*self.waker.get()).take() };
                self.state.fetch_and(!WAKING, Ordering::Release);
                waker
            }
            // REGISTERING: the registering side will observe our WAKING bit and deliver.
            // WAKING already set: another taker owns delivery. Either way, nothing for us to take.
            _ => None,
        }
    }
}

impl Default for AtomicWaker {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for AtomicWaker {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AtomicWaker").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    use super::*;

    struct CountingWaker(AtomicUsize);
    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counting() -> (Waker, Arc<CountingWaker>) {
        let inner = Arc::new(CountingWaker(AtomicUsize::new(0)));
        (Waker::from(Arc::clone(&inner)), inner)
    }

    #[test]
    fn register_then_wake_delivers_once() {
        let cell = AtomicWaker::new();
        let (waker, count) = counting();
        cell.register(&waker);
        cell.wake();
        assert_eq!(count.0.load(Ordering::SeqCst), 1);
        // The cell is now empty: a second wake does nothing.
        cell.wake();
        assert_eq!(count.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn wake_without_registration_is_noop() {
        let cell = AtomicWaker::new();
        cell.wake();
        assert!(cell.take().is_none());
    }

    #[test]
    fn reregistering_replaces_the_waker() {
        let cell = AtomicWaker::new();
        let (w1, c1) = counting();
        let (w2, c2) = counting();
        cell.register(&w1);
        cell.register(&w2);
        cell.wake();
        assert_eq!(c1.0.load(Ordering::SeqCst), 0, "old waker must not be woken");
        assert_eq!(c2.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn take_returns_the_registered_waker() {
        let cell = AtomicWaker::new();
        let (waker, count) = counting();
        cell.register(&waker);
        let taken = cell.take().expect("a waker was registered");
        taken.wake();
        assert_eq!(count.0.load(Ordering::SeqCst), 1);
        assert!(cell.take().is_none());
    }
}
