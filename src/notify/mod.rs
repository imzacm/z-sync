mod listener;
mod select;
mod waker_queue;

use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::Waker;

use crossbeam_utils::CachePadded;
use parking_lot::Mutex;
use parking_lot_core::DEFAULT_UNPARK_TOKEN;

pub use self::listener::NotifyListener;
pub use self::select::select_blocking;
use self::waker_queue::WakerQueue;

const ASYNC_CAPACITY: usize = 2;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct NotifyState(u64);

impl NotifyState {
    #[inline(always)]
    const fn epoch(self) -> u32 {
        (self.0 >> 32) as u32
    }

    #[inline(always)]
    const fn parked(self) -> u16 {
        (self.0 >> 16) as u16
    }

    #[inline(always)]
    const fn wakers(self) -> u16 {
        self.0 as u16
    }
}

/// A lightweight notification primitive supporting both blocking and async waiters.
///
/// Designed as a drop-in replacement for `event_listener::Event`, optimised for
/// the check → listen → check → wait pattern used throughout this crate.
///
/// The implementation uses a monotonically increasing "epoch" counter.
/// A [`NotifyListener`] captures the epoch at creation time; it only completes
/// once the epoch has advanced past that snapshot, which means a notification
/// was fired *after* the listener was registered.
#[derive(Debug)]
pub struct Notify {
    /// Bit layout:
    /// - 0..16: async wakers count (u16)
    /// - 16..32: parked threads count (u16)
    /// - 32..64: epoch (u32)
    state: CachePadded<AtomicU64>,
    async_wakers: Mutex<WakerQueue<ASYNC_CAPACITY>>,
}

impl Default for Notify {
    fn default() -> Self {
        Self::new()
    }
}

impl Notify {
    pub fn new() -> Self {
        Self {
            state: CachePadded::new(AtomicU64::new(0)),
            async_wakers: Mutex::new(WakerQueue::new()),
        }
    }

    #[inline(always)]
    fn load_state(&self, ordering: Ordering) -> NotifyState {
        let value = self.state.load(ordering);
        NotifyState(value)
    }

    #[inline(always)]
    fn add_parkers(&self, n: u16, ordering: Ordering) {
        self.state.fetch_add((n as u64) << 16, ordering);
    }

    #[inline(always)]
    fn add_wakers(&self, n: u16, ordering: Ordering) {
        self.state.fetch_add(n as u64, ordering);
    }

    #[inline(always)]
    fn sub_parkers(&self, n: u16, ordering: Ordering) {
        self.state.fetch_sub((n as u64) << 16, ordering);
    }

    #[inline(always)]
    fn sub_wakers(&self, n: u16, ordering: Ordering) {
        self.state.fetch_sub(n as u64, ordering);
    }

    /// Creates a listener that captures the current epoch.
    ///
    /// Typical use:
    /// ```ignore
    /// let listener = notify.listener();
    /// // re-check your condition here
    /// listener.wait();   // or  listener.await
    /// ```
    #[inline(always)]
    pub fn listener(&self) -> NotifyListener<'_> {
        let epoch = self.load_state(Ordering::Acquire).epoch();
        NotifyListener::new(self, epoch)
    }

    /// Wake up to `n` waiting tasks/threads.
    ///
    /// Semantics: advances the epoch, then wakes at most `n` waiters
    /// (a mix of async wakers and parked threads).
    pub fn notify(&self, n: usize) {
        if n == 0 {
            return;
        }

        // Increment epoch and read counts at same time.
        //
        // Incrementing the epoch by 1 adds `1 << 32` to the underlying u64.
        // This leaves the lower 32 bits (parked and wakers) perfectly intact.
        let state = NotifyState(self.state.fetch_add(1 << 32, Ordering::Release));

        let mut remaining = n;

        // Wake async waiters first (cheaper than syscalls).
        if state.wakers() > 0 {
            remaining = self.wake_async(remaining);
        }

        // Wake blocked threads only if any are actually parked.
        if state.parked() > 0 && remaining > 0 {
            self.wake_blocking(remaining);
        }
    }

    /// Returns remaining.
    fn wake_async(&self, mut remaining: usize) -> usize {
        const BATCH_SIZE: usize = 32;

        loop {
            let mut popped = 0;

            // Bypass the 512-byte memset overhead completely.
            let mut wakers: [MaybeUninit<Waker>; BATCH_SIZE] =
                [const { MaybeUninit::uninit() }; BATCH_SIZE];

            {
                let mut queue = self.async_wakers.lock();
                while remaining > 0 && popped < BATCH_SIZE {
                    let Some(waker) = queue.pop_and_take() else { break };
                    wakers[popped].write(waker);
                    popped += 1;
                    remaining -= 1;
                }
            }

            if popped == 0 {
                break;
            }

            self.sub_wakers(popped as u16, Ordering::SeqCst);

            for waker in &mut wakers[..popped] {
                // SAFETY: We explicitly initialized exactly `popped` elements
                // inside the mutex lock above.
                unsafe {
                    waker.assume_init_read().wake();
                }
            }

            if remaining == 0 {
                break;
            }
        }

        remaining
    }

    /// Returns remaining.
    fn wake_blocking(&self, n: usize) -> usize {
        let mut remaining = n;

        let key = self.parking_key();
        if remaining == usize::MAX {
            let unparked = unsafe { parking_lot_core::unpark_all(key, DEFAULT_UNPARK_TOKEN) };
            remaining = remaining.saturating_sub(unparked);
            return remaining;
        }

        let mut unparked = 0;
        unsafe {
            parking_lot_core::unpark_filter(
                key,
                |_| {
                    if unparked < n {
                        unparked += 1;
                        parking_lot_core::FilterOp::Unpark
                    } else {
                        parking_lot_core::FilterOp::Stop
                    }
                },
                |_| DEFAULT_UNPARK_TOKEN,
            );
        }

        remaining - unparked
    }

    /// The address used as the parking key.
    #[inline(always)]
    fn parking_key(&self) -> usize {
        core::ptr::from_ref(&self.state) as usize
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn test_async() {
        let notify = Arc::new(Notify::new());

        let listener = notify.listener();
        assert!(!listener.is_notified());

        let notify_clone = notify.clone();
        tokio::spawn(async move {
            notify_clone.notify(1);
        });

        listener.await;

        notify.notify(1);
        let listener = notify.listener();
        assert!(!listener.is_notified());
        notify.notify(1);
        assert!(listener.is_notified());
    }
}
