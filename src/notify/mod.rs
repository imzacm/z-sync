mod listener;
mod select;
mod state;

use alloc::boxed::Box;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicPtr, Ordering};
use core::task::Waker;

use num_traits::{ConstZero, NumCast};

pub use self::listener::NotifyListener;
pub use self::select::select_blocking;
pub use self::state::*;
use crate::park_strategy::{DefaultParkStrategy, FilterOp, ParkStrategy};
use crate::waker_queue::WakerQueueLock;

const ASYNC_CAPACITY: usize = 2;

pub type Notify16<P = DefaultParkStrategy> = Notify<NotifyStateU16, P>;
pub type Notify32<P = DefaultParkStrategy> = Notify<NotifyStateU32, P>;
pub type Notify64<P = DefaultParkStrategy> = Notify<NotifyStateU64, P>;

pub type Notify16Listener<'a, P = DefaultParkStrategy> = NotifyListener<'a, NotifyStateU16, P>;
pub type Notify32Listener<'a, P = DefaultParkStrategy> = NotifyListener<'a, NotifyStateU32, P>;
pub type Notify64Listener<'a, P = DefaultParkStrategy> = NotifyListener<'a, NotifyStateU64, P>;

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
pub struct Notify<S: NotifyState, P = DefaultParkStrategy> {
    _marker: core::marker::PhantomData<P>,
    /// Bit layout:
    /// - 0..16: async wakers count (u16)
    /// - 16..32: parked threads count (u16)
    /// - 32..64: epoch (u32)
    state: S::Atomic,
    async_wakers: AtomicPtr<WakerQueueLock<ASYNC_CAPACITY>>,
}

impl<S: NotifyState, P: ParkStrategy> Default for Notify<S, P> {
    fn default() -> Self {
        Self::with_park_strategy()
    }
}

impl<S: NotifyState> Notify<S, DefaultParkStrategy> {
    pub const fn new() -> Self {
        Self::with_park_strategy()
    }
}

impl<S: NotifyState, P: ParkStrategy> Notify<S, P> {
    pub const fn with_park_strategy() -> Self {
        Self {
            _marker: core::marker::PhantomData,
            state: S::INITIAL_ATOMIC,
            async_wakers: AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    #[cold]
    fn init_waker_queue(
        ptr: &AtomicPtr<WakerQueueLock<ASYNC_CAPACITY>>,
    ) -> &WakerQueueLock<ASYNC_CAPACITY> {
        let queue = Box::into_raw(Box::new(WakerQueueLock::new()));
        match ptr.compare_exchange(
            core::ptr::null_mut(),
            queue,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => unsafe { &*queue },
            Err(existing) => {
                unsafe { drop(Box::from_raw(queue)) };
                unsafe { &*existing }
            }
        }
    }

    #[inline(always)]
    fn get_async_wakers(&self) -> &WakerQueueLock<ASYNC_CAPACITY> {
        let ptr = self.async_wakers.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe { &*ptr }
        } else {
            Self::init_waker_queue(&self.async_wakers)
        }
    }

    #[inline(always)]
    fn load_state(&self, ordering: Ordering) -> S {
        S::atomic_load(&self.state, ordering)
    }

    #[inline(always)]
    pub fn has_listeners(&self) -> bool {
        self.load_state(Ordering::Acquire).has_listeners()
    }

    #[inline(always)]
    fn add_parkers(&self, n: S::Parked, ordering: Ordering) {
        S::atomic_add_parkers(&self.state, n, ordering);
    }

    #[inline(always)]
    fn add_wakers(&self, n: S::Wakers, ordering: Ordering) {
        S::atomic_add_wakers(&self.state, n, ordering);
    }

    #[inline(always)]
    fn sub_parkers(&self, n: S::Parked, ordering: Ordering) {
        S::atomic_sub_parkers(&self.state, n, ordering);
    }

    #[inline(always)]
    fn sub_wakers(&self, n: S::Wakers, ordering: Ordering) {
        S::atomic_sub_wakers(&self.state, n, ordering);
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
    pub fn listener(&self) -> NotifyListener<'_, S, P> {
        let epoch = self.load_state(Ordering::Acquire).epoch();
        NotifyListener::new(self, epoch)
    }

    /// Wake up to `n` waiting tasks/threads.
    ///
    /// Semantics: advances the epoch, then wakes at most `n` waiters
    /// (a mix of async wakers and parked threads).
    #[inline(always)]
    pub fn notify(&self, n: usize) {
        if n == 0 {
            return;
        }

        // Increment epoch and read counts at same time.
        let state = S::atomic_inc_epoch(&self.state, Ordering::Release);

        if state.has_listeners() {
            self.notify_cold(n, state);
        }
    }

    #[cold]
    #[inline(never)]
    fn notify_cold(&self, n: usize, state: S) {
        let mut remaining = n;

        // Wake async waiters first (cheaper than syscalls).
        if state.wakers() > S::Wakers::ZERO {
            remaining = self.wake_async(remaining);
        }

        // Wake blocked threads only if any are actually parked.
        if state.parked() > S::Parked::ZERO && remaining > 0 {
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
                let mut queue = self.get_async_wakers().lock();
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

            let popped_wakers: S::Wakers = NumCast::from(popped).unwrap();
            self.sub_wakers(popped_wakers, Ordering::SeqCst);

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
            let unparked = P::unpark_all(key);
            remaining = remaining.saturating_sub(unparked);
            return remaining;
        }

        let mut unparked = 0;
        P::unpark_filter(key, || {
            if unparked < n {
                unparked += 1;
                FilterOp::Unpark
            } else {
                FilterOp::Stop
            }
        });

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
        let notify = Arc::new(Notify32::new());

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

    #[test]
    fn verify_struct_sizes() {
        assert_eq!(
            size_of::<Notify32<crate::park_strategy::ParkingLot>>(),
            size_of::<Notify32<crate::park_strategy::Spin>>()
        );
    }
}
