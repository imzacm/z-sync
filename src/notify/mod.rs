mod listener;
mod owned_listener;
mod select;
mod state;

use alloc::rc::Rc;
use alloc::sync::Arc;
use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;
use core::task::Waker;

use num_traits::{ConstZero, NumCast};
#[cfg(feature = "triomphe-arc")]
use triomphe::Arc as TriompheArc;

pub use self::listener::NotifyListener;
pub use self::owned_listener::NotifyOwnedListener;
pub use self::select::select_blocking;
pub use self::state::*;
use crate::park_strategy::{DefaultParkStrategy, FilterOp, ParkStrategy};
use crate::waker_queue::WakerQueueLock;
use crate::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};

pub(crate) const ASYNC_CAPACITY: usize = 2;

// Async waker-storage variants. The bare `Notify{16,32,64}` names alias the default
// representation (inline: allocation-free, larger struct); the `Inline`/`Boxed` names select the
// representation explicitly. See [`WakerStorage`].

/// [`Notify16`] with inline waker storage (the default): larger struct, but allocation-free and
/// indirection-free on the async path.
pub type Notify16Inline<P = DefaultParkStrategy> =
    Notify<NotifyStateU16, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Notify32`]. See [`Notify16Inline`].
pub type Notify32Inline<P = DefaultParkStrategy> =
    Notify<NotifyStateU32, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Notify64`]. See [`Notify16Inline`].
pub type Notify64Inline<P = DefaultParkStrategy> =
    Notify<NotifyStateU64, P, InlineWakers<ASYNC_CAPACITY>>;

/// [`Notify16`] with boxed waker storage: pointer-sized struct that allocates its waker queue
/// lazily (and never at all for blocking-only usage).
pub type Notify16Boxed<P = DefaultParkStrategy> =
    Notify<NotifyStateU16, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Notify32`]. See [`Notify16Boxed`].
pub type Notify32Boxed<P = DefaultParkStrategy> =
    Notify<NotifyStateU32, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Notify64`]. See [`Notify16Boxed`].
pub type Notify64Boxed<P = DefaultParkStrategy> =
    Notify<NotifyStateU64, P, BoxedWakers<ASYNC_CAPACITY>>;

pub type Notify16<P = DefaultParkStrategy> = Notify16Inline<P>;
pub type Notify32<P = DefaultParkStrategy> = Notify32Inline<P>;
pub type Notify64<P = DefaultParkStrategy> = Notify64Inline<P>;

pub type Notify16Listener<'a, P = DefaultParkStrategy> = NotifyListener<'a, NotifyStateU16, P>;
pub type Notify32Listener<'a, P = DefaultParkStrategy> = NotifyListener<'a, NotifyStateU32, P>;
pub type Notify64Listener<'a, P = DefaultParkStrategy> = NotifyListener<'a, NotifyStateU64, P>;

/// A [`NotifyOwnedListener`] holding the `Notify` through an [`Rc`](alloc::rc::Rc).
pub type NotifyRcListener<
    S = NotifyStateU64,
    P = DefaultParkStrategy,
    W = InlineWakers<ASYNC_CAPACITY>,
> = NotifyOwnedListener<S, P, W, Rc<Notify<S, P, W>>>;

/// A [`NotifyOwnedListener`] holding the `Notify` through a std [`Arc`](alloc::sync::Arc).
pub type NotifyArcListener<
    S = NotifyStateU64,
    P = DefaultParkStrategy,
    W = InlineWakers<ASYNC_CAPACITY>,
> = NotifyOwnedListener<S, P, W, Arc<Notify<S, P, W>>>;

/// A [`NotifyOwnedListener`] holding the `Notify` through a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type NotifyTriompheArcListener<
    S = NotifyStateU64,
    P = DefaultParkStrategy,
    W = InlineWakers<ASYNC_CAPACITY>,
> = NotifyOwnedListener<S, P, W, TriompheArc<Notify<S, P, W>>>;

/// A lightweight notification primitive supporting both blocking and async waiters.
///
/// Designed as a drop-in replacement for `event_listener::Event`, optimised for
/// the check → listen → check → wait pattern used throughout this crate.
///
/// The implementation uses a monotonically increasing "epoch" counter.
/// A [`NotifyListener`] captures the epoch at creation time; it only completes
/// once the epoch has advanced past that snapshot, which means a notification
/// was fired *after* the listener was registered.
/// The `W` parameter selects how the async waker queue is stored — [`InlineWakers`] (the default,
/// no allocation but a larger struct) or [`BoxedWakers`] (pointer-sized, allocates lazily). It has
/// no effect on the blocking path. See [`WakerStorage`] and the [`Notify16Boxed`] aliases.
#[derive(Debug)]
pub struct Notify<S: NotifyState, P = DefaultParkStrategy, W = InlineWakers<ASYNC_CAPACITY>> {
    _marker: core::marker::PhantomData<P>,
    /// Bit layout:
    /// - 0..16: async wakers count (u16)
    /// - 16..32: parked threads count (u16)
    /// - 32..64: epoch (u32)
    state: S::Atomic,
    async_wakers: W,
}

impl<S: NotifyState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Default for Notify<S, P, W> {
    fn default() -> Self {
        Self::with_park_strategy()
    }
}

impl<S: NotifyState, W: WakerStorage<ASYNC_CAPACITY>> Notify<S, DefaultParkStrategy, W> {
    pub const fn new() -> Self {
        Self::with_park_strategy()
    }
}

impl<S: NotifyState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Notify<S, P, W> {
    pub const fn with_park_strategy() -> Self {
        Self {
            _marker: core::marker::PhantomData,
            state: S::INITIAL_ATOMIC,
            async_wakers: W::INIT,
        }
    }

    #[inline(always)]
    fn get_async_wakers(&self) -> &WakerQueueLock<ASYNC_CAPACITY> {
        self.async_wakers.queue()
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
    pub fn listener(&self) -> NotifyListener<'_, S, P, W> {
        let epoch = self.load_state(Ordering::Acquire).epoch();
        NotifyListener::new(self, epoch)
    }

    /// Creates an owned [`Rc`](alloc::rc::Rc)-backed listener.
    #[inline(always)]
    pub fn rc_listener(self: &Rc<Self>) -> NotifyRcListener<S, P, W> {
        NotifyOwnedListener::new(Rc::clone(self))
    }

    /// Creates an owned std [`Arc`](alloc::sync::Arc)-backed listener (movable across threads).
    #[inline(always)]
    pub fn arc_listener(self: &Arc<Self>) -> NotifyArcListener<S, P, W> {
        NotifyOwnedListener::new(Arc::clone(self))
    }

    /// Creates an owned `triomphe::Arc`-backed listener.
    ///
    /// This is a free-standing associated function (call `Notify::triomphe_arc_listener(&arc)`)
    /// rather than a method, because `triomphe::Arc` cannot be used as a `self` receiver on stable.
    #[cfg(feature = "triomphe-arc")]
    #[inline(always)]
    pub fn triomphe_arc_listener(this: &TriompheArc<Self>) -> NotifyTriompheArcListener<S, P, W> {
        NotifyOwnedListener::new(TriompheArc::clone(this))
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

    #[test]
    fn arc_listener_moves_across_threads_and_wakes() {
        use std::time::Duration;

        // The owned Arc listener borrows nothing, so it can be moved into a spawned thread.
        let notify = Arc::new(Notify32::new());
        let listener = notify.arc_listener();

        let waiter = std::thread::spawn(move || listener.wait());
        std::thread::sleep(Duration::from_millis(20));
        notify.notify(1);
        waiter.join().unwrap();
    }

    #[cfg(feature = "triomphe-arc")]
    #[test]
    fn triomphe_arc_listener_moves_across_threads_and_wakes() {
        use std::time::Duration;

        let notify = triomphe::Arc::new(Notify32::new());
        let listener = Notify32::triomphe_arc_listener(&notify);

        let waiter = std::thread::spawn(move || listener.wait());
        std::thread::sleep(Duration::from_millis(20));
        notify.notify(1);
        waiter.join().unwrap();
    }

    #[test]
    fn rc_listener_still_works() {
        let notify = Rc::new(Notify32::new());
        let listener: NotifyRcListener<_> = notify.rc_listener();
        assert!(!listener.is_notified());
        notify.notify(1);
        assert!(listener.is_notified());
    }

    #[test]
    fn boxed_storage_is_pointer_sized_and_smaller_than_inline() {
        // The boxed variant keeps the notify small (state word + one pointer); the inline default
        // trades size for allocation-free async waking.
        assert_eq!(size_of::<Notify32Boxed>(), size_of::<usize>() * 2);
        assert!(size_of::<Notify32Boxed>() < size_of::<Notify32>());
    }

    #[tokio::test]
    async fn boxed_storage_async_and_blocking_work() {
        let notify = Arc::new(Notify32Boxed::new());

        let listener = notify.listener();
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
