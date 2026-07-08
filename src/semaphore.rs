use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};
use core::task::{Context, Poll, Waker};

use num_traits::{ConstOne, ConstZero, NumCast, ToPrimitive};

use crate::park_strategy::{DefaultParkStrategy, FilterOp, ParkStrategy};
use crate::waker_queue::{WakerQueueLock, WakerTicket};
use crate::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};

pub(crate) const ASYNC_CAPACITY: usize = 4;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const ACQUIRE_SPIN_MAX: usize = 64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SPIN_CAP: usize = 32;

// Non-x86 spin tuning: aligned with the x86 values (more spinning, no post-spin `yield_now`). See
// the matching note in `lock/mod.rs`.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const ACQUIRE_SPIN_MAX: usize = 64;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const SPIN_CAP: usize = 32;

// Async waker-storage variants. The bare `Semaphore{16,32,64}` names alias the default
// representation (boxed: small, allocates lazily); the `Boxed`/`Inline` names select the
// representation explicitly. See [`WakerStorage`].

/// [`Semaphore16`] with boxed waker storage (the default): small, allocates its waker queue lazily
/// (and never at all for blocking-only usage).
pub type Semaphore16Boxed<P = DefaultParkStrategy> =
    Semaphore<SemaphoreStateU16, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Semaphore32`]. See [`Semaphore16Boxed`].
pub type Semaphore32Boxed<P = DefaultParkStrategy> =
    Semaphore<SemaphoreStateU32, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Semaphore64`]. See [`Semaphore16Boxed`].
pub type Semaphore64Boxed<P = DefaultParkStrategy> =
    Semaphore<SemaphoreStateU64, P, BoxedWakers<ASYNC_CAPACITY>>;

/// [`Semaphore16`] with inline waker storage: allocation-free on the async path (usable without a
/// global allocator), at the cost of a larger semaphore.
pub type Semaphore16Inline<P = DefaultParkStrategy> =
    Semaphore<SemaphoreStateU16, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Semaphore32`]. See [`Semaphore16Inline`].
pub type Semaphore32Inline<P = DefaultParkStrategy> =
    Semaphore<SemaphoreStateU32, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Semaphore64`]. See [`Semaphore16Inline`].
pub type Semaphore64Inline<P = DefaultParkStrategy> =
    Semaphore<SemaphoreStateU64, P, InlineWakers<ASYNC_CAPACITY>>;

pub type Semaphore16<P = DefaultParkStrategy> = Semaphore16Boxed<P>;
pub type Semaphore32<P = DefaultParkStrategy> = Semaphore32Boxed<P>;
pub type Semaphore64<P = DefaultParkStrategy> = Semaphore64Boxed<P>;

pub trait SemaphoreState: Sized + Copy + Clone + PartialEq + Eq {
    type Atomic: core::fmt::Debug;

    type Permits: Copy + Eq + Ord + NumCast + ConstZero + ConstOne;
    type Wakers: Eq + Ord + NumCast + ConstZero + ConstOne;
    type Parked: Eq + Ord + NumCast + ConstZero + ConstOne;

    // --- State Getters ---
    fn permits(self) -> Self::Permits;
    fn wakers(self) -> Self::Wakers;
    fn parked(self) -> Self::Parked;
    fn has_waiters(self) -> bool;

    // --- State Mutations (Pure) ---
    fn sub_permits_state(self, n: Self::Permits) -> Self;

    // --- Atomic Operations ---
    fn atomic_with_permits(permits: Self::Permits) -> Self::Atomic;
    fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_compare_exchange_weak(
        atomic: &Self::Atomic,
        current: Self,
        new: Self,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self, Self>;
    fn atomic_add_permits(atomic: &Self::Atomic, n: Self::Permits, order: Ordering) -> Self;

    fn atomic_add_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering);
    fn atomic_sub_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering);
    fn atomic_add_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering);
    fn atomic_sub_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering);
}

#[macro_export]
macro_rules! atomic_semaphore_state {
    (
        $vis:vis struct $struct_name:ident(
            $atomic_ty:ident($prim_ty:ty) {
                wakers: $w_ty:ty = $w_bits:expr,
                parked: $p_ty:ty = $p_bits:expr,
                permits: $pm_ty:ty = $pm_bits:expr $(,)?
            }
        )
    ) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        $vis struct $struct_name(pub $prim_ty);

        impl $struct_name {
            pub const WAKER_SHIFT: $prim_ty = 0;
            pub const PARKER_SHIFT: $prim_ty = Self::WAKER_SHIFT + $w_bits;
            pub const PERMIT_SHIFT: $prim_ty = Self::PARKER_SHIFT + $p_bits;

            const _ASSERT_SIZE: () = assert!(
                ($w_bits + $p_bits + $pm_bits) <= <$prim_ty>::BITS as $prim_ty,
                "Total bits specified exceed the capacity of the chosen primitive type."
            );

            const fn mask(bits: $prim_ty, shift: $prim_ty) -> $prim_ty {
                if bits == 0 {
                    0
                } else if bits == <$prim_ty>::BITS as $prim_ty {
                    !0
                } else {
                    ((1 << bits) - 1) << shift
                }
            }

            pub const WAKERS_MASK: $prim_ty = Self::mask($w_bits, Self::WAKER_SHIFT);
            pub const PARKED_MASK: $prim_ty = Self::mask($p_bits, Self::PARKER_SHIFT);
            pub const PERMITS_MASK: $prim_ty = Self::mask($pm_bits, Self::PERMIT_SHIFT);
            pub const WAITERS_MASK: $prim_ty = Self::WAKERS_MASK | Self::PARKED_MASK;
        }

        impl SemaphoreState for $struct_name {
            type Atomic = $atomic_ty;
            type Permits = $pm_ty;
            type Wakers = $w_ty;
            type Parked = $p_ty;

            #[inline(always)] fn permits(self) -> Self::Permits { ((self.0 & Self::PERMITS_MASK) >> Self::PERMIT_SHIFT) as Self::Permits }
            #[inline(always)] fn wakers(self) -> Self::Wakers { ((self.0 & Self::WAKERS_MASK) >> Self::WAKER_SHIFT) as Self::Wakers }
            #[inline(always)] fn parked(self) -> Self::Parked { ((self.0 & Self::PARKED_MASK) >> Self::PARKER_SHIFT) as Self::Parked }
            #[inline(always)] fn has_waiters(self) -> bool { (self.0 & Self::WAITERS_MASK) != 0 }

            #[inline(always)] fn sub_permits_state(self, n: Self::Permits) -> Self { Self(self.0 - ((n as $prim_ty) << Self::PERMIT_SHIFT)) }

            #[inline(always)] fn atomic_with_permits(permits: Self::Permits) -> Self::Atomic { <$atomic_ty>::new((permits as $prim_ty) << Self::PERMIT_SHIFT) }
            #[inline(always)] fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.load(order)) }
            #[inline(always)] fn atomic_compare_exchange_weak(
                atomic: &Self::Atomic,
                current: Self,
                new: Self,
                success: Ordering,
                failure: Ordering,
            ) -> Result<Self, Self> {
                atomic.compare_exchange_weak(current.0, new.0, success, failure)
                    .map(Self)
                    .map_err(Self)
            }
            #[inline(always)] fn atomic_add_permits(atomic: &Self::Atomic, n: Self::Permits, order: Ordering) -> Self { Self(atomic.fetch_add((n as $prim_ty) << Self::PERMIT_SHIFT, order)) }

            #[inline(always)] fn atomic_add_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering) { atomic.fetch_add((n as $prim_ty) << Self::PARKER_SHIFT, order); }
            #[inline(always)] fn atomic_sub_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering) { atomic.fetch_sub((n as $prim_ty) << Self::PARKER_SHIFT, order); }
            #[inline(always)] fn atomic_add_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering) { atomic.fetch_add((n as $prim_ty) << Self::WAKER_SHIFT, order); }
            #[inline(always)] fn atomic_sub_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering) { atomic.fetch_sub((n as $prim_ty) << Self::WAKER_SHIFT, order); }
        }
    };
}

atomic_semaphore_state!(pub struct SemaphoreStateU64(
    AtomicU64(u64) {
        wakers: u16 = 16,
        parked: u16 = 16,
        permits: u32 = 32,
    }
));

atomic_semaphore_state!(pub struct SemaphoreStateU32(
    AtomicU32(u32) {
        wakers: u8 = 8,
        parked: u8 = 8,
        permits: u16 = 16,
    }
));

atomic_semaphore_state!(pub struct SemaphoreStateU16(
    AtomicU16(u16) {
        wakers: u8 = 4,
        parked: u8 = 4,
        permits: u16 = 8,
    }
));

/// A counting semaphore supporting both blocking and async acquisition.
///
/// Permits are handed out by [`acquire`](Semaphore::acquire) (blocking) or
/// [`acquire_async`](Semaphore::acquire_async) (async) and returned when the resulting
/// [`SemaphorePermit`] is dropped. Blocking and async waiters share the same wait state, so a
/// permit released from one side wakes a waiter on the other.
///
/// The `16`/`32`/`64` suffix selects the width of the packed atomic state word, which bounds the
/// maximum permit count and the number of simultaneously parked/waking waiters.
///
/// The `W` parameter selects how the async waker queue is stored — [`BoxedWakers`] (the default,
/// keeping the semaphore small and allocating lazily) or [`InlineWakers`] (queue stored inline:
/// larger, but allocation-free on the async path). It has no effect on the blocking path. See
/// [`WakerStorage`] and the [`Semaphore16Boxed`] / [`Semaphore16Inline`] aliases.
#[derive(Debug)]
pub struct Semaphore<
    S: SemaphoreState = SemaphoreStateU32,
    P = DefaultParkStrategy,
    W = BoxedWakers<ASYNC_CAPACITY>,
> {
    _marker: PhantomData<P>,
    /// Bit layout (`SemaphoreStateU64`):
    /// - 0..16: async wakers count (u16)
    /// - 16..32: parked threads count (u16)
    /// - 32..64: available permits (u32)
    state: S::Atomic,
    async_wakers: W,
}

impl<S: SemaphoreState, W: WakerStorage<ASYNC_CAPACITY>> Semaphore<S, DefaultParkStrategy, W> {
    pub fn new(permits: usize) -> Self {
        Self::with_park_strategy(permits)
    }
}

impl<S: SemaphoreState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Semaphore<S, P, W> {
    pub fn with_park_strategy(permits: usize) -> Self {
        let permits: S::Permits =
            NumCast::from(permits).expect("permit count exceeds semaphore capacity");
        Self {
            _marker: PhantomData,
            state: S::atomic_with_permits(permits),
            async_wakers: W::INIT,
        }
    }

    #[inline(always)]
    fn get_wakers(&self) -> &WakerQueueLock<ASYNC_CAPACITY> {
        self.async_wakers.queue()
    }

    #[inline(always)]
    fn load_state(&self, ordering: Ordering) -> S {
        S::atomic_load(&self.state, ordering)
    }

    /// Returns the number of permits currently available.
    #[inline(always)]
    pub fn available_permits(&self) -> usize {
        self.load_state(Ordering::Acquire).permits().to_usize().unwrap_or(usize::MAX)
    }

    /// Attempts to acquire a single permit without blocking.
    #[inline(always)]
    pub fn try_acquire(&self) -> Option<SemaphorePermit<'_, S, P, W>> {
        self.try_acquire_many(1)
    }

    /// Attempts to acquire `n` permits without blocking.
    ///
    /// Returns `None` if fewer than `n` permits are available (or `n` exceeds this semaphore's
    /// capacity).
    pub fn try_acquire_many(&self, n: usize) -> Option<SemaphorePermit<'_, S, P, W>> {
        let need: S::Permits = NumCast::from(n)?;
        if need == S::Permits::ZERO {
            return Some(SemaphorePermit { semaphore: self, permits: 0 });
        }

        let mut state = self.load_state(Ordering::Relaxed);
        loop {
            if state.permits() < need {
                return None;
            }

            let new = state.sub_permits_state(need);
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                new,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(SemaphorePermit { semaphore: self, permits: n }),
                Err(v) => state = v,
            }
        }
    }

    fn spin_try_acquire(&self, n: usize) -> Option<SemaphorePermit<'_, S, P, W>> {
        let need: S::Permits = NumCast::from(n)?;

        let mut backoff = 1;
        for _ in 0..ACQUIRE_SPIN_MAX {
            if self.load_state(Ordering::Relaxed).permits() >= need
                && let Some(guard) = self.try_acquire_many(n)
            {
                return Some(guard);
            }
            for _ in 0..backoff {
                core::hint::spin_loop();
            }
            if backoff < SPIN_CAP {
                backoff <<= 1;
            }
        }

        None
    }

    /// Acquires a single permit, blocking the current thread until one is available.
    #[inline(always)]
    pub fn acquire(&self) -> SemaphorePermit<'_, S, P, W> {
        self.acquire_many(1)
    }

    /// Acquires `n` permits, blocking the current thread until they are available.
    #[inline(always)]
    pub fn acquire_many(&self, n: usize) -> SemaphorePermit<'_, S, P, W> {
        if let Some(guard) = self.try_acquire_many(n) {
            return guard;
        }
        self.acquire_slow(n)
    }

    #[cold]
    #[inline(never)]
    fn acquire_slow(&self, n: usize) -> SemaphorePermit<'_, S, P, W> {
        if let Some(guard) = self.spin_try_acquire(n) {
            return guard;
        }

        let need: S::Permits = NumCast::from(n).expect("permit count exceeds semaphore capacity");

        self.add_parker(Ordering::SeqCst);

        loop {
            P::park(self.parking_key(), || self.load_state(Ordering::Acquire).permits() < need);

            if let Some(guard) = self.try_acquire_many(n) {
                self.sub_parker(Ordering::Relaxed);
                return guard;
            }
        }
    }

    /// Acquires a single permit, resolving once one is available.
    #[inline(always)]
    pub fn acquire_async(&self) -> AcquireFuture<'_, S, P, W> {
        self.acquire_many_async(1)
    }

    /// Acquires `n` permits, resolving once they are available.
    #[inline(always)]
    pub fn acquire_many_async(&self, n: usize) -> AcquireFuture<'_, S, P, W> {
        AcquireFuture { semaphore: self, n, waker_node_ticket: None }
    }

    /// Adds `n` permits back to the semaphore, waking waiters as needed.
    ///
    /// Permits acquired through a [`SemaphorePermit`] are returned automatically on drop; this is
    /// for handing out permits that were never acquired (or increasing the total).
    #[inline(always)]
    pub fn add_permits(&self, n: usize) {
        if n == 0 {
            return;
        }

        let add: S::Permits = NumCast::from(n).expect("permit count exceeds semaphore capacity");
        let state = S::atomic_add_permits(&self.state, add, Ordering::Release);

        if state.has_waiters() {
            self.wake_cold(n, state);
        }
    }

    #[cold]
    #[inline(never)]
    fn wake_cold(&self, n: usize, state: S) {
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

            // Bypass the memset overhead completely.
            let mut wakers: [MaybeUninit<Waker>; BATCH_SIZE] =
                [const { MaybeUninit::uninit() }; BATCH_SIZE];

            {
                let mut queue = self.get_wakers().lock();
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

    /// Forwards a single wakeup to the next waiter.
    ///
    /// Used when an [`AcquireFuture`] is dropped after a release already consumed its waker: the
    /// released permit is still in the pool, so we pass the wakeup along to avoid stranding it.
    fn forward_wake(&self) {
        let state = self.load_state(Ordering::Acquire);
        if !state.has_waiters() {
            return;
        }

        let mut remaining = 1;
        if state.wakers() > S::Wakers::ZERO {
            remaining = self.wake_async(remaining);
        }
        if remaining > 0 && state.parked() > S::Parked::ZERO {
            self.wake_blocking(remaining);
        }
    }

    #[inline(always)]
    fn add_parker(&self, ordering: Ordering) {
        S::atomic_add_parkers(&self.state, S::Parked::ONE, ordering);
    }

    #[inline(always)]
    fn sub_parker(&self, ordering: Ordering) {
        S::atomic_sub_parkers(&self.state, S::Parked::ONE, ordering);
    }

    #[inline(always)]
    fn add_waker(&self, ordering: Ordering) {
        S::atomic_add_wakers(&self.state, S::Wakers::ONE, ordering);
    }

    #[inline(always)]
    fn sub_waker(&self, ordering: Ordering) {
        S::atomic_sub_wakers(&self.state, S::Wakers::ONE, ordering);
    }

    #[inline(always)]
    fn sub_wakers(&self, n: S::Wakers, ordering: Ordering) {
        S::atomic_sub_wakers(&self.state, n, ordering);
    }

    /// The address used as the parking key.
    #[inline(always)]
    fn parking_key(&self) -> usize {
        core::ptr::from_ref(&self.state) as usize
    }
}

unsafe impl<S: SemaphoreState, P, W> Send for Semaphore<S, P, W> {}
unsafe impl<S: SemaphoreState, P, W> Sync for Semaphore<S, P, W> {}

/// A guard representing one or more acquired permits.
///
/// The permits are returned to the semaphore when this guard is dropped, unless
/// [`forget`](SemaphorePermit::forget) is called first.
#[derive(Debug)]
pub struct SemaphorePermit<
    'a,
    S: SemaphoreState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    semaphore: &'a Semaphore<S, P, W>,
    permits: usize,
}

impl<S: SemaphoreState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>>
    SemaphorePermit<'_, S, P, W>
{
    /// The number of permits held by this guard.
    #[inline(always)]
    pub fn permits(&self) -> usize {
        self.permits
    }

    /// Drops the guard without returning its permits to the semaphore, permanently reducing the
    /// number of available permits.
    #[inline(always)]
    pub fn forget(self) {
        core::mem::forget(self);
    }
}

impl<S: SemaphoreState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for SemaphorePermit<'_, S, P, W>
{
    fn drop(&mut self) {
        if self.permits > 0 {
            self.semaphore.add_permits(self.permits);
        }
    }
}

#[derive(Debug)]
pub struct AcquireFuture<
    'a,
    S: SemaphoreState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    semaphore: &'a Semaphore<S, P, W>,
    n: usize,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, S: SemaphoreState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Future
    for AcquireFuture<'a, S, P, W>
{
    type Output = SemaphorePermit<'a, S, P, W>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();

        if let Some(guard) = this.semaphore.try_acquire_many(this.n) {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.semaphore.get_wakers().lock().remove(ticket)
            {
                this.semaphore.sub_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        {
            let mut queue = this.semaphore.get_wakers().lock();

            if let Some(guard) = this.semaphore.try_acquire_many(this.n) {
                if let Some(ticket) = this.waker_node_ticket.take()
                    && queue.remove(ticket)
                {
                    this.semaphore.sub_waker(Ordering::Relaxed);
                }
                return Poll::Ready(guard);
            }

            if let Some(ticket) = this.waker_node_ticket {
                let node = queue.node_mut(ticket.index());

                if node.generation() == ticket.generation() {
                    if node.waker().is_none_or(|w| !w.will_wake(cx.waker())) {
                        *node.waker_mut() = Some(cx.waker().clone());
                    }
                } else {
                    // Our slot was popped and recycled by a previous wakeup. We must re-enqueue
                    // ourselves to prevent a lost wakeup.
                    this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                    this.semaphore.add_waker(Ordering::SeqCst);
                }
            } else {
                this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                this.semaphore.add_waker(Ordering::SeqCst);
            }
        }

        if let Some(guard) = this.semaphore.try_acquire_many(this.n) {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.semaphore.get_wakers().lock().remove(ticket)
            {
                this.semaphore.sub_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        Poll::Pending
    }
}

impl<S: SemaphoreState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for AcquireFuture<'_, S, P, W>
{
    fn drop(&mut self) {
        let Some(ticket) = self.waker_node_ticket.take() else { return };

        if self.semaphore.get_wakers().lock().remove(ticket) {
            self.semaphore.sub_waker(Ordering::Relaxed);
        } else {
            // Our waker was already consumed by a release that intended to wake us. As we're being
            // dropped before acquiring, forward the wakeup so the released permit isn't stranded.
            self.semaphore.forward_wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;
    use std::vec::Vec;

    // Use SemaphoreStateU64 directly under the hood to preserve all test cases identically.
    type Semaphore = super::Semaphore<SemaphoreStateU64>;

    use super::*;

    // -------------------------------------------------------------------------
    // Blocking tests
    // -------------------------------------------------------------------------

    #[test]
    fn try_acquire_reduces_available_permits() {
        let sem = Semaphore::new(2);
        let a = sem.try_acquire().unwrap();
        assert_eq!(sem.available_permits(), 1);
        let b = sem.try_acquire().unwrap();
        assert_eq!(sem.available_permits(), 0);
        drop((a, b));
    }

    #[test]
    fn try_acquire_fails_when_exhausted() {
        let sem = Semaphore::new(1);
        let _a = sem.try_acquire().unwrap();
        assert!(sem.try_acquire().is_none());
    }

    #[test]
    fn permits_returned_on_drop() {
        let sem = Semaphore::new(1);
        {
            let _a = sem.acquire();
            assert_eq!(sem.available_permits(), 0);
        }
        assert_eq!(sem.available_permits(), 1);
    }

    #[test]
    fn forget_keeps_permits() {
        let sem = Semaphore::new(1);
        sem.acquire().forget();
        assert_eq!(sem.available_permits(), 0);
        assert!(sem.try_acquire().is_none());
    }

    #[test]
    fn try_acquire_many_all_or_nothing() {
        let sem = Semaphore::new(3);
        assert!(sem.try_acquire_many(4).is_none());
        let g = sem.try_acquire_many(3).unwrap();
        assert_eq!(sem.available_permits(), 0);
        drop(g);
        assert_eq!(sem.available_permits(), 3);
    }

    #[test]
    fn blocking_acquire_waits_for_release() {
        let sem = Arc::new(Semaphore::new(1));
        let held = sem.acquire();

        let sem2 = Arc::clone(&sem);
        let waiter = std::thread::spawn(move || {
            let _g = sem2.acquire();
        });

        // Give the waiter time to park.
        std::thread::sleep(Duration::from_millis(20));
        drop(held);

        waiter.join().unwrap();
        assert_eq!(sem.available_permits(), 1);
    }

    #[test]
    fn concurrent_blocking_acquire_release_is_bounded() {
        const THREADS: usize = 8;
        const OPS: usize = 200;
        const PERMITS: usize = 3;

        let sem = Arc::new(Semaphore::new(PERMITS));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let s = Arc::clone(&sem);
                std::thread::spawn(move || {
                    for _ in 0..OPS {
                        let _g = s.acquire();
                        assert!(s.available_permits() < PERMITS);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(sem.available_permits(), PERMITS);
    }

    // -------------------------------------------------------------------------
    // Async tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn async_acquire_returns_permit() {
        let sem = Semaphore::new(1);
        let g = sem.acquire_async().await;
        assert_eq!(sem.available_permits(), 0);
        drop(g);
        assert_eq!(sem.available_permits(), 1);
    }

    #[tokio::test]
    async fn async_acquire_waits_for_release() {
        let sem = Arc::new(Semaphore::new(1));
        let held = sem.acquire();

        let sem2 = Arc::clone(&sem);
        let waiter = tokio::spawn(async move {
            let _g = sem2.acquire_async().await;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(held);

        waiter.await.unwrap();
        assert_eq!(sem.available_permits(), 1);
    }

    #[tokio::test]
    async fn async_concurrent_acquire_release_is_bounded() {
        const TASKS: usize = 8;
        const OPS: usize = 200;
        const PERMITS: usize = 3;

        let sem = Arc::new(Semaphore::new(PERMITS));

        let handles: Vec<_> = (0..TASKS)
            .map(|_| {
                let s = Arc::clone(&sem);
                tokio::spawn(async move {
                    for _ in 0..OPS {
                        let _g = s.acquire_async().await;
                        assert!(s.available_permits() < PERMITS);
                    }
                })
            })
            .collect();

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(sem.available_permits(), PERMITS);
    }

    #[tokio::test]
    async fn dropped_acquire_future_does_not_leak_waker_count() {
        let sem = Arc::new(Semaphore::new(1));

        // Hold the only permit so acquire_async will park.
        let held = sem.acquire();

        let sem2 = Arc::clone(&sem);
        let fut = tokio::spawn(async move {
            // select! drops the losing future, exercising AcquireFuture::drop.
            tokio::select! {
                _g = sem2.acquire_async() => {},
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        });

        fut.await.unwrap();
        drop(held);

        assert_eq!(
            sem.load_state(Ordering::Relaxed).wakers(),
            0,
            "waker count leaked after AcquireFuture was dropped"
        );
        assert_eq!(sem.available_permits(), 1);
    }

    #[tokio::test]
    async fn mixed_blocking_and_async_acquire_is_bounded() {
        const ASYNC_TASKS: usize = 4;
        const BLOCKING_THREADS: usize = 4;
        const OPS: usize = 100;
        const PERMITS: usize = 2;

        let sem = Arc::new(Semaphore::new(PERMITS));

        let async_handles: Vec<_> = (0..ASYNC_TASKS)
            .map(|_| {
                let s = Arc::clone(&sem);
                tokio::spawn(async move {
                    for _ in 0..OPS {
                        let _g = s.acquire_async().await;
                    }
                })
            })
            .collect();

        let blocking_handles: Vec<_> = (0..BLOCKING_THREADS)
            .map(|_| {
                let s = Arc::clone(&sem);
                std::thread::spawn(move || {
                    for _ in 0..OPS {
                        let _g = s.acquire();
                    }
                })
            })
            .collect();

        for h in async_handles {
            h.await.unwrap();
        }
        for h in blocking_handles {
            h.join().unwrap();
        }

        assert_eq!(sem.available_permits(), PERMITS);
    }

    #[test]
    fn boxed_storage_is_pointer_sized_and_smaller_than_inline() {
        // The bare alias resolves to the boxed (default) representation; boxed keeps the semaphore
        // small, inline trades size for allocation-free async.
        assert_eq!(size_of::<Semaphore32>(), size_of::<Semaphore32Boxed>());
        assert!(size_of::<Semaphore32Boxed>() < size_of::<Semaphore32Inline>());
    }

    #[tokio::test]
    async fn inline_storage_async_and_blocking_work() {
        let sem = Arc::new(Semaphore32Inline::new(1));

        let held = sem.acquire();
        let sem2 = Arc::clone(&sem);
        let waiter = tokio::spawn(async move {
            let _g = sem2.acquire_async().await;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(held);

        waiter.await.unwrap();
        assert_eq!(sem.available_permits(), 1);
    }
}
