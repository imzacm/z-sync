//! An N-party rendezvous [`Barrier`], usable from blocking and async code.
//!
//! A barrier blocks each party at [`wait`](Barrier::wait) / [`wait_async`](Barrier::wait_async)
//! until `n` parties have arrived, then releases them all at once. It is reusable: after a release
//! the barrier resets for the next round. Exactly one party per round is designated the *leader*
//! (via [`BarrierWaitResult::is_leader`]), which callers can use to elect a single thread to run
//! post-rendezvous work.

use core::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};

use num_traits::{ConstOne, ConstZero, NumCast, ToPrimitive};

use crate::notify::{ASYNC_CAPACITY, Notify, NotifyStateU64};
use crate::park_strategy::{DefaultParkStrategy, ParkStrategy};
use crate::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};

// Async waker-storage variants. The bare `Barrier{16,32,64}` names alias the default representation
// (boxed: small, allocates its waker queue lazily); the `Boxed`/`Inline` names select the
// representation explicitly. See [`WakerStorage`].

/// [`Barrier16`] with boxed waker storage (the default): small, allocates its waker queue lazily
/// (and never at all for blocking-only usage).
pub type Barrier16Boxed<P = DefaultParkStrategy> =
    Barrier<BarrierStateU16, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Barrier32`]. See [`Barrier16Boxed`].
pub type Barrier32Boxed<P = DefaultParkStrategy> =
    Barrier<BarrierStateU32, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Barrier64`]. See [`Barrier16Boxed`].
pub type Barrier64Boxed<P = DefaultParkStrategy> =
    Barrier<BarrierStateU64, P, BoxedWakers<ASYNC_CAPACITY>>;

/// [`Barrier16`] with inline waker storage: allocation-free on the async path (usable without a
/// global allocator), at the cost of a larger barrier.
pub type Barrier16Inline<P = DefaultParkStrategy> =
    Barrier<BarrierStateU16, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Barrier32`]. See [`Barrier16Inline`].
pub type Barrier32Inline<P = DefaultParkStrategy> =
    Barrier<BarrierStateU32, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Barrier64`]. See [`Barrier16Inline`].
pub type Barrier64Inline<P = DefaultParkStrategy> =
    Barrier<BarrierStateU64, P, InlineWakers<ASYNC_CAPACITY>>;

pub type Barrier16<P = DefaultParkStrategy> = Barrier16Boxed<P>;
pub type Barrier32<P = DefaultParkStrategy> = Barrier32Boxed<P>;
pub type Barrier64<P = DefaultParkStrategy> = Barrier64Boxed<P>;

/// The packed `(generation, count)` state of a [`Barrier`].
///
/// `count` is the number of parties that have arrived in the current round; `generation` is a
/// wrapping round counter that a waiting party compares against to learn when its round has been
/// released. The `16`/`32`/`64` suffix on the concrete types selects the width of the packed atomic
/// word, which bounds the maximum party count and how many rounds may elapse before the generation
/// counter wraps.
pub trait BarrierState: Sized + Copy + Clone + PartialEq + Eq {
    type Atomic: core::fmt::Debug;

    type Count: Copy + Eq + Ord + NumCast + ConstZero + ConstOne;
    type Generation: Copy + Eq;

    // --- State Getters ---
    fn count(self) -> Self::Count;
    fn generation(self) -> Self::Generation;

    // --- State Mutations (Pure) ---
    /// Adds one to the arrival count, leaving the generation untouched.
    fn with_incremented_count(self) -> Self;
    /// Resets the arrival count to zero and advances the generation by one (wrapping) — the round
    /// is complete.
    fn released(self) -> Self;

    // --- Atomic Operations ---
    fn atomic_initial() -> Self::Atomic;
    fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_compare_exchange_weak(
        atomic: &Self::Atomic,
        current: Self,
        new: Self,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self, Self>;
}

#[macro_export]
macro_rules! atomic_barrier_state {
    (
        $vis:vis struct $struct_name:ident(
            $atomic_ty:ident($prim_ty:ty) {
                count: $c_ty:ty = $c_bits:expr,
                generation: $g_ty:ty = $g_bits:expr $(,)?
            }
        )
    ) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        $vis struct $struct_name(pub $prim_ty);

        impl $struct_name {
            pub const COUNT_SHIFT: $prim_ty = 0;
            pub const GENERATION_SHIFT: $prim_ty = Self::COUNT_SHIFT + $c_bits;

            const _ASSERT_SIZE: () = assert!(
                ($c_bits + $g_bits) <= <$prim_ty>::BITS as $prim_ty,
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

            pub const COUNT_MASK: $prim_ty = Self::mask($c_bits, Self::COUNT_SHIFT);
            pub const GENERATION_MASK: $prim_ty = Self::mask($g_bits, Self::GENERATION_SHIFT);
        }

        impl BarrierState for $struct_name {
            type Atomic = $atomic_ty;
            type Count = $c_ty;
            type Generation = $g_ty;

            #[inline(always)] fn count(self) -> Self::Count { ((self.0 & Self::COUNT_MASK) >> Self::COUNT_SHIFT) as Self::Count }
            #[inline(always)] fn generation(self) -> Self::Generation { ((self.0 & Self::GENERATION_MASK) >> Self::GENERATION_SHIFT) as Self::Generation }

            #[inline(always)] fn with_incremented_count(self) -> Self { Self(self.0 + (1 << Self::COUNT_SHIFT)) }
            #[inline(always)] fn released(self) -> Self { Self((self.0 & Self::GENERATION_MASK).wrapping_add(1 << Self::GENERATION_SHIFT) & Self::GENERATION_MASK) }

            #[inline(always)] fn atomic_initial() -> Self::Atomic { <$atomic_ty>::new(0) }
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
        }
    };
}

atomic_barrier_state!(pub struct BarrierStateU64(
    AtomicU64(u64) {
        count: u32 = 32,
        generation: u32 = 32,
    }
));

atomic_barrier_state!(pub struct BarrierStateU32(
    AtomicU32(u32) {
        count: u16 = 16,
        generation: u16 = 16,
    }
));

atomic_barrier_state!(pub struct BarrierStateU16(
    AtomicU16(u16) {
        count: u8 = 8,
        generation: u8 = 8,
    }
));

/// The outcome of a single arrival, telling this party whether it was elected leader.
enum Arrival<G> {
    /// This party was the last to arrive and released the round.
    Leader,
    /// This party arrived early and must wait for the round (identified by `G`) to be released.
    Waiter(G),
}

/// A rendezvous point for a fixed number of parties, supporting blocking and async waiting.
///
/// Construct with [`Barrier::new`] for `n` parties. Each call to [`wait`](Barrier::wait) or
/// [`wait_async`](Barrier::wait_async) blocks until `n` parties have arrived, then all are released
/// together; exactly one receives a [`BarrierWaitResult`] with
/// [`is_leader`](BarrierWaitResult::is_leader) set. Blocking and async parties share the same
/// round, so a mix of the two rendezvous correctly. The barrier is reusable — after releasing it
/// resets for the next round.
///
/// The `16`/`32`/`64` suffix selects the width of the packed atomic state word, bounding the party
/// count and the number of rounds before the generation counter wraps (see [`BarrierState`]).
///
/// The `W` parameter selects how the async waker queue is stored — [`BoxedWakers`] (the default,
/// keeping the barrier small and allocating lazily) or [`InlineWakers`] (queue stored inline:
/// larger, but allocation-free on the async path). It has no effect on the blocking path. See
/// [`WakerStorage`] and the [`Barrier16Boxed`] / [`Barrier16Inline`] aliases.
///
/// Note: like `std::sync::Barrier`, a party is committed once it has arrived. Dropping a
/// [`wait_async`](Barrier::wait_async) future after it has been polled (e.g. cancelling it) leaves
/// the barrier one party short for that round.
pub struct Barrier<
    S: BarrierState = BarrierStateU32,
    P = DefaultParkStrategy,
    W = BoxedWakers<ASYNC_CAPACITY>,
> {
    parties: usize,
    /// Bit layout (`BarrierStateU64`):
    /// - 0..32: parties arrived in the current round (u32)
    /// - 32..64: generation / round counter (u32)
    state: S::Atomic,
    notify: Notify<NotifyStateU64, P, W>,
}

impl<S: BarrierState, W: WakerStorage<ASYNC_CAPACITY>> Barrier<S, DefaultParkStrategy, W> {
    /// Creates a barrier that releases once `n` parties have arrived.
    ///
    /// As with `std::sync::Barrier`, `n` of `0` or `1` never blocks: every party is released
    /// immediately (and is its own leader).
    pub fn new(n: usize) -> Self {
        Self::with_park_strategy(n)
    }
}

impl<S: BarrierState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Barrier<S, P, W> {
    /// Creates a barrier for `n` parties with an explicit [`ParkStrategy`].
    pub fn with_park_strategy(n: usize) -> Self {
        // The arrival count only ever reaches `n - 1` before a reset, so `n` must fit the count
        // field. Since every width uses a count field as wide as its `Count` type, this cast is the
        // capacity check.
        let _: S::Count = NumCast::from(n).expect("party count exceeds barrier capacity");
        Self {
            parties: n,
            state: S::atomic_initial(),
            notify: Notify::with_park_strategy(),
        }
    }

    /// The number of parties required to release the barrier.
    #[inline(always)]
    pub fn num_parties(&self) -> usize {
        self.parties
    }

    #[inline(always)]
    fn generation(&self) -> S::Generation {
        S::atomic_load(&self.state, Ordering::Acquire).generation()
    }

    /// Registers this party's arrival, releasing the round (and returning [`Arrival::Leader`]) if
    /// it completes the count, otherwise returning the round to wait on.
    fn arrive(&self) -> Arrival<S::Generation> {
        let mut state = S::atomic_load(&self.state, Ordering::Relaxed);
        loop {
            // `count` never exceeds `parties`, so this fits `usize` on every supported width.
            let arrived = state.count().to_usize().unwrap_or(usize::MAX) + 1;

            if arrived >= self.parties {
                match S::atomic_compare_exchange_weak(
                    &self.state,
                    state,
                    state.released(),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        self.notify.notify(usize::MAX);
                        return Arrival::Leader;
                    }
                    Err(v) => state = v,
                }
            } else {
                match S::atomic_compare_exchange_weak(
                    &self.state,
                    state,
                    state.with_incremented_count(),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Arrival::Waiter(state.generation()),
                    Err(v) => state = v,
                }
            }
        }
    }

    /// Blocks the current thread until all parties have arrived.
    pub fn wait(&self) -> BarrierWaitResult {
        let my_generation = match self.arrive() {
            Arrival::Leader => return BarrierWaitResult { is_leader: true },
            Arrival::Waiter(generation) => generation,
        };

        // Check → listen → check: create the listener before re-reading the generation so a release
        // that lands in the gap still advances the notify epoch our listener snapshotted.
        loop {
            let listener = self.notify.listener();
            if self.generation() != my_generation {
                return BarrierWaitResult { is_leader: false };
            }
            listener.wait();
        }
    }

    /// Resolves once all parties have arrived.
    pub async fn wait_async(&self) -> BarrierWaitResult {
        let my_generation = match self.arrive() {
            Arrival::Leader => return BarrierWaitResult { is_leader: true },
            Arrival::Waiter(generation) => generation,
        };

        loop {
            let listener = self.notify.listener();
            if self.generation() != my_generation {
                return BarrierWaitResult { is_leader: false };
            }
            listener.await;
        }
    }
}

// SAFETY: the packed `state` is an atomic and `notify` is itself `Send`/`Sync`; all sharing of
// interior state goes through those, exactly as for the crate's other primitives.
unsafe impl<S: BarrierState, P, W> Send for Barrier<S, P, W> {}
unsafe impl<S: BarrierState, P, W> Sync for Barrier<S, P, W> {}

impl<S: BarrierState, P, W> core::fmt::Debug for Barrier<S, P, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Barrier")
            .field("parties", &self.parties)
            .finish_non_exhaustive()
    }
}

/// Returned by [`Barrier::wait`] / [`Barrier::wait_async`]; identifies the single leader of a
/// round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarrierWaitResult {
    is_leader: bool,
}

impl BarrierWaitResult {
    /// Returns `true` for exactly one party per round — the one that completed the rendezvous.
    #[inline(always)]
    pub fn is_leader(&self) -> bool {
        self.is_leader
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use std::vec::Vec;

    // Exercise the default width under the alias used by the public API.
    type Barrier = super::Barrier;

    use super::*;

    #[test]
    fn single_party_never_blocks_and_leads() {
        let barrier = Barrier::new(1);
        assert!(barrier.wait().is_leader());
        // Reusable: still releases immediately on the next round.
        assert!(barrier.wait().is_leader());
    }

    #[test]
    fn zero_parties_never_blocks() {
        let barrier = Barrier::new(0);
        assert!(barrier.wait().is_leader());
    }

    #[test]
    fn releases_when_all_arrive_with_one_leader() {
        const THREADS: usize = 8;
        let barrier = Arc::new(Barrier::new(THREADS));
        let arrived = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let arrived = Arc::clone(&arrived);
                std::thread::spawn(move || {
                    arrived.fetch_add(1, Ordering::Relaxed);
                    let result = barrier.wait();
                    // Every party must have arrived before any is released.
                    assert_eq!(arrived.load(Ordering::Relaxed), THREADS);
                    result.is_leader()
                })
            })
            .collect();

        let leaders = handles.into_iter().map(|h| h.join().unwrap()).filter(|&l| l).count();
        assert_eq!(leaders, 1);
    }

    #[test]
    fn many_more_threads_than_parties_cycle_cleanly() {
        // With more threads than parties, arrivals roll into successive rounds; every arrival is
        // released and the number of leaders equals the number of completed rounds.
        const THREADS: usize = 12;
        const PARTIES: usize = 3;
        let barrier = Arc::new(Barrier::new(PARTIES));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || barrier.wait().is_leader())
            })
            .collect();

        let leaders = handles.into_iter().map(|h| h.join().unwrap()).filter(|&l| l).count();
        assert_eq!(leaders, THREADS / PARTIES);
    }

    #[test]
    fn reusable_across_many_rounds() {
        const THREADS: usize = 4;
        const ROUNDS: usize = 50;
        let barrier = Arc::new(Barrier::new(THREADS));
        let leaders = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let leaders = Arc::clone(&leaders);
                std::thread::spawn(move || {
                    for _ in 0..ROUNDS {
                        if barrier.wait().is_leader() {
                            leaders.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        // Exactly one leader per round.
        assert_eq!(leaders.load(Ordering::Relaxed), ROUNDS);
    }

    #[test]
    fn wait_blocks_until_last_party() {
        let barrier = Arc::new(Barrier::new(2));
        let done = Arc::new(AtomicUsize::new(0));

        let b2 = Arc::clone(&barrier);
        let d2 = Arc::clone(&done);
        let waiter = std::thread::spawn(move || {
            b2.wait();
            d2.store(1, Ordering::Release);
        });

        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(done.load(Ordering::Acquire), 0, "waiter released before the second party");

        barrier.wait();
        waiter.join().unwrap();
        assert_eq!(done.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn async_releases_when_all_arrive_with_one_leader() {
        const TASKS: usize = 8;
        let barrier = Arc::new(Barrier::new(TASKS));
        let arrived = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..TASKS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let arrived = Arc::clone(&arrived);
                tokio::spawn(async move {
                    arrived.fetch_add(1, Ordering::Relaxed);
                    let result = barrier.wait_async().await;
                    assert_eq!(arrived.load(Ordering::Relaxed), TASKS);
                    result.is_leader()
                })
            })
            .collect();

        let mut leaders = 0;
        for h in handles {
            if h.await.unwrap() {
                leaders += 1;
            }
        }
        assert_eq!(leaders, 1);
    }

    #[tokio::test]
    async fn async_wait_waits_for_blocking_party() {
        let barrier = Arc::new(Barrier::new(2));

        let b2 = Arc::clone(&barrier);
        let waiter = tokio::spawn(async move { b2.wait_async().await.is_leader() });

        tokio::time::sleep(Duration::from_millis(20)).await;
        // A blocking party completes the round the async task is waiting on.
        let blocking_leader = barrier.wait().is_leader();
        let async_leader = waiter.await.unwrap();

        // Exactly one of the two is the leader.
        assert!(blocking_leader ^ async_leader);
    }

    #[test]
    fn boxed_storage_is_smaller_than_inline() {
        // The bare alias resolves to the boxed (default) representation.
        assert_eq!(size_of::<Barrier32>(), size_of::<Barrier32Boxed>());
        assert!(size_of::<Barrier32Boxed>() < size_of::<Barrier32Inline>());
    }

    #[tokio::test]
    async fn inline_storage_works() {
        let barrier = Arc::new(Barrier32Inline::new(2));

        let b2 = Arc::clone(&barrier);
        let waiter = tokio::spawn(async move { b2.wait_async().await });

        tokio::time::sleep(Duration::from_millis(20)).await;
        barrier.wait();
        waiter.await.unwrap();
    }
}
