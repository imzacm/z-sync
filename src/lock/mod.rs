mod state;

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use core::task::{Context, Poll, Waker};

use num_traits::{ConstZero, NumCast};

pub use self::state::*;
use crate::NotifyState;
use crate::park_strategy::{DefaultParkStrategy, ParkStrategy};
use crate::waker_queue::{WakerQueueLock, WakerTicket};
use crate::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};

// Async waker-storage variants. The bare `Lock{16,32,64}` names alias the default representation
// (boxed: small struct, allocates lazily); the `Boxed`/`Inline` names select the representation
// explicitly. Unlike `Notify`, inline is *not* recommended for contended locks — it regresses
// contended async by putting both waker queues on the hot `state` cache line. See [`WakerStorage`].

/// [`Lock16`] with boxed waker storage (the default): the lock stays small and allocates its two
/// waker queues lazily (and never at all for blocking-only usage).
pub type Lock16Boxed<T, P = DefaultParkStrategy> =
    Lock<T, LockStateU16, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Lock32`]. See [`Lock16Boxed`].
pub type Lock32Boxed<T, P = DefaultParkStrategy> =
    Lock<T, LockStateU32, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Lock64`]. See [`Lock16Boxed`].
pub type Lock64Boxed<T, P = DefaultParkStrategy> =
    Lock<T, LockStateU64, P, BoxedWakers<ASYNC_CAPACITY>>;

/// [`Lock16`] with inline waker storage: allocation-free on the async path (usable without a global
/// allocator), at the cost of a much larger lock. Prefer the default boxed form under contention.
pub type Lock16Inline<T, P = DefaultParkStrategy> =
    Lock<T, LockStateU16, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Lock32`]. See [`Lock16Inline`].
pub type Lock32Inline<T, P = DefaultParkStrategy> =
    Lock<T, LockStateU32, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Lock64`]. See [`Lock16Inline`].
pub type Lock64Inline<T, P = DefaultParkStrategy> =
    Lock<T, LockStateU64, P, InlineWakers<ASYNC_CAPACITY>>;

pub type Lock16<T, P = DefaultParkStrategy> = Lock16Boxed<T, P>;
pub type Lock32<T, P = DefaultParkStrategy> = Lock32Boxed<T, P>;
pub type Lock64<T, P = DefaultParkStrategy> = Lock64Boxed<T, P>;

pub(crate) const ASYNC_CAPACITY: usize = 4;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const READ_SPIN_MAX: usize = 64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const WRITE_SPIN_MAX: usize = 64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SPIN_CAP: usize = 32;

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const READ_SPIN_MAX: usize = 16;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const WRITE_SPIN_MAX: usize = 32;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const SPIN_CAP: usize = 16;

#[cfg(all(not(any(target_arch = "x86", target_arch = "x86_64")), feature = "std"))]
const SPIN_YIELD_MAX: usize = 8;

/// An efficient multi-purpose blocking and async lock supporting Mutex and RwLock style usage.
///
/// The `W` parameter selects how the async waker queues are stored — [`BoxedWakers`] (the default,
/// keeping the lock small and allocating lazily) or [`InlineWakers`] (queues stored inline: larger
/// lock, but allocation-free and indirection-free on the async path). It has no effect on the
/// blocking path. See [`WakerStorage`] and the [`Lock16Inline`] aliases.
#[derive(Debug)]
pub struct Lock<T, S: LockState, P = DefaultParkStrategy, W = BoxedWakers<ASYNC_CAPACITY>> {
    _marker: core::marker::PhantomData<P>,
    /// Bit layout:
    /// - 0..16:   read async wakers count (u16)
    /// - 16..24:  read parked threads count (u8)
    /// - 24..32:  write async wakers count (u8)
    /// - 32..40:  write parked threads count (u8)
    /// - 40..48:  writer count (u8)
    /// - 48..64:  readers count (u16)
    state: S::Atomic,
    data: UnsafeCell<T>,
    read_wakers: W,
    write_wakers: W,
}

impl<T, S: LockState, W: WakerStorage<ASYNC_CAPACITY>> Lock<T, S, DefaultParkStrategy, W> {
    pub const fn new(data: T) -> Self {
        Self::with_park_strategy(data)
    }
}

impl<T, S: LockState, P, W> Default for Lock<T, S, P, W>
where
    T: Default,
    P: ParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY>,
{
    fn default() -> Self {
        Self::with_park_strategy(T::default())
    }
}

impl<T, S: LockState, P: ParkStrategy> Lock<T, S, P, BoxedWakers<ASYNC_CAPACITY>> {
    #[inline(always)]
    pub fn into_observable<N: NotifyState>(self) -> crate::ObservableLock<T, S, N, P>
    where
        T: Clone,
    {
        crate::ObservableLock::from_lock(self)
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Lock<T, S, P, W> {
    pub const fn with_park_strategy(data: T) -> Self {
        Self {
            _marker: core::marker::PhantomData,
            state: S::INITIAL_ATOMIC,
            data: UnsafeCell::new(data),
            read_wakers: W::INIT,
            write_wakers: W::INIT,
        }
    }

    #[inline(always)]
    fn get_read_wakers(&self) -> &WakerQueueLock<ASYNC_CAPACITY> {
        self.read_wakers.queue()
    }

    #[inline(always)]
    fn get_write_wakers(&self) -> &WakerQueueLock<ASYNC_CAPACITY> {
        self.write_wakers.queue()
    }

    pub fn try_read(&self) -> Option<ReadGuard<'_, T, S, P, W>> {
        // Fast test: Don't dirty the cache line if a writer is waiting/active.
        if cfg!(not(any(target_arch = "x86", target_arch = "x86_64")))
            && self.load_state(Ordering::Relaxed).has_any_write_state()
        {
            return None;
        }

        let old_state = self.add_reader(Ordering::Acquire);
        // This will drop on None, so we don't need to worry about it.
        let guard = ReadGuard { lock: self };

        if old_state.has_any_write_state() {
            return None;
        }

        Some(guard)
    }

    fn spin_try_read(&self) -> Option<ReadGuard<'_, T, S, P, W>> {
        let mut backoff = 1;
        for _ in 0..READ_SPIN_MAX {
            let state = self.load_state(Ordering::Relaxed);

            if !state.has_any_write_state()
                && let Some(guard) = self.try_read()
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

        // x86 seems to perform better without yielding.
        #[cfg(all(not(any(target_arch = "x86", target_arch = "x86_64")), feature = "std"))]
        for _ in 0..SPIN_YIELD_MAX {
            let state = self.load_state(Ordering::Relaxed);
            if !state.has_any_write_state()
                && let Some(guard) = self.try_read()
            {
                return Some(guard);
            }
            std::thread::yield_now();
        }

        None
    }

    pub fn try_write(&self) -> Option<WriteGuard<'_, T, S, P, W>> {
        let mut state = match S::atomic_compare_exchange_weak(
            &self.state,
            S::empty(),
            S::with_writer(),
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Some(WriteGuard { lock: self }),
            Err(v) => v,
        };

        loop {
            // Instantly check if any readers, writers, or an upgradable reader exist
            if state.has_readers_or_writers() || state.has_upgradable() {
                return None;
            }

            let new_state = state.add_writer_state();
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                new_state,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(WriteGuard { lock: self }),
                Err(v) => state = v,
            }
        }
    }

    fn spin_try_write(&self) -> Option<WriteGuard<'_, T, S, P, W>> {
        let mut backoff = 1;
        for _ in 0..WRITE_SPIN_MAX {
            let state = self.load_state(Ordering::Relaxed);

            if !state.has_readers_or_writers()
                && !state.has_upgradable()
                && let Some(guard) = self.try_write()
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

        // x86 seems to perform better without yielding.
        #[cfg(all(not(any(target_arch = "x86", target_arch = "x86_64")), feature = "std"))]
        for _ in 0..SPIN_YIELD_MAX {
            let state = self.load_state(Ordering::Relaxed);
            if !state.has_readers_or_writers()
                && !state.has_upgradable()
                && let Some(guard) = self.try_write()
            {
                return Some(guard);
            }
            std::thread::yield_now();
        }

        None
    }

    #[inline(always)]
    pub fn read(&self) -> ReadGuard<'_, T, S, P, W> {
        if let Some(guard) = self.try_read() {
            return guard;
        }
        self.read_slow()
    }

    #[cold]
    #[inline(never)]
    fn read_slow(&self) -> ReadGuard<'_, T, S, P, W> {
        if let Some(guard) = self.spin_try_read() {
            return guard;
        }

        self.add_read_parker(Ordering::Relaxed);

        loop {
            P::park(self.reader_parking_key(), || {
                let s = self.load_state(Ordering::Relaxed);
                s.writers() > S::Writers::ZERO
                    || s.write_parked() > S::WriteParked::ZERO
                    || s.write_wakers() > S::WriteWakers::ZERO
            });

            if let Some(guard) = self.try_read() {
                self.sub_read_parker(Ordering::Relaxed);
                return guard;
            }
        }
    }

    #[inline(always)]
    pub fn write(&self) -> WriteGuard<'_, T, S, P, W> {
        if let Some(guard) = self.try_write() {
            return guard;
        }
        self.write_slow()
    }

    #[cold]
    #[inline(never)]
    fn write_slow(&self) -> WriteGuard<'_, T, S, P, W> {
        if let Some(guard) = self.spin_try_write() {
            return guard;
        }

        self.add_write_parker(Ordering::Relaxed);

        loop {
            P::park(self.writer_parking_key(), || {
                let s = self.load_state(Ordering::Relaxed);
                s.writers() > S::Writers::ZERO
                    || s.readers() > S::Readers::ZERO
                    || s.has_upgradable()
            });

            if let Some(guard) = self.try_write() {
                self.sub_write_parker(Ordering::Relaxed);
                return guard;
            }
        }
    }

    #[inline(always)]
    pub fn read_async(&self) -> ReadFuture<'_, T, S, P, W> {
        ReadFuture { lock: self, waker_node_ticket: None }
    }

    #[inline(always)]
    pub fn write_async(&self) -> WriteFuture<'_, T, S, P, W> {
        WriteFuture { lock: self, waker_node_ticket: None }
    }

    #[inline(always)]
    fn common_dropped<const IS_READER: bool>(&self) {
        let state = if IS_READER {
            self.sub_reader(Ordering::Release).sub_reader_state()
        } else {
            self.sub_writer(Ordering::Release).sub_writer_state()
        };

        if state.has_readers_or_writers() || !state.has_any_waiters() {
            return;
        }

        self.common_dropped_slow(state);
    }

    #[cold]
    #[inline(never)]
    fn common_dropped_slow(&self, state: S) {
        if state.has_write_waiters() {
            if state.has_upgradable() {
                // A reader left while an upgrade is pending. The upgrading thread waits as a
                // write-side waiter but may sit behind other write-waiters, so wake them all to
                // guarantee it re-checks whether the readers have drained.
                self.wake_all_write_waiters();
            } else if state.write_wakers() > S::WriteWakers::ZERO {
                self.wake_one_in_queue::<false>();
            } else if state.write_parked() > S::WriteParked::ZERO {
                P::unpark_one(self.writer_parking_key());
            }
        } else if state.has_read_waiters() {
            if state.read_wakers() > S::ReadWakers::ZERO {
                self.wake_all_in_queue::<true>();
            }
            if state.read_parked() > S::ReadParked::ZERO {
                P::unpark_all(self.reader_parking_key());
            }
        }
    }

    fn wake_one_in_queue<const IS_READER: bool>(&self) {
        let queue = if IS_READER { self.get_read_wakers() } else { self.get_write_wakers() };

        let waker = queue.lock().pop_and_take();
        if let Some(waker) = waker {
            if IS_READER {
                self.sub_read_waker(Ordering::Release);
            } else {
                self.sub_write_waker(Ordering::Release);
            }
            waker.wake();
        }
    }

    fn wake_all_in_queue<const IS_READER: bool>(&self) {
        const BATCH_SIZE: usize = 32;

        let queue = if IS_READER { self.get_read_wakers() } else { self.get_write_wakers() };

        let mut batch_sub = S::batch_sub_new();

        loop {
            let mut popped = 0;

            let mut wakers: [MaybeUninit<Waker>; BATCH_SIZE] =
                [const { MaybeUninit::uninit() }; BATCH_SIZE];

            {
                let mut queue = queue.lock();
                while popped < BATCH_SIZE {
                    let Some(waker) = queue.pop_and_take() else { break };
                    wakers[popped].write(waker);
                    popped += 1;
                }
            }

            if popped == 0 {
                break;
            }

            if IS_READER {
                batch_sub = S::batch_sub_read_waker(batch_sub, NumCast::from(popped).unwrap());
            } else {
                batch_sub = S::batch_sub_write_waker(batch_sub, NumCast::from(popped).unwrap());
            }

            for waker in &mut wakers[..popped] {
                // SAFETY: We explicitly initialized exactly `popped` elements
                // inside the mutex lock above.
                unsafe {
                    waker.assume_init_read().wake();
                }
            }
        }

        S::atomic_fetch_sub_batch(&self.state, batch_sub, Ordering::Release);
    }

    /// Wakes every parked/async reader (shared reads are now permitted).
    fn wake_all_read_waiters(&self) {
        let state = self.load_state(Ordering::Acquire);
        if state.read_wakers() > S::ReadWakers::ZERO {
            self.wake_all_in_queue::<true>();
        }
        if state.read_parked() > S::ReadParked::ZERO {
            P::unpark_all(self.reader_parking_key());
        }
    }

    /// Wakes every parked/async write-side waiter (blocked writers and upgraders).
    fn wake_all_write_waiters(&self) {
        let state = self.load_state(Ordering::Acquire);
        if state.write_wakers() > S::WriteWakers::ZERO {
            self.wake_all_in_queue::<false>();
        }
        if state.write_parked() > S::WriteParked::ZERO {
            P::unpark_all(self.writer_parking_key());
        }
    }

    /// Atomically converts a held write lock into a read lock (`downgrade`). The caller must own
    /// the sole writer; on return a single reader is held. Parked/async readers are woken since
    /// shared reads are now permitted, while writers stay blocked (we still hold a read lock).
    fn write_to_read(&self) {
        let mut state = self.load_state(Ordering::Relaxed);
        loop {
            let new = state.sub_writer_state().add_reader_state();
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                new,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => state = v,
            }
        }
        self.wake_all_read_waiters();
    }

    /// Atomically converts a held write lock into an upgradable read lock. Shared readers become
    /// compatible again (woken), while writers and other upgraders stay excluded by the flag.
    fn write_to_upgradable(&self) {
        let mut state = self.load_state(Ordering::Relaxed);
        loop {
            let new = state.sub_writer_state().add_upgradable_state();
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                new,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => state = v,
            }
        }
        self.wake_all_read_waiters();
    }

    /// Attempts to acquire an upgradable read lock without blocking: succeeds only when no writer,
    /// write-waiter, or other upgrader is present (shared readers are fine). Yields to waiting
    /// writers for fairness.
    pub fn try_upgradable_read(&self) -> Option<UpgradableReadGuard<'_, T, S, P, W>> {
        let mut state = self.load_state(Ordering::Relaxed);
        loop {
            if state.has_any_write_state() || state.has_upgradable() {
                return None;
            }
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                state.add_upgradable_state(),
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(UpgradableReadGuard { lock: self }),
                Err(v) => state = v,
            }
        }
    }

    /// Post-park upgradable acquire: ignores write-waiters (which may include this thread) and only
    /// yields to an actual writer or another live upgrader.
    fn try_acquire_upgradable_ignoring_waiters(
        &self,
    ) -> Option<UpgradableReadGuard<'_, T, S, P, W>> {
        let mut state = self.load_state(Ordering::Relaxed);
        loop {
            if state.writers() > S::Writers::ZERO || state.has_upgradable() {
                return None;
            }
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                state.add_upgradable_state(),
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(UpgradableReadGuard { lock: self }),
                Err(v) => state = v,
            }
        }
    }

    fn spin_try_upgradable_read(&self) -> Option<UpgradableReadGuard<'_, T, S, P, W>> {
        let mut backoff = 1;
        for _ in 0..WRITE_SPIN_MAX {
            let state = self.load_state(Ordering::Relaxed);
            if !state.has_any_write_state()
                && !state.has_upgradable()
                && let Some(guard) = self.try_upgradable_read()
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

        #[cfg(all(not(any(target_arch = "x86", target_arch = "x86_64")), feature = "std"))]
        for _ in 0..SPIN_YIELD_MAX {
            let state = self.load_state(Ordering::Relaxed);
            if !state.has_any_write_state()
                && !state.has_upgradable()
                && let Some(guard) = self.try_upgradable_read()
            {
                return Some(guard);
            }
            std::thread::yield_now();
        }

        None
    }

    /// Acquires an upgradable read lock, blocking until no writer or other upgrader holds it.
    #[inline(always)]
    pub fn upgradable_read(&self) -> UpgradableReadGuard<'_, T, S, P, W> {
        if let Some(guard) = self.try_upgradable_read() {
            return guard;
        }
        self.upgradable_read_slow()
    }

    #[cold]
    #[inline(never)]
    fn upgradable_read_slow(&self) -> UpgradableReadGuard<'_, T, S, P, W> {
        if let Some(guard) = self.spin_try_upgradable_read() {
            return guard;
        }

        // Wait as a write-side waiter. This blocks new shared readers while we wait (they yield to
        // us as they would to a pending writer) — the one behaviour that differs from parking_lot.
        self.add_write_parker(Ordering::Relaxed);

        loop {
            P::park(self.writer_parking_key(), || {
                let s = self.load_state(Ordering::Relaxed);
                s.writers() > S::Writers::ZERO || s.has_upgradable()
            });

            if let Some(guard) = self.try_acquire_upgradable_ignoring_waiters() {
                self.sub_write_parker(Ordering::Relaxed);
                return guard;
            }
        }
    }

    /// Resolves to an upgradable read lock once no writer or other upgrader holds it.
    #[inline(always)]
    pub fn upgradable_read_async(&self) -> UpgradableReadFuture<'_, T, S, P, W> {
        UpgradableReadFuture { lock: self, waker_node_ticket: None }
    }

    /// Attempts the upgradable → write transition: succeeds only if this upgrader is the sole
    /// remaining reader (no shared readers) and no writer is present.
    fn try_upgrade_inner(&self) -> bool {
        let mut state = self.load_state(Ordering::Relaxed);
        loop {
            if state.readers() > S::Readers::ZERO || state.writers() > S::Writers::ZERO {
                return false;
            }
            let new = state.sub_upgradable_state().add_writer_state();
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                new,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(v) => state = v,
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn upgrade_slow(&self) -> WriteGuard<'_, T, S, P, W> {
        // Register as a write-waiter first: this blocks *new* readers (via `has_any_write_state`)
        // so that the existing readers can drain to zero.
        self.add_write_parker(Ordering::Relaxed);

        let mut backoff = 1;
        for _ in 0..WRITE_SPIN_MAX {
            if self.try_upgrade_inner() {
                self.sub_write_parker(Ordering::Relaxed);
                return WriteGuard { lock: self };
            }
            for _ in 0..backoff {
                core::hint::spin_loop();
            }
            if backoff < SPIN_CAP {
                backoff <<= 1;
            }
        }

        loop {
            P::park(self.writer_parking_key(), || {
                self.load_state(Ordering::Relaxed).readers() > S::Readers::ZERO
            });
            if self.try_upgrade_inner() {
                self.sub_write_parker(Ordering::Relaxed);
                return WriteGuard { lock: self };
            }
        }
    }

    /// Converts a held upgradable read lock into a plain read lock (clears the flag, keeps a reader
    /// slot). A parked upgrader — which can now proceed — is woken.
    fn upgradable_to_read(&self) {
        let mut state = self.load_state(Ordering::Relaxed);
        loop {
            let new = state.sub_upgradable_state().add_reader_state();
            match S::atomic_compare_exchange_weak(
                &self.state,
                state,
                new,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => state = v,
            }
        }
        let state = self.load_state(Ordering::Acquire);
        if state.has_write_waiters() {
            self.wake_all_write_waiters();
        }
    }

    /// Releases a held upgradable read lock (the flag), waking any write-side waiter it excluded.
    fn upgrader_dropped(&self) {
        let old = S::atomic_sub_upgrader(&self.state, Ordering::Release);
        let state = old.sub_upgradable_state();
        if state.has_write_waiters() {
            self.wake_all_write_waiters();
        }
    }
}

impl<T, S: LockState, P, W> Lock<T, S, P, W> {
    #[inline(always)]
    fn add_read_waker(&self, ordering: Ordering) -> S {
        S::atomic_add_read_waker(&self.state, ordering)
    }

    #[inline(always)]
    fn add_read_parker(&self, ordering: Ordering) -> S {
        S::atomic_add_read_parker(&self.state, ordering)
    }

    #[inline(always)]
    fn add_write_waker(&self, ordering: Ordering) -> S {
        S::atomic_add_write_waker(&self.state, ordering)
    }

    #[inline(always)]
    fn add_write_parker(&self, ordering: Ordering) -> S {
        S::atomic_add_write_parker(&self.state, ordering)
    }

    #[inline(always)]
    fn add_reader(&self, ordering: Ordering) -> S {
        S::atomic_add_reader(&self.state, ordering)
    }

    #[inline(always)]
    fn sub_read_waker(&self, ordering: Ordering) -> S {
        S::atomic_sub_read_waker(&self.state, ordering)
    }

    #[inline(always)]
    fn sub_read_parker(&self, ordering: Ordering) -> S {
        S::atomic_sub_read_parker(&self.state, ordering)
    }

    #[inline(always)]
    fn sub_write_waker(&self, ordering: Ordering) -> S {
        S::atomic_sub_write_waker(&self.state, ordering)
    }

    #[inline(always)]
    fn sub_write_parker(&self, ordering: Ordering) -> S {
        S::atomic_sub_write_parker(&self.state, ordering)
    }

    #[inline(always)]
    fn sub_writer(&self, ordering: Ordering) -> S {
        S::atomic_sub_writer(&self.state, ordering)
    }

    #[inline(always)]
    fn sub_reader(&self, ordering: Ordering) -> S {
        S::atomic_sub_reader(&self.state, ordering)
    }

    #[inline(always)]
    fn load_state(&self, ordering: Ordering) -> S {
        S::atomic_load(&self.state, ordering)
    }

    #[inline(always)]
    fn reader_parking_key(&self) -> usize {
        core::ptr::from_ref(self) as usize
    }

    #[inline(always)]
    fn writer_parking_key(&self) -> usize {
        self.reader_parking_key() | 1
    }
}

unsafe impl<T: Send, S: LockState, P, W> Send for Lock<T, S, P, W> {}
unsafe impl<T: Send, S: LockState, P, W> Sync for Lock<T, S, P, W> {}

#[derive(Debug)]
pub struct ReadGuard<
    'a,
    T,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    lock: &'a Lock<T, S, P, W>,
}

impl<'a, T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>>
    ReadGuard<'a, T, S, P, W>
{
    #[inline(always)]
    pub fn map<U, F>(guard: ReadGuard<'a, T, S, P, W>, f: F) -> MappedReadGuard<'a, T, U, S, P, W>
    where
        F: FnOnce(&T) -> &U,
        U: ?Sized,
    {
        let value = unsafe { &*guard.lock.data.get() };
        MappedReadGuard { _guard: guard, value: f(value) }
    }

    /// Like [`map`](ReadGuard::map), but `f` may decline the projection by returning `None`, in
    /// which case the original guard is handed back unchanged.
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    pub fn try_map<U, F>(
        guard: ReadGuard<'a, T, S, P, W>,
        f: F,
    ) -> Result<MappedReadGuard<'a, T, U, S, P, W>, ReadGuard<'a, T, S, P, W>>
    where
        F: FnOnce(&T) -> Option<&U>,
        U: ?Sized,
    {
        let value = unsafe { &*guard.lock.data.get() };
        match f(value) {
            Some(value) => Ok(MappedReadGuard { _guard: guard, value }),
            None => Err(guard),
        }
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for ReadGuard<'_, T, S, P, W>
{
    fn drop(&mut self) {
        self.lock.common_dropped::<true>();
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Deref
    for ReadGuard<'_, T, S, P, W>
{
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

#[derive(Debug)]
pub struct MappedReadGuard<
    'a,
    T,
    U: ?Sized,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    _guard: ReadGuard<'a, T, S, P, W>,
    value: &'a U,
}

impl<'a, T, U: ?Sized, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>>
    MappedReadGuard<'a, T, U, S, P, W>
{
    /// Projects the already-mapped reference further to one of its parts.
    #[inline(always)]
    pub fn map<V, F>(guard: Self, f: F) -> MappedReadGuard<'a, T, V, S, P, W>
    where
        F: FnOnce(&U) -> &V,
        V: ?Sized,
    {
        let value = f(guard.value);
        MappedReadGuard { _guard: guard._guard, value }
    }

    /// Like [`map`](MappedReadGuard::map), but `f` may decline by returning `None`, handing the
    /// original mapped guard back unchanged.
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    pub fn try_map<V, F>(guard: Self, f: F) -> Result<MappedReadGuard<'a, T, V, S, P, W>, Self>
    where
        F: FnOnce(&U) -> Option<&V>,
        V: ?Sized,
    {
        match f(guard.value) {
            Some(value) => Ok(MappedReadGuard { _guard: guard._guard, value }),
            None => Err(guard),
        }
    }
}

impl<'a, T, U: ?Sized, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Deref
    for MappedReadGuard<'a, T, U, S, P, W>
{
    type Target = U;

    fn deref(&self) -> &U {
        self.value
    }
}

/// A [`WriteGuard`] projected to part of the guarded value (see [`WriteGuard::map`]). Derefs
/// mutably to that part and holds the write lock until dropped.
#[derive(Debug)]
pub struct MappedWriteGuard<
    'a,
    T,
    U: ?Sized,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    _guard: WriteGuard<'a, T, S, P, W>,
    value: NonNull<U>,
}

// SAFETY: a `MappedWriteGuard` grants unique access to a `U` living inside the lock. Sending it
// transfers that unique access (sound when `U: Send` and the write guard itself is `Send`, i.e.
// `T: Send`); sharing `&MappedWriteGuard` only ever exposes `&U` (sound when `U: Sync`).
unsafe impl<
    'a,
    T: Send,
    U: ?Sized + Send,
    S: LockState,
    P: ParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY>,
> Send for MappedWriteGuard<'a, T, U, S, P, W>
{
}
unsafe impl<
    'a,
    T: Send,
    U: ?Sized + Sync,
    S: LockState,
    P: ParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY>,
> Sync for MappedWriteGuard<'a, T, U, S, P, W>
{
}

impl<'a, T, U: ?Sized, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>>
    MappedWriteGuard<'a, T, U, S, P, W>
{
    /// Projects the already-mapped reference further to one of its parts.
    #[inline(always)]
    pub fn map<V, F>(mut guard: Self, f: F) -> MappedWriteGuard<'a, T, V, S, P, W>
    where
        F: FnOnce(&mut U) -> &mut V,
        V: ?Sized,
    {
        let value = NonNull::from(f(&mut *guard));
        MappedWriteGuard { _guard: guard._guard, value }
    }

    /// Like [`map`](MappedWriteGuard::map), but `f` may decline by returning `None`, handing the
    /// original mapped guard back unchanged.
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    pub fn try_map<V, F>(mut guard: Self, f: F) -> Result<MappedWriteGuard<'a, T, V, S, P, W>, Self>
    where
        F: FnOnce(&mut U) -> Option<&mut V>,
        V: ?Sized,
    {
        // Compute the projection through a temporary borrow; only commit the move on success.
        match f(unsafe { guard.value.as_mut() }) {
            Some(value) => {
                let value = NonNull::from(value);
                Ok(MappedWriteGuard { _guard: guard._guard, value })
            }
            None => Err(guard),
        }
    }
}

impl<'a, T, U: ?Sized, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Deref
    for MappedWriteGuard<'a, T, U, S, P, W>
{
    type Target = U;

    fn deref(&self) -> &U {
        unsafe { self.value.as_ref() }
    }
}

impl<'a, T, U: ?Sized, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> DerefMut
    for MappedWriteGuard<'a, T, U, S, P, W>
{
    fn deref_mut(&mut self) -> &mut U {
        unsafe { self.value.as_mut() }
    }
}

#[derive(Debug)]
pub struct WriteGuard<
    'a,
    T,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    lock: &'a Lock<T, S, P, W>,
}

impl<'a, T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>>
    WriteGuard<'a, T, S, P, W>
{
    pub(crate) unsafe fn get_lock(guard: &Self) -> &'a Lock<T, S, P, W> {
        guard.lock
    }

    /// Projects the guarded value to one of its parts, yielding a guard that derefs (mutably) to
    /// that part while continuing to hold the write lock.
    #[inline(always)]
    pub fn map<U, F>(mut guard: Self, f: F) -> MappedWriteGuard<'a, T, U, S, P, W>
    where
        F: FnOnce(&mut T) -> &mut U,
        U: ?Sized,
    {
        let value = NonNull::from(f(&mut *guard));
        MappedWriteGuard { _guard: guard, value }
    }

    /// Like [`map`](WriteGuard::map), but `f` may decline the projection by returning `None`, in
    /// which case the original guard is handed back unchanged.
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    pub fn try_map<U, F>(mut guard: Self, f: F) -> Result<MappedWriteGuard<'a, T, U, S, P, W>, Self>
    where
        F: FnOnce(&mut T) -> Option<&mut U>,
        U: ?Sized,
    {
        match f(&mut *guard) {
            Some(value) => {
                let value = NonNull::from(value);
                Ok(MappedWriteGuard { _guard: guard, value })
            }
            None => Err(guard),
        }
    }

    /// Atomically converts this write guard into a read guard without releasing the lock in
    /// between, so no other writer can interpose. Parked readers are woken.
    #[inline]
    pub fn downgrade(self) -> ReadGuard<'a, T, S, P, W> {
        let lock = self.lock;
        // We transition the state ourselves; skip the guard's writer-release Drop.
        core::mem::forget(self);
        lock.write_to_read();
        ReadGuard { lock }
    }

    /// Atomically converts this write guard into an upgradable read guard without releasing the
    /// lock in between. Shared readers are admitted again (and woken); the upgradable lock can
    /// later be re-upgraded.
    #[inline]
    pub fn downgrade_to_upgradable(self) -> UpgradableReadGuard<'a, T, S, P, W> {
        let lock = self.lock;
        core::mem::forget(self);
        lock.write_to_upgradable();
        UpgradableReadGuard { lock }
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for WriteGuard<'_, T, S, P, W>
{
    fn drop(&mut self) {
        self.lock.common_dropped::<false>();
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Deref
    for WriteGuard<'_, T, S, P, W>
{
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> DerefMut
    for WriteGuard<'_, T, S, P, W>
{
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

#[derive(Debug)]
pub struct ReadFuture<
    'a,
    T,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    lock: &'a Lock<T, S, P, W>,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Future
    for ReadFuture<'a, T, S, P, W>
{
    type Output = ReadGuard<'a, T, S, P, W>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();

        if let Some(guard) = this.lock.try_read() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_read_wakers().lock().remove(ticket)
            {
                this.lock.sub_read_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        {
            let mut queue = this.lock.get_read_wakers().lock();

            if let Some(ticket) = this.waker_node_ticket {
                let node = queue.node_mut(ticket.index());

                if node.generation() == ticket.generation() {
                    if node.waker().is_none_or(|w| !w.will_wake(cx.waker())) {
                        *node.waker_mut() = Some(cx.waker().clone());
                    }
                } else {
                    this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                    this.lock.add_read_waker(Ordering::Release);
                }
            } else {
                this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                this.lock.add_read_waker(Ordering::Release);
            }
        }

        if let Some(guard) = this.lock.try_read() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_read_wakers().lock().remove(ticket)
            {
                this.lock.sub_read_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        Poll::Pending
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for ReadFuture<'_, T, S, P, W>
{
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.get_read_wakers().lock().remove(ticket)
        {
            self.lock.sub_read_waker(Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
pub struct WriteFuture<
    'a,
    T,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    lock: &'a Lock<T, S, P, W>,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Future
    for WriteFuture<'a, T, S, P, W>
{
    type Output = WriteGuard<'a, T, S, P, W>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();

        if let Some(guard) = this.lock.try_write() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_write_wakers().lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        {
            let mut queue = this.lock.get_write_wakers().lock();

            if let Some(guard) = this.lock.try_write() {
                if let Some(ticket) = this.waker_node_ticket.take()
                    && queue.remove(ticket)
                {
                    this.lock.sub_write_waker(Ordering::Relaxed);
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
                    this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                    this.lock.add_write_waker(Ordering::Release);
                }
            } else {
                this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                this.lock.add_write_waker(Ordering::Release);
            }
        }

        if let Some(guard) = this.lock.try_write() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_write_wakers().lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        Poll::Pending
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for WriteFuture<'_, T, S, P, W>
{
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.get_write_wakers().lock().remove(ticket)
        {
            self.lock.sub_write_waker(Ordering::Relaxed);
        }
    }
}

/// A guard holding the lock in *upgradable read* mode: it grants shared read access (other readers
/// may hold the lock concurrently) but excludes writers and other upgradable readers, so it can be
/// atomically [`upgrade`](UpgradableReadGuard::upgrade)d to a write lock or
/// [`downgrade`](UpgradableReadGuard::downgrade)d to a plain read lock.
#[derive(Debug)]
pub struct UpgradableReadGuard<
    'a,
    T,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    lock: &'a Lock<T, S, P, W>,
}

impl<'a, T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>>
    UpgradableReadGuard<'a, T, S, P, W>
{
    /// Upgrades to a write lock, blocking until every shared reader has released. No other writer
    /// or upgrader can interpose, since this guard already excludes them.
    #[inline]
    pub fn upgrade(self) -> WriteGuard<'a, T, S, P, W> {
        let lock = self.lock;
        // We drive the upgradable → write transition ourselves; skip the flag-releasing Drop.
        core::mem::forget(self);
        if lock.try_upgrade_inner() {
            return WriteGuard { lock };
        }
        lock.upgrade_slow()
    }

    /// Attempts to upgrade without blocking: succeeds only if there are no other shared readers,
    /// otherwise hands this guard back unchanged.
    #[inline]
    pub fn try_upgrade(self) -> Result<WriteGuard<'a, T, S, P, W>, Self> {
        let lock = self.lock;
        if lock.try_upgrade_inner() {
            core::mem::forget(self);
            Ok(WriteGuard { lock })
        } else {
            Err(self)
        }
    }

    /// Resolves to a write lock once every shared reader has released.
    ///
    /// Cancellation note: if the returned future is dropped before it resolves, the upgradable lock
    /// it held is released (there is no guard to hand back from a dropped future).
    #[inline]
    pub fn upgrade_async(self) -> UpgradeFuture<'a, T, S, P, W> {
        let lock = self.lock;
        // Transfer the held upgradable flag into the future.
        core::mem::forget(self);
        UpgradeFuture { lock, waker_node_ticket: None, completed: false }
    }

    /// Downgrades to a plain read lock, keeping read access but releasing the upgrade privilege so
    /// another upgrader or writer may proceed.
    #[inline]
    pub fn downgrade(self) -> ReadGuard<'a, T, S, P, W> {
        let lock = self.lock;
        core::mem::forget(self);
        lock.upgradable_to_read();
        ReadGuard { lock }
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Deref
    for UpgradableReadGuard<'_, T, S, P, W>
{
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for UpgradableReadGuard<'_, T, S, P, W>
{
    fn drop(&mut self) {
        self.lock.upgrader_dropped();
    }
}

/// The future returned by [`Lock::upgradable_read_async`].
#[derive(Debug)]
pub struct UpgradableReadFuture<
    'a,
    T,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    lock: &'a Lock<T, S, P, W>,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Future
    for UpgradableReadFuture<'a, T, S, P, W>
{
    type Output = UpgradableReadGuard<'a, T, S, P, W>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();

        if let Some(guard) = this.lock.try_acquire_upgradable_ignoring_waiters() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_write_wakers().lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        {
            let mut queue = this.lock.get_write_wakers().lock();

            if let Some(guard) = this.lock.try_acquire_upgradable_ignoring_waiters() {
                if let Some(ticket) = this.waker_node_ticket.take()
                    && queue.remove(ticket)
                {
                    this.lock.sub_write_waker(Ordering::Relaxed);
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
                    this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                    this.lock.add_write_waker(Ordering::SeqCst);
                }
            } else {
                this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                this.lock.add_write_waker(Ordering::SeqCst);
            }
        }

        if let Some(guard) = this.lock.try_acquire_upgradable_ignoring_waiters() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_write_wakers().lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        Poll::Pending
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for UpgradableReadFuture<'_, T, S, P, W>
{
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.get_write_wakers().lock().remove(ticket)
        {
            self.lock.sub_write_waker(Ordering::Relaxed);
        }
    }
}

/// The future returned by [`UpgradableReadGuard::upgrade_async`].
#[derive(Debug)]
pub struct UpgradeFuture<
    'a,
    T,
    S: LockState,
    P: ParkStrategy = DefaultParkStrategy,
    W: WakerStorage<ASYNC_CAPACITY> = BoxedWakers<ASYNC_CAPACITY>,
> {
    lock: &'a Lock<T, S, P, W>,
    waker_node_ticket: Option<WakerTicket>,
    completed: bool,
}

impl<'a, T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Future
    for UpgradeFuture<'a, T, S, P, W>
{
    type Output = WriteGuard<'a, T, S, P, W>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();

        if this.lock.try_upgrade_inner() {
            this.completed = true;
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_write_wakers().lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(WriteGuard { lock: this.lock });
        }

        {
            let mut queue = this.lock.get_write_wakers().lock();

            if this.lock.try_upgrade_inner() {
                this.completed = true;
                if let Some(ticket) = this.waker_node_ticket.take()
                    && queue.remove(ticket)
                {
                    this.lock.sub_write_waker(Ordering::Relaxed);
                }
                return Poll::Ready(WriteGuard { lock: this.lock });
            }

            if let Some(ticket) = this.waker_node_ticket {
                let node = queue.node_mut(ticket.index());

                if node.generation() == ticket.generation() {
                    if node.waker().is_none_or(|w| !w.will_wake(cx.waker())) {
                        *node.waker_mut() = Some(cx.waker().clone());
                    }
                } else {
                    this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                    this.lock.add_write_waker(Ordering::SeqCst);
                }
            } else {
                this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                this.lock.add_write_waker(Ordering::SeqCst);
            }
        }

        if this.lock.try_upgrade_inner() {
            this.completed = true;
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.get_write_wakers().lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(WriteGuard { lock: this.lock });
        }

        Poll::Pending
    }
}

impl<T, S: LockState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Drop
    for UpgradeFuture<'_, T, S, P, W>
{
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.get_write_wakers().lock().remove(ticket)
        {
            self.lock.sub_write_waker(Ordering::Relaxed);
        }
        if !self.completed {
            // The future was cancelled mid-upgrade while still holding the upgradable flag; release
            // it so the lock does not stay stuck.
            self.lock.upgrader_dropped();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;
    use std::vec::Vec;

    // Use LockStateU64 directly under the hood to preserve all test cases identically
    type Lock<T> = super::Lock<T, crate::LockStateU64>;

    use super::*;

    // -------------------------------------------------------------------------
    // Blocking tests
    // -------------------------------------------------------------------------

    #[test]
    fn read_guard_allows_shared_access() {
        let lock = Lock::new(42);
        let a = lock.read();
        let b = lock.read();
        assert_eq!(*a, 42);
        assert_eq!(*b, 42);
    }

    #[test]
    fn write_guard_allows_mutation() {
        let lock = Lock::new(0);
        {
            let mut w = lock.write();
            *w = 99;
        }
        assert_eq!(*lock.read(), 99);
    }

    #[test]
    fn try_read_fails_while_writer_held() {
        let lock = Lock::new(());
        let _w = lock.write();
        assert!(lock.try_read().is_none());
    }

    #[test]
    fn try_write_fails_while_reader_held() {
        let lock = Lock::new(());
        let _r = lock.read();
        assert!(lock.try_write().is_none());
    }

    #[test]
    fn try_write_fails_while_writer_held() {
        let lock = Lock::new(());
        let _w = lock.write();
        assert!(lock.try_write().is_none());
    }

    #[test]
    fn try_read_succeeds_after_writer_dropped() {
        let lock = Lock::new(());
        let w = lock.write();
        drop(w);
        assert!(lock.try_read().is_some());
    }

    #[test]
    fn try_write_succeeds_after_reader_dropped() {
        let lock = Lock::new(());
        let r = lock.read();
        drop(r);
        assert!(lock.try_write().is_some());
    }

    #[test]
    fn blocking_read_unparks_after_writer_releases() {
        let lock = Arc::new(Lock::new(0u32));

        let w = lock.write();

        let lock2 = Arc::clone(&lock);
        let reader = std::thread::spawn(move || *lock2.read());

        // Give the reader thread time to park.
        std::thread::sleep(Duration::from_millis(20));
        drop(w);

        assert_eq!(reader.join().unwrap(), 0);
    }

    #[test]
    fn blocking_write_unparks_after_reader_releases() {
        let lock = Arc::new(Lock::new(0u32));

        let r = lock.read();

        let lock2 = Arc::clone(&lock);
        let writer = std::thread::spawn(move || {
            let mut g = lock2.write();
            *g = 7;
        });

        std::thread::sleep(Duration::from_millis(20));
        drop(r);

        writer.join().unwrap();
        assert_eq!(*lock.read(), 7);
    }

    #[test]
    fn blocking_write_unparks_after_writer_releases() {
        let lock = Arc::new(Lock::new(0u32));

        let w = lock.write();

        let lock2 = Arc::clone(&lock);
        let writer = std::thread::spawn(move || {
            let mut g = lock2.write();
            *g = 13;
        });

        std::thread::sleep(Duration::from_millis(20));
        drop(w);

        writer.join().unwrap();
        assert_eq!(*lock.read(), 13);
    }

    #[test]
    fn multiple_readers_unpark_concurrently_after_writer() {
        const READERS: usize = 8;
        let lock = Arc::new(Lock::new(()));
        let w = lock.write();

        let handles: Vec<_> = (0..READERS)
            .map(|_| {
                let l = Arc::clone(&lock);
                std::thread::spawn(move || {
                    let _r = l.read();
                })
            })
            .collect();

        std::thread::sleep(Duration::from_millis(20));
        drop(w);

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn concurrent_writes_are_serialised() {
        const THREADS: usize = 8;
        const INCREMENTS: usize = 100;

        let lock = Arc::new(Lock::new(0usize));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let l = Arc::clone(&lock);
                std::thread::spawn(move || {
                    for _ in 0..INCREMENTS {
                        *l.write() += 1;
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(*lock.read(), THREADS * INCREMENTS);
    }

    #[test]
    fn state_counters_return_to_zero_after_all_guards_dropped() {
        let lock = Lock::new(());

        {
            let _r1 = lock.read();
            let _r2 = lock.read();
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());

        {
            let _w = lock.write();
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }

    // -------------------------------------------------------------------------
    // Mapping + downgrade tests
    // -------------------------------------------------------------------------

    #[test]
    fn read_guard_map_projects_field() {
        let lock = Lock::new((7u32, 8u32));
        let g = ReadGuard::map(lock.read(), |t| &t.1);
        assert_eq!(*g, 8);
        // Other readers still allowed while a mapped read guard is held.
        assert_eq!(lock.read().1, 8);
    }

    #[test]
    fn read_guard_try_map_some_and_none() {
        let lock = Lock::new(Some(5u32));
        let mapped = ReadGuard::try_map(lock.read(), |t| t.as_ref());
        assert!(mapped.is_ok());
        assert_eq!(*mapped.unwrap(), 5);

        let lock2 = Lock::new(None::<u32>);
        let back = ReadGuard::try_map(lock2.read(), |t| t.as_ref());
        assert!(back.is_err());
        // Returned guard still works and releases cleanly.
        assert!(back.err().unwrap().is_none());
    }

    #[test]
    fn mapped_read_guard_further_map() {
        let lock = Lock::new((1u32, (2u32, 3u32)));
        let g = ReadGuard::map(lock.read(), |t| &t.1);
        let g = MappedReadGuard::map(g, |t| &t.1);
        assert_eq!(*g, 3);
    }

    #[test]
    fn write_guard_map_mutates_projection() {
        let lock = Lock::new((1u32, 2u32));
        {
            let mut g = WriteGuard::map(lock.write(), |t| &mut t.1);
            *g += 40;
        }
        assert_eq!(*lock.read(), (1, 42));
    }

    #[test]
    fn write_guard_try_map_some_and_none() {
        let lock = Lock::new(Some(1u32));
        {
            let mut g = WriteGuard::try_map(lock.write(), |t| t.as_mut()).ok().unwrap();
            *g += 9;
        }
        assert_eq!(*lock.read(), Some(10));

        let lock2 = Lock::new(None::<u32>);
        let back = WriteGuard::try_map(lock2.write(), |t| t.as_mut());
        assert!(back.is_err());
    }

    #[test]
    fn mapped_write_guard_further_map() {
        let lock = Lock::new((0u32, (0u32, 0u32)));
        {
            let g = WriteGuard::map(lock.write(), |t| &mut t.1);
            let mut g = MappedWriteGuard::map(g, |t| &mut t.1);
            *g = 99;
        }
        assert_eq!(*lock.read(), (0, (0, 99)));
    }

    #[test]
    fn write_downgrade_yields_read_and_wakes_readers() {
        let lock = Arc::new(Lock::new(0u32));
        let mut w = lock.write();
        *w = 5;

        // A reader that parks while the write lock is held.
        let lock2 = Arc::clone(&lock);
        let reader = std::thread::spawn(move || *lock2.read());
        std::thread::sleep(Duration::from_millis(20));

        // Downgrade hands us a read guard without releasing exclusivity to another writer.
        let r = w.downgrade();
        assert_eq!(*r, 5);
        // The parked reader is woken and observes the downgraded value.
        assert_eq!(reader.join().unwrap(), 5);
        drop(r);
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }

    // -------------------------------------------------------------------------
    // Upgradable read tests
    // -------------------------------------------------------------------------

    #[test]
    fn upgradable_allows_shared_readers_excludes_writers_and_upgraders() {
        let lock = Lock::new(1u32);
        let up = lock.upgradable_read();
        assert_eq!(*up, 1);
        // Shared readers are compatible with an upgradable reader.
        assert!(lock.try_read().is_some());
        assert_eq!(*lock.read(), 1);
        // Writers and other upgraders are excluded.
        assert!(lock.try_write().is_none());
        assert!(lock.try_upgradable_read().is_none());
        drop(up);
        // Fully released afterwards.
        assert!(lock.try_write().is_some());
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }

    #[test]
    fn try_upgrade_alone_succeeds_with_readers_fails() {
        let lock = Lock::new(10u32);

        // No other readers: upgrade succeeds.
        let up = lock.upgradable_read();
        let mut w = up.try_upgrade().ok().expect("no readers, upgrade should succeed");
        *w += 1;
        drop(w);
        assert_eq!(*lock.read(), 11);

        // A concurrent shared reader blocks a non-blocking upgrade.
        let up = lock.upgradable_read();
        let r = lock.read();
        let up = up.try_upgrade().err().expect("shared reader present, try_upgrade must fail");
        drop(r);
        // Once the reader is gone it can upgrade.
        assert!(up.try_upgrade().is_ok());
    }

    #[test]
    fn upgrade_waits_for_readers_then_is_exclusive() {
        let lock = Arc::new(Lock::new(0u32));
        let up = lock.upgradable_read();

        // A shared reader holds the lock briefly.
        let lock2 = Arc::clone(&lock);
        let reader = std::thread::spawn(move || {
            let r = lock2.read();
            std::thread::sleep(Duration::from_millis(30));
            *r
        });
        std::thread::sleep(Duration::from_millis(5));

        // Blocking upgrade waits for the reader to drain, then holds it exclusively.
        let mut w = up.upgrade();
        *w = 7;
        assert!(lock.try_read().is_none(), "write lock must be exclusive after upgrade");
        assert_eq!(reader.join().unwrap(), 0);
        drop(w);
        assert_eq!(*lock.read(), 7);
    }

    #[test]
    fn upgradable_downgrade_to_read_and_write_downgrade_to_upgradable() {
        let lock = Lock::new(3u32);

        // upgradable -> read: releases the upgrade privilege, so a new upgrader can appear.
        let up = lock.upgradable_read();
        let r = up.downgrade();
        assert_eq!(*r, 3);
        assert!(lock.try_upgradable_read().is_some());
        assert!(lock.try_write().is_none(), "still holding a read guard");
        drop(r);

        // write -> upgradable: keeps exclusion of writers, admits shared readers.
        let mut w = lock.write();
        *w = 9;
        let up = w.downgrade_to_upgradable();
        assert_eq!(*up, 9);
        assert!(lock.try_read().is_some());
        assert!(lock.try_write().is_none());
        drop(up);
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }

    #[test]
    fn upgrade_never_observes_intervening_write_under_contention() {
        // The core invariant: while a thread holds the upgradable lock, no writer can modify the
        // data, so the value it reads as an upgrader is unchanged when it upgrades.
        const UPGRADERS: usize = 4;
        const WRITERS: usize = 4;
        const READERS: usize = 4;
        const ITERS: usize = 300;

        let lock = Arc::new(Lock::new(0u64));

        let upgraders: Vec<_> = (0..UPGRADERS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                std::thread::spawn(move || {
                    for _ in 0..ITERS {
                        let up = lock.upgradable_read();
                        let seen = *up;
                        let mut w = up.upgrade();
                        assert_eq!(*w, seen, "a writer modified data while upgradable was held");
                        *w += 1;
                    }
                })
            })
            .collect();

        let writers: Vec<_> = (0..WRITERS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                std::thread::spawn(move || {
                    for _ in 0..ITERS {
                        *lock.write() += 1;
                    }
                })
            })
            .collect();

        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                std::thread::spawn(move || {
                    for _ in 0..ITERS {
                        std::hint::black_box(*lock.read());
                    }
                })
            })
            .collect();

        for h in upgraders {
            h.join().unwrap();
        }
        for h in writers {
            h.join().unwrap();
        }
        for h in readers {
            h.join().unwrap();
        }

        assert_eq!(*lock.read(), ((UPGRADERS + WRITERS) * ITERS) as u64);
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }

    // -------------------------------------------------------------------------
    // Async tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn async_read_returns_guard() {
        let lock = Lock::new(42u32);
        let g = lock.read_async().await;
        assert_eq!(*g, 42);
    }

    #[tokio::test]
    async fn async_write_returns_guard() {
        let lock = Lock::new(0u32);
        {
            let mut g = lock.write_async().await;
            *g = 55;
        }
        assert_eq!(*lock.read_async().await, 55);
    }

    #[tokio::test]
    async fn async_read_waits_for_writer() {
        let lock = Arc::new(Lock::new(0u32));

        let w = lock.write();

        let lock2 = Arc::clone(&lock);
        let reader = tokio::spawn(async move { *lock2.read_async().await });

        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(w);

        assert_eq!(reader.await.unwrap(), 0);
    }

    #[tokio::test]
    async fn async_write_waits_for_reader() {
        let lock = Arc::new(Lock::new(0u32));

        let r = lock.read();

        let lock2 = Arc::clone(&lock);
        let writer = tokio::spawn(async move {
            *lock2.write_async().await = 21;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(r);

        writer.await.unwrap();
        assert_eq!(*lock.read(), 21);
    }

    #[tokio::test]
    async fn async_write_waits_for_writer() {
        let lock = Arc::new(Lock::new(0u32));

        let w = lock.write();

        let lock2 = Arc::clone(&lock);
        let writer = tokio::spawn(async move {
            *lock2.write_async().await = 33;
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(w);

        writer.await.unwrap();
        assert_eq!(*lock.read(), 33);
    }

    #[tokio::test]
    async fn async_multiple_readers_resolve_after_writer() {
        const READERS: usize = 8;
        let lock = Arc::new(Lock::new(()));
        let w = lock.write();

        let handles: Vec<_> = (0..READERS)
            .map(|_| {
                let l = Arc::clone(&lock);
                tokio::spawn(async move { drop(l.read_async().await) })
            })
            .collect();

        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(w);

        for h in handles {
            h.await.unwrap();
        }
    }

    #[tokio::test]
    async fn async_concurrent_writes_are_serialised() {
        const TASKS: usize = 8;
        const INCREMENTS: usize = 100;

        let lock = Arc::new(Lock::new(0usize));

        let handles: Vec<_> = (0..TASKS)
            .map(|_| {
                let l = Arc::clone(&lock);
                tokio::spawn(async move {
                    for _ in 0..INCREMENTS {
                        *l.write_async().await += 1;
                    }
                })
            })
            .collect();

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(*lock.read(), TASKS * INCREMENTS);
    }

    #[tokio::test]
    async fn dropped_read_future_does_not_leak_waker_count() {
        let lock = Arc::new(Lock::new(()));

        // Hold a write guard so read_async will park.
        let w = lock.write();

        let lock2 = Arc::clone(&lock);
        let fut = tokio::spawn(async move {
            // select! drops the losing future, exercising ReadFuture::drop.
            tokio::select! {
                _g = lock2.read_async() => {},
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        });

        fut.await.unwrap();
        drop(w);

        // After the future is dropped the waker count must be back to zero.
        assert_eq!(
            lock.load_state(Ordering::Relaxed).read_wakers(),
            0,
            "read_waker count leaked after ReadFuture was dropped"
        );
    }

    #[tokio::test]
    async fn dropped_write_future_does_not_leak_waker_count() {
        let lock = Arc::new(Lock::new(()));

        let w = lock.write();

        let lock2 = Arc::clone(&lock);
        let fut = tokio::spawn(async move {
            tokio::select! {
                _g = lock2.write_async() => {},
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        });

        fut.await.unwrap();
        drop(w);

        assert_eq!(
            lock.load_state(Ordering::Relaxed).write_wakers(),
            0,
            "write_waker count leaked after WriteFuture was dropped"
        );
    }

    #[tokio::test]
    async fn mixed_blocking_and_async_writers_serialised() {
        const ASYNC_TASKS: usize = 4;
        const BLOCKING_THREADS: usize = 4;
        const INCREMENTS: usize = 50;

        let lock = Arc::new(Lock::new(0usize));

        let async_handles: Vec<_> = (0..ASYNC_TASKS)
            .map(|_| {
                let l = Arc::clone(&lock);
                tokio::spawn(async move {
                    for _ in 0..INCREMENTS {
                        *l.write_async().await += 1;
                    }
                })
            })
            .collect();

        let blocking_handles: Vec<_> = (0..BLOCKING_THREADS)
            .map(|_| {
                let l = Arc::clone(&lock);
                std::thread::spawn(move || {
                    for _ in 0..INCREMENTS {
                        *l.write() += 1;
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

        let expected = (ASYNC_TASKS + BLOCKING_THREADS) * INCREMENTS;
        assert_eq!(*lock.read(), expected);
    }

    #[tokio::test]
    async fn async_upgradable_read_and_upgrade() {
        let lock = Arc::new(Lock::new(0u32));

        let up = lock.upgradable_read_async().await;
        assert_eq!(*up, 0);

        // A shared reader briefly holds the lock; the async upgrade waits for it.
        let lock2 = Arc::clone(&lock);
        let reader = tokio::spawn(async move {
            let r = lock2.read_async().await;
            tokio::time::sleep(Duration::from_millis(20)).await;
            *r
        });
        tokio::time::sleep(Duration::from_millis(5)).await;

        let mut w = up.upgrade_async().await;
        *w = 42;
        assert_eq!(reader.await.unwrap(), 0);
        drop(w);
        assert_eq!(*lock.read(), 42);
    }

    #[tokio::test]
    async fn async_upgraders_are_mutually_exclusive() {
        let lock = Arc::new(Lock::new(0u64));
        const TASKS: usize = 6;
        const ITERS: usize = 100;

        let handles: Vec<_> = (0..TASKS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                tokio::spawn(async move {
                    for _ in 0..ITERS {
                        let up = lock.upgradable_read_async().await;
                        let seen = *up;
                        let mut w = up.upgrade_async().await;
                        assert_eq!(*w, seen);
                        *w += 1;
                    }
                })
            })
            .collect();

        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(*lock.read(), (TASKS * ITERS) as u64);
    }

    #[tokio::test]
    async fn cancelled_upgrade_future_releases_upgradable_lock() {
        let lock = Arc::new(Lock::new(0u32));

        // Hold a reader so the upgrade cannot complete, then drop the upgrade future.
        let r = lock.read();
        let up = lock.upgradable_read();
        {
            let fut = up.upgrade_async();
            tokio::pin!(fut);
            // Poll once so it registers as a waiter, then cancel by dropping.
            tokio::select! {
                _ = fut.as_mut() => panic!("should not complete while a reader is held"),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }
        drop(r);
        // The upgradable lock was released on cancellation, so the lock is fully free.
        assert!(lock.try_write().is_some());
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }

    #[tokio::test]
    async fn state_counters_return_to_zero_after_async_guards_dropped() {
        let lock = Lock::new(());

        {
            let _r1 = lock.read_async().await;
            let _r2 = lock.read_async().await;
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());

        {
            let _w = lock.write_async().await;
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }
}
