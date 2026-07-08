//! A condition variable [`Condvar`] pairing with [`Lock`](crate::Lock) write guards, usable from
//! blocking and async code.
//!
//! A condition variable lets a thread (or task) release a held lock and sleep until another party
//! signals that some condition may now hold, then re-acquire the lock. Unlike a hand-rolled
//! [`Notify`] + re-check, [`Condvar::wait`] does the release/register/re-acquire dance for you and,
//! crucially, registers interest *before* releasing the lock — so a signal that races the release
//! can never be lost.
//!
//! It pairs with the exclusive [`WriteGuard`] (the mutex-equivalent of the crate's reader/writer
//! [`Lock`](crate::Lock)), mirroring `std::sync::Condvar` / `parking_lot::Condvar`. The lock and
//! the condvar are independent objects with independent type parameters; a single condvar can serve
//! blocking and async waiters on the same lock at once.

use crate::lock::{ASYNC_CAPACITY as LOCK_ASYNC_CAPACITY, LockState, WriteGuard};
use crate::notify::{
    ASYNC_CAPACITY, Notify, NotifyState, NotifyStateU16, NotifyStateU32, NotifyStateU64,
};
use crate::park_strategy::{DefaultParkStrategy, ParkStrategy};
use crate::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};

// Async waker-storage variants. The bare `Condvar{16,32,64}` names alias the default representation
// (inline: allocation-free, larger struct); the `Inline`/`Boxed` names select the representation
// explicitly. Inline is the default because the condvar's wait/notify path is `Notify`-backed
// (which itself defaults to inline). See [`WakerStorage`].

/// [`Condvar16`] with inline waker storage (the default): larger struct, but allocation-free and
/// indirection-free on the async path.
pub type Condvar16Inline<P = DefaultParkStrategy> =
    Condvar<NotifyStateU16, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Condvar32`]. See [`Condvar16Inline`].
pub type Condvar32Inline<P = DefaultParkStrategy> =
    Condvar<NotifyStateU32, P, InlineWakers<ASYNC_CAPACITY>>;
/// Inline-waker variant of [`Condvar64`]. See [`Condvar16Inline`].
pub type Condvar64Inline<P = DefaultParkStrategy> =
    Condvar<NotifyStateU64, P, InlineWakers<ASYNC_CAPACITY>>;

/// [`Condvar16`] with boxed waker storage: pointer-sized struct that allocates its waker queue
/// lazily (and never at all for blocking-only usage).
pub type Condvar16Boxed<P = DefaultParkStrategy> =
    Condvar<NotifyStateU16, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Condvar32`]. See [`Condvar16Boxed`].
pub type Condvar32Boxed<P = DefaultParkStrategy> =
    Condvar<NotifyStateU32, P, BoxedWakers<ASYNC_CAPACITY>>;
/// Boxed-waker variant of [`Condvar64`]. See [`Condvar16Boxed`].
pub type Condvar64Boxed<P = DefaultParkStrategy> =
    Condvar<NotifyStateU64, P, BoxedWakers<ASYNC_CAPACITY>>;

pub type Condvar16<P = DefaultParkStrategy> = Condvar16Inline<P>;
pub type Condvar32<P = DefaultParkStrategy> = Condvar32Inline<P>;
pub type Condvar64<P = DefaultParkStrategy> = Condvar64Inline<P>;

/// A condition variable supporting blocking and async waiters, paired with [`Lock`](crate::Lock)
/// write guards.
///
/// [`wait`](Condvar::wait) / [`wait_async`](Condvar::wait_async) atomically release the supplied
/// [`WriteGuard`], sleep until signalled by [`notify_one`](Condvar::notify_one) /
/// [`notify_all`](Condvar::notify_all), then re-acquire the lock and hand the guard back. As is
/// standard for condition variables, a wake does not by itself prove the condition holds (another
/// waiter may have consumed it, and — like every condvar — a wake can be spurious), so callers
/// re-check in a loop; [`wait_while`](Condvar::wait_while) encapsulates that loop.
///
/// The backing [`Notify`] snapshots its epoch when the listener is created, which happens *before*
/// the guard is released. A notifier must first take the lock the waiter still holds, so its epoch
/// bump is ordered after the snapshot — the wake can never be lost.
///
/// The `S` width selects the [`NotifyState`] backing the epoch counter (the default,
/// [`NotifyStateU64`], gives the widest epoch and so the most signals before it wraps). The `W`
/// parameter selects how the async waker queue is stored — [`InlineWakers`] (the default) or
/// [`BoxedWakers`] (pointer-sized, allocates lazily); it has no effect on the blocking path. See
/// [`WakerStorage`] and the [`Condvar16Inline`] / [`Condvar16Boxed`] aliases.
pub struct Condvar<
    S: NotifyState = NotifyStateU64,
    P = DefaultParkStrategy,
    W = InlineWakers<ASYNC_CAPACITY>,
> {
    notify: Notify<S, P, W>,
}

impl<S: NotifyState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Default
    for Condvar<S, P, W>
{
    fn default() -> Self {
        Self::with_park_strategy()
    }
}

impl<S: NotifyState, W: WakerStorage<ASYNC_CAPACITY>> Condvar<S, DefaultParkStrategy, W> {
    /// Creates a new condition variable.
    pub const fn new() -> Self {
        Self::with_park_strategy()
    }
}

impl<S: NotifyState, P: ParkStrategy, W: WakerStorage<ASYNC_CAPACITY>> Condvar<S, P, W> {
    /// Creates a condition variable with an explicit [`ParkStrategy`].
    pub const fn with_park_strategy() -> Self {
        Self { notify: Notify::with_park_strategy() }
    }

    /// Wakes at most one waiter currently blocked in [`wait`](Condvar::wait) /
    /// [`wait_async`](Condvar::wait_async).
    ///
    /// As with every condition variable, the woken party must re-check its condition: it is not
    /// guaranteed to be the one whose condition became true.
    #[inline]
    pub fn notify_one(&self) {
        self.notify.notify(1);
    }

    /// Wakes every waiter currently blocked in [`wait`](Condvar::wait) /
    /// [`wait_async`](Condvar::wait_async).
    #[inline]
    pub fn notify_all(&self) {
        self.notify.notify(usize::MAX);
    }

    /// Returns `true` if at least one waiter is currently registered.
    #[inline]
    pub fn has_waiters(&self) -> bool {
        self.notify.has_listeners()
    }

    /// Atomically releases `guard` and blocks until notified, then re-acquires the lock and returns
    /// the guard.
    ///
    /// A wake does not guarantee the condition holds — re-check it and call `wait` again if needed,
    /// or use [`wait_while`](Condvar::wait_while).
    pub fn wait<'a, T, LS, LP, LW>(
        &self,
        guard: WriteGuard<'a, T, LS, LP, LW>,
    ) -> WriteGuard<'a, T, LS, LP, LW>
    where
        LS: LockState,
        LP: ParkStrategy,
        LW: WakerStorage<LOCK_ASYNC_CAPACITY>,
    {
        // Register interest *before* releasing the lock: a notifier must first acquire the lock we
        // still hold, so its epoch bump is ordered after this snapshot — the wake can't be lost.
        let listener = self.notify.listener();
        // SAFETY: the lock outlives the guard, so the returned `&'a Lock` stays valid after we drop
        // the guard; we use it only to re-acquire once the wait completes.
        let lock = unsafe { WriteGuard::get_lock(&guard) };
        drop(guard);
        listener.wait();
        lock.write()
    }

    /// Atomically releases `guard` and awaits a notification, then re-acquires the lock and returns
    /// the guard.
    ///
    /// The async counterpart of [`wait`](Condvar::wait); the same re-check discipline applies.
    pub async fn wait_async<'a, T, LS, LP, LW>(
        &self,
        guard: WriteGuard<'a, T, LS, LP, LW>,
    ) -> WriteGuard<'a, T, LS, LP, LW>
    where
        LS: LockState,
        LP: ParkStrategy,
        LW: WakerStorage<LOCK_ASYNC_CAPACITY>,
    {
        let listener = self.notify.listener();
        // SAFETY: as in `wait` — the lock outlives the guard, and we only re-acquire after the
        // guard is dropped.
        let lock = unsafe { WriteGuard::get_lock(&guard) };
        drop(guard);
        listener.await;
        lock.write_async().await
    }

    /// Blocks until `condition` (evaluated against the guarded value) returns `false`, releasing
    /// and re-acquiring the lock across each wait. Returns the re-acquired guard.
    ///
    /// This is the loop callers should almost always use: it re-checks `condition` on every wake,
    /// so it is immune to spurious wakeups and to another waiter having consumed the signal
    /// first.
    pub fn wait_while<'a, T, LS, LP, LW, F>(
        &self,
        mut guard: WriteGuard<'a, T, LS, LP, LW>,
        mut condition: F,
    ) -> WriteGuard<'a, T, LS, LP, LW>
    where
        LS: LockState,
        LP: ParkStrategy,
        LW: WakerStorage<LOCK_ASYNC_CAPACITY>,
        F: FnMut(&mut T) -> bool,
    {
        while condition(&mut guard) {
            guard = self.wait(guard);
        }
        guard
    }

    /// The async counterpart of [`wait_while`](Condvar::wait_while).
    pub async fn wait_while_async<'a, T, LS, LP, LW, F>(
        &self,
        mut guard: WriteGuard<'a, T, LS, LP, LW>,
        mut condition: F,
    ) -> WriteGuard<'a, T, LS, LP, LW>
    where
        LS: LockState,
        LP: ParkStrategy,
        LW: WakerStorage<LOCK_ASYNC_CAPACITY>,
        F: FnMut(&mut T) -> bool,
    {
        while condition(&mut guard) {
            guard = self.wait_async(guard).await;
        }
        guard
    }
}

impl<S: NotifyState, P, W> core::fmt::Debug for Condvar<S, P, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Condvar").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use std::vec::Vec;

    use super::*;

    // Pair the condvar against the crate's own exclusive lock, exercising the public default alias.
    type Lock<T> = crate::lock::Lock<T, crate::LockStateU64>;
    type Condvar = super::Condvar;

    #[test]
    fn notify_one_wakes_blocking_waiter() {
        let pair = Arc::new((Lock::new(false), Condvar::new()));

        let pair2 = Arc::clone(&pair);
        let waiter = std::thread::spawn(move || {
            let (lock, cvar) = &*pair2;
            let mut ready = lock.write();
            while !*ready {
                ready = cvar.wait(ready);
            }
        });

        std::thread::sleep(Duration::from_millis(20));
        {
            let (lock, cvar) = &*pair;
            *lock.write() = true;
            cvar.notify_one();
        }
        waiter.join().unwrap();
    }

    #[test]
    fn wait_while_blocks_until_condition_false() {
        let pair = Arc::new((Lock::new(0u32), Condvar::new()));

        let pair2 = Arc::clone(&pair);
        let waiter = std::thread::spawn(move || {
            let (lock, cvar) = &*pair2;
            let guard = cvar.wait_while(lock.write(), |v| *v < 3);
            *guard
        });

        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(5));
            let (lock, cvar) = &*pair;
            *lock.write() += 1;
            cvar.notify_one();
        }

        assert_eq!(waiter.join().unwrap(), 3);
    }

    #[test]
    fn notify_all_wakes_every_waiter() {
        const WAITERS: usize = 6;
        let pair = Arc::new((Lock::new(false), Condvar::new()));
        let woken = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..WAITERS)
            .map(|_| {
                let pair = Arc::clone(&pair);
                let woken = Arc::clone(&woken);
                std::thread::spawn(move || {
                    let (lock, cvar) = &*pair;
                    let guard = cvar.wait_while(lock.write(), |ready| !*ready);
                    drop(guard);
                    woken.fetch_add(1, Ordering::Relaxed);
                })
            })
            .collect();

        std::thread::sleep(Duration::from_millis(30));
        {
            let (lock, cvar) = &*pair;
            *lock.write() = true;
            cvar.notify_all();
        }

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(woken.load(Ordering::Relaxed), WAITERS);
    }

    #[test]
    fn notify_one_with_no_waiters_is_a_noop() {
        let cvar = Condvar::new();
        cvar.notify_one();
        cvar.notify_all();
        assert!(!cvar.has_waiters());
    }

    #[tokio::test]
    async fn async_notify_one_wakes_waiter() {
        let pair = Arc::new((Lock::new(false), Condvar::new()));

        let pair2 = Arc::clone(&pair);
        let waiter = tokio::spawn(async move {
            let (lock, cvar) = &*pair2;
            let guard = cvar.wait_while_async(lock.write_async().await, |ready| !*ready).await;
            *guard
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        {
            let (lock, cvar) = &*pair;
            *lock.write_async().await = true;
            cvar.notify_one();
        }
        assert!(waiter.await.unwrap());
    }

    #[tokio::test]
    async fn async_notify_all_wakes_every_waiter() {
        const WAITERS: usize = 6;
        let pair = Arc::new((Lock::new(false), Condvar::new()));

        let handles: Vec<_> = (0..WAITERS)
            .map(|_| {
                let pair = Arc::clone(&pair);
                tokio::spawn(async move {
                    let (lock, cvar) = &*pair;
                    let guard =
                        cvar.wait_while_async(lock.write_async().await, |ready| !*ready).await;
                    *guard
                })
            })
            .collect();

        tokio::time::sleep(Duration::from_millis(30)).await;
        {
            let (lock, cvar) = &*pair;
            *lock.write_async().await = true;
            cvar.notify_all();
        }

        for h in handles {
            assert!(h.await.unwrap());
        }
    }

    #[tokio::test]
    async fn mixed_blocking_and_async_waiters_share_a_condvar() {
        let pair = Arc::new((Lock::new(false), Condvar::new()));

        let pb = Arc::clone(&pair);
        let blocking = std::thread::spawn(move || {
            let (lock, cvar) = &*pb;
            drop(cvar.wait_while(lock.write(), |ready| !*ready));
        });

        let pa = Arc::clone(&pair);
        let asyncing = tokio::spawn(async move {
            let (lock, cvar) = &*pa;
            drop(cvar.wait_while_async(lock.write_async().await, |ready| !*ready).await);
        });

        tokio::time::sleep(Duration::from_millis(30)).await;
        {
            let (lock, cvar) = &*pair;
            *lock.write() = true;
            cvar.notify_all();
        }

        asyncing.await.unwrap();
        blocking.join().unwrap();
    }

    #[test]
    fn inline_storage_is_default() {
        assert_eq!(size_of::<Condvar32>(), size_of::<Condvar32Inline>());
        assert!(size_of::<Condvar32Boxed>() < size_of::<Condvar32Inline>());
    }

    #[tokio::test]
    async fn boxed_storage_works() {
        let pair = Arc::new((Lock::new(false), Condvar32Boxed::new()));

        let pair2 = Arc::clone(&pair);
        let waiter = tokio::spawn(async move {
            let (lock, cvar) = &*pair2;
            drop(cvar.wait_while_async(lock.write_async().await, |ready| !*ready).await);
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        {
            let (lock, cvar) = &*pair;
            *lock.write() = true;
            cvar.notify_all();
        }
        waiter.await.unwrap();
    }
}
