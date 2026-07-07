//! One-time initialisation: [`Once`], [`OnceCell`], and [`Lazy`].
//!
//! All three run their initialiser exactly once and let other callers wait for it — from blocking
//! code (`call_once` / `get_or_init`) or async code (`call_once_async` / `get_or_init_async`) —
//! reusing [`Notify`](crate::Notify) for the "wait for the other initialiser" path. Fallible
//! `try_*` variants leave the primitive uninitialised on error so a later call retries (matching
//! `std::sync::OnceLock`; no poisoning).

use core::cell::UnsafeCell;
use core::convert::Infallible;
use core::future::Future;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::sync::atomic::{AtomicU8, Ordering};

use crate::Notify32;

const INCOMPLETE: u8 = 0;
const RUNNING: u8 = 1;
const COMPLETE: u8 = 2;

/// A primitive that runs an initialiser exactly once.
///
/// The first caller runs the closure while others wait; on success the `Once` is permanently
/// complete. If the closure fails ([`call_once_try`](Once::call_once_try)) or panics, the `Once`
/// returns to the incomplete state so a later call can retry.
#[derive(Debug)]
pub struct Once {
    state: AtomicU8,
    notify: Notify32,
}

impl Default for Once {
    fn default() -> Self {
        Self::new()
    }
}

/// Restores the incomplete state (and wakes waiters) if the running initialiser is abandoned before
/// completing — i.e. it returned `Err` or panicked. Defused on success.
struct RunGuard<'a> {
    once: &'a Once,
    defused: bool,
}

impl RunGuard<'_> {
    fn complete(mut self) {
        self.once.state.store(COMPLETE, Ordering::Release);
        self.once.notify.notify(usize::MAX);
        self.defused = true;
    }
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        if !self.defused {
            self.once.state.store(INCOMPLETE, Ordering::Release);
            self.once.notify.notify(usize::MAX);
        }
    }
}

impl Once {
    /// Creates a new, incomplete `Once`.
    pub const fn new() -> Self {
        Self { state: AtomicU8::new(INCOMPLETE), notify: Notify32::new() }
    }

    /// Returns `true` once the initialiser has completed successfully.
    #[inline]
    pub fn is_completed(&self) -> bool {
        self.state.load(Ordering::Acquire) == COMPLETE
    }

    /// Runs `f` exactly once, blocking until initialisation has completed (by this or another
    /// caller).
    #[inline]
    pub fn call_once<F: FnOnce()>(&self, f: F) {
        if self.is_completed() {
            return;
        }
        let _ = self.run_blocking(|| {
            f();
            Ok::<(), Infallible>(())
        });
    }

    /// Runs `f` exactly once, blocking. If `f` returns `Err`, the `Once` stays incomplete (a later
    /// call retries) and the error is returned.
    #[inline]
    pub fn call_once_try<E, F: FnOnce() -> Result<(), E>>(&self, f: F) -> Result<(), E> {
        if self.is_completed() {
            return Ok(());
        }
        self.run_blocking(f)
    }

    fn run_blocking<E, F: FnOnce() -> Result<(), E>>(&self, f: F) -> Result<(), E> {
        loop {
            match self.state.compare_exchange(
                INCOMPLETE,
                RUNNING,
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let guard = RunGuard { once: self, defused: false };
                    f()?;
                    guard.complete();
                    return Ok(());
                }
                Err(COMPLETE) => return Ok(()),
                Err(state) => {
                    // RUNNING: wait for the runner. INCOMPLETE (a reset): loop and retry the CAS.
                    if state == RUNNING {
                        let listener = self.notify.listener();
                        if self.state.load(Ordering::Acquire) == RUNNING {
                            listener.wait();
                        }
                    }
                }
            }
        }
    }

    /// Runs the future produced by `f` exactly once, awaiting until initialisation has completed.
    #[inline]
    pub async fn call_once_async<Fut, F>(&self, f: F)
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = ()>,
    {
        if self.is_completed() {
            return;
        }
        let _ = self
            .run_async(|| async move {
                f().await;
                Ok::<(), Infallible>(())
            })
            .await;
    }

    /// Runs the future produced by `f` exactly once, awaiting. If it resolves to `Err`, the `Once`
    /// stays incomplete and the error is returned.
    #[inline]
    pub async fn call_once_try_async<E, Fut, F>(&self, f: F) -> Result<(), E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(), E>>,
    {
        if self.is_completed() {
            return Ok(());
        }
        self.run_async(f).await
    }

    async fn run_async<E, Fut, F>(&self, f: F) -> Result<(), E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(), E>>,
    {
        // `f` is consumed by the single runner; hold it in an Option so the loop can move it out.
        let mut f = Some(f);
        loop {
            match self.state.compare_exchange(
                INCOMPLETE,
                RUNNING,
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let guard = RunGuard { once: self, defused: false };
                    (f.take().expect("runner claimed the initialiser once"))().await?;
                    guard.complete();
                    return Ok(());
                }
                Err(COMPLETE) => return Ok(()),
                Err(state) => {
                    if state == RUNNING {
                        let listener = self.notify.listener();
                        if self.state.load(Ordering::Acquire) == RUNNING {
                            listener.await;
                        }
                    }
                }
            }
        }
    }

    /// Blocks until the initialiser has completed successfully (by another caller).
    pub fn wait(&self) {
        loop {
            if self.is_completed() {
                return;
            }
            let listener = self.notify.listener();
            if self.is_completed() {
                return;
            }
            listener.wait();
        }
    }

    /// Resolves once the initialiser has completed successfully (by another caller).
    pub async fn wait_async(&self) {
        loop {
            if self.is_completed() {
                return;
            }
            let listener = self.notify.listener();
            if self.is_completed() {
                return;
            }
            listener.await;
        }
    }
}

/// A cell holding a value written at most once, with blocking and async initialisation.
///
/// A drop-in-style alternative to `std::sync::OnceLock` / `once_cell` that also supports async
/// initialisation.
pub struct OnceCell<T> {
    once: Once,
    value: UnsafeCell<MaybeUninit<T>>,
}

// SAFETY: access to `value` is gated by `once`'s state machine; the value is written once under the
// RUNNING → COMPLETE transition and only read after (via an Acquire load of COMPLETE).
unsafe impl<T: Send> Send for OnceCell<T> {}
unsafe impl<T: Send + Sync> Sync for OnceCell<T> {}

impl<T> Default for OnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for OnceCell<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OnceCell").field("value", &self.get()).finish()
    }
}

impl<T> OnceCell<T> {
    /// Creates an empty cell.
    pub const fn new() -> Self {
        Self { once: Once::new(), value: UnsafeCell::new(MaybeUninit::uninit()) }
    }

    /// Creates a cell already holding `value`.
    pub fn with_value(value: T) -> Self {
        let cell = Self::new();
        cell.once.state.store(COMPLETE, Ordering::Release);
        unsafe { (*cell.value.get()).write(value) };
        cell
    }

    /// SAFETY: the caller must ensure the cell is initialised.
    #[inline]
    unsafe fn get_unchecked(&self) -> &T {
        unsafe { (*self.value.get()).assume_init_ref() }
    }

    /// Returns a reference to the value if the cell has been initialised.
    #[inline]
    pub fn get(&self) -> Option<&T> {
        if self.once.is_completed() {
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value if the cell has been initialised.
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if *self.once.state.get_mut() == COMPLETE {
            Some(unsafe { (*self.value.get()).assume_init_mut() })
        } else {
            None
        }
    }

    /// Sets the value if the cell is empty. Returns `Err(value)` if it was already initialised.
    pub fn set(&self, value: T) -> Result<(), T> {
        match self.try_insert(value) {
            Ok(_) => Ok(()),
            Err((_, value)) => Err(value),
        }
    }

    /// Sets the value and returns a reference to it if the cell is empty; otherwise returns the
    /// existing reference together with the rejected `value`.
    pub fn try_insert(&self, value: T) -> Result<&T, (&T, T)> {
        let mut value = Some(value);
        self.once.call_once(|| unsafe {
            (*self.value.get()).write(value.take().expect("initialiser runs at most once"));
        });
        match value {
            None => Ok(unsafe { self.get_unchecked() }),
            Some(value) => Err((unsafe { self.get_unchecked() }, value)),
        }
    }

    /// Returns the value, initialising it with `f` (blocking) if the cell is empty.
    pub fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
        let _ = self.once.call_once_try(|| {
            unsafe { (*self.value.get()).write(f()) };
            Ok::<(), Infallible>(())
        });
        unsafe { self.get_unchecked() }
    }

    /// Returns the value, initialising it with `f` (blocking) if the cell is empty. If `f` returns
    /// `Err`, the cell stays empty and the error is returned.
    pub fn get_or_try_init<E, F: FnOnce() -> Result<T, E>>(&self, f: F) -> Result<&T, E> {
        self.once.call_once_try(|| {
            unsafe { (*self.value.get()).write(f()?) };
            Ok(())
        })?;
        Ok(unsafe { self.get_unchecked() })
    }

    /// Returns the value, awaiting `f` to initialise it if the cell is empty.
    pub async fn get_or_init_async<Fut, F>(&self, f: F) -> &T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        self.once
            .call_once_async(|| async {
                let value = f().await;
                unsafe { (*self.value.get()).write(value) };
            })
            .await;
        unsafe { self.get_unchecked() }
    }

    /// Returns the value, awaiting `f` to initialise it if the cell is empty. If `f` resolves to
    /// `Err`, the cell stays empty and the error is returned.
    pub async fn get_or_try_init_async<E, Fut, F>(&self, f: F) -> Result<&T, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        self.once
            .call_once_try_async(|| async {
                let value = f().await?;
                unsafe { (*self.value.get()).write(value) };
                Ok(())
            })
            .await?;
        Ok(unsafe { self.get_unchecked() })
    }

    /// Blocks until the cell is initialised (by another caller) and returns the value.
    pub fn wait(&self) -> &T {
        self.once.wait();
        unsafe { self.get_unchecked() }
    }

    /// Resolves once the cell is initialised (by another caller) and returns the value.
    pub async fn wait_async(&self) -> &T {
        self.once.wait_async().await;
        unsafe { self.get_unchecked() }
    }

    /// Takes the value out of the cell, leaving it empty. Requires unique access.
    pub fn take(&mut self) -> Option<T> {
        if *self.once.state.get_mut() == COMPLETE {
            *self.once.state.get_mut() = INCOMPLETE;
            Some(unsafe { (*self.value.get()).assume_init_read() })
        } else {
            None
        }
    }

    /// Consumes the cell, returning the value if it was initialised.
    pub fn into_inner(mut self) -> Option<T> {
        self.take()
    }
}

impl<T> Drop for OnceCell<T> {
    fn drop(&mut self) {
        if *self.once.state.get_mut() == COMPLETE {
            unsafe { (*self.value.get()).assume_init_drop() };
        }
    }
}

/// A value initialised on first access by a stored closure.
///
/// Like `std::sync::LazyLock` / `once_cell::sync::Lazy`: [`Deref`] runs the closure once and yields
/// the value thereafter.
pub struct Lazy<T, F = fn() -> T> {
    cell: OnceCell<T>,
    init: UnsafeCell<Option<F>>,
}

// SAFETY: `init` is taken exactly once, by whichever thread wins the `OnceCell` initialisation, so
// it never aliases. `T: Send + Sync` covers the shared `&T`; `F: Send` covers moving the closure to
// the initialising thread.
unsafe impl<T: Send + Sync, F: Send> Sync for Lazy<T, F> {}

impl<T, F> Lazy<T, F> {
    /// Creates a `Lazy` that will run `f` on first access.
    pub const fn new(f: F) -> Self {
        Self { cell: OnceCell::new(), init: UnsafeCell::new(Some(f)) }
    }

    /// Returns the value if it has already been initialised.
    #[inline]
    pub fn get(this: &Lazy<T, F>) -> Option<&T> {
        this.cell.get()
    }
}

impl<T, F: FnOnce() -> T> Lazy<T, F> {
    /// Forces initialisation (if it has not happened yet) and returns the value.
    pub fn force(this: &Lazy<T, F>) -> &T {
        this.cell.get_or_init(|| {
            let f = unsafe { (*this.init.get()).take() }.expect("Lazy initialised more than once");
            f()
        })
    }
}

impl<T, F: FnOnce() -> T> Deref for Lazy<T, F> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        Lazy::force(self)
    }
}

impl<T: core::fmt::Debug, F> core::fmt::Debug for Lazy<T, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lazy").field("value", &Lazy::get(self)).finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn call_once_runs_exactly_once() {
        let once = Once::new();
        let count = AtomicUsize::new(0);
        for _ in 0..5 {
            once.call_once(|| {
                count.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(count.load(Ordering::Relaxed), 1);
        assert!(once.is_completed());
    }

    #[test]
    fn call_once_concurrent_runs_once() {
        const THREADS: usize = 8;
        let once = Arc::new(Once::new());
        let count = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(std::sync::Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let once = Arc::clone(&once);
                let count = Arc::clone(&count);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    once.call_once(|| {
                        count.fetch_add(1, Ordering::Relaxed);
                    });
                    assert!(once.is_completed());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn call_once_try_retries_after_error() {
        let once = Once::new();
        assert_eq!(once.call_once_try(|| Err::<(), _>("boom")), Err("boom"));
        assert!(!once.is_completed());
        // A later call gets to run because the first failed.
        let mut ran = false;
        once.call_once_try::<(), _>(|| {
            ran = true;
            Ok(())
        })
        .unwrap();
        assert!(ran);
        assert!(once.is_completed());
    }

    #[test]
    fn oncecell_get_set() {
        let cell = OnceCell::new();
        assert_eq!(cell.get(), None);
        assert_eq!(cell.set(10u32), Ok(()));
        assert_eq!(cell.get(), Some(&10));
        assert_eq!(cell.set(20), Err(20));
        assert_eq!(cell.get(), Some(&10));
    }

    #[test]
    fn oncecell_get_or_init_runs_once() {
        let cell = OnceCell::new();
        let count = AtomicUsize::new(0);
        let a = *cell.get_or_init(|| {
            count.fetch_add(1, Ordering::Relaxed);
            7u32
        });
        let b = *cell.get_or_init(|| {
            count.fetch_add(1, Ordering::Relaxed);
            99u32
        });
        assert_eq!((a, b), (7, 7));
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn oncecell_get_or_try_init_stays_empty_on_error() {
        let cell = OnceCell::<u32>::new();
        assert_eq!(cell.get_or_try_init(|| Err::<u32, _>("no")), Err("no"));
        assert_eq!(cell.get(), None);
        assert_eq!(cell.get_or_try_init(|| Ok::<_, &str>(5)), Ok(&5));
        assert_eq!(cell.get(), Some(&5));
    }

    #[test]
    fn oncecell_take_and_into_inner() {
        let mut cell = OnceCell::new();
        cell.set(3u32).unwrap();
        assert_eq!(cell.take(), Some(3));
        assert_eq!(cell.get(), None);
        cell.set(4).unwrap();
        assert_eq!(cell.into_inner(), Some(4));
    }

    #[test]
    fn oncecell_drops_value() {
        let dropped = Arc::new(AtomicUsize::new(0));
        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        {
            let cell = OnceCell::new();
            cell.set(Guard(Arc::clone(&dropped))).ok();
        }
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn oncecell_wait_unblocks_after_init() {
        let cell = Arc::new(OnceCell::<u32>::new());
        let waiter = {
            let cell = Arc::clone(&cell);
            std::thread::spawn(move || *cell.wait())
        };
        std::thread::sleep(Duration::from_millis(20));
        cell.set(42).unwrap();
        assert_eq!(waiter.join().unwrap(), 42);
    }

    #[test]
    fn lazy_initialises_on_deref() {
        let count = AtomicUsize::new(0);
        let lazy = Lazy::new(|| {
            count.fetch_add(1, Ordering::Relaxed);
            41u32 + 1
        });
        assert_eq!(Lazy::get(&lazy), None);
        assert_eq!(*lazy, 42);
        assert_eq!(*lazy, 42);
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn oncecell_get_or_init_async() {
        let cell = OnceCell::new();
        let value = *cell
            .get_or_init_async(|| async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                123u32
            })
            .await;
        assert_eq!(value, 123);
        assert_eq!(cell.get(), Some(&123));
    }

    #[tokio::test]
    async fn oncecell_get_or_try_init_async_error_retries() {
        let cell = OnceCell::<u32>::new();
        let r = cell.get_or_try_init_async(|| async { Err::<u32, &str>("nope") }).await;
        assert_eq!(r, Err("nope"));
        assert_eq!(cell.get(), None);
        let r = cell.get_or_try_init_async(|| async { Ok::<_, &str>(9) }).await;
        assert_eq!(r, Ok(&9));
    }

    #[tokio::test]
    async fn oncecell_async_init_is_shared() {
        let cell: Arc<OnceCell<u32>> = Arc::new(OnceCell::new());
        let count = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let cell = Arc::clone(&cell);
                let count = Arc::clone(&count);
                tokio::spawn(async move {
                    *cell
                        .get_or_init_async(|| async {
                            count.fetch_add(1, Ordering::Relaxed);
                            tokio::time::sleep(Duration::from_millis(10)).await;
                            5u32
                        })
                        .await
                })
            })
            .collect();

        for t in tasks {
            assert_eq!(t.await.unwrap(), 5);
        }
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}
