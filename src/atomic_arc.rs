//! A lock-free cell holding the latest published shared value.
//!
//! [`AtomicArc`] lets many readers [`load`](AtomicArc::load) a consistent snapshot while a writer
//! [`store`](AtomicArc::store)s a new one, without either side blocking the other on a lock. It is
//! what [`ObservableLock`](crate::ObservableLock) publishes its value through: a plain
//! writer-preference [`Lock`](crate::Lock) starves the reader under a steady stream of writers,
//! which this avoids.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::Backoff;

/// A shared pointer (an `Arc`) that this module can hold as a raw pointer and rebuild.
///
/// Implemented for the standard-library [`Arc`](alloc::sync::Arc) and, with the `triomphe-arc`
/// feature, for [`triomphe::Arc`], so [`AtomicArc`] works with either exactly as the rest of the
/// crate's owned handles do.
///
/// # Safety
///
/// Implementors must guarantee that:
/// - [`into_raw`](RawArc::into_raw) yields a pointer carrying one owned strong count, and
///   [`from_raw`](RawArc::from_raw) reclaims exactly that count, round-tripping losslessly;
/// - [`Clone`] bumps a *shared* strong count, so the pointee stays live while any clone exists.
pub unsafe trait RawArc: Clone {
    /// The pointee type.
    type Target;

    /// Wraps `value` in a new handle carrying one strong count.
    fn new(value: Self::Target) -> Self;

    /// Consumes the handle into its raw pointer, transferring one owned strong count to the caller.
    fn into_raw(this: Self) -> *const Self::Target;

    /// Rebuilds the handle from a pointer previously produced by [`into_raw`](RawArc::into_raw).
    ///
    /// # Safety
    ///
    /// `ptr` must have come from [`into_raw`](RawArc::into_raw) and not yet been reclaimed by a
    /// prior `from_raw`.
    unsafe fn from_raw(ptr: *const Self::Target) -> Self;
}

// SAFETY: `Arc::into_raw`/`from_raw` round-trip exactly one strong count and `Arc::clone` bumps the
// shared count, as required.
unsafe impl<T> RawArc for alloc::sync::Arc<T> {
    type Target = T;

    #[inline(always)]
    fn new(value: T) -> Self {
        alloc::sync::Arc::new(value)
    }

    #[inline(always)]
    fn into_raw(this: Self) -> *const T {
        alloc::sync::Arc::into_raw(this)
    }

    #[inline(always)]
    unsafe fn from_raw(ptr: *const T) -> Self {
        // SAFETY: forwarded from this trait method's contract.
        unsafe { alloc::sync::Arc::from_raw(ptr) }
    }
}

// SAFETY: as for the standard-library `Arc` impl above.
#[cfg(feature = "triomphe-arc")]
unsafe impl<T> RawArc for triomphe::Arc<T> {
    type Target = T;

    #[inline(always)]
    fn new(value: T) -> Self {
        triomphe::Arc::new(value)
    }

    #[inline(always)]
    fn into_raw(this: Self) -> *const T {
        triomphe::Arc::into_raw(this)
    }

    #[inline(always)]
    unsafe fn from_raw(ptr: *const T) -> Self {
        // SAFETY: forwarded from this trait method's contract.
        unsafe { triomphe::Arc::from_raw(ptr) }
    }
}

/// A lock-free single-value cell of an [`Arc`]-like handle `A`.
///
/// Reads never block. A [`store`](AtomicArc::store) waits only for the O(1) windows in which
/// in-flight readers are cloning the value it replaces -- never for a lock -- so writers cannot be
/// starved by a stream of readers, nor readers by writers.
///
/// The one hazard a naive atomic `Arc` has is a reader loading the pointer and then cloning it just
/// as a writer swaps the value out and drops the last reference, freeing it mid-clone. It is closed
/// by having each reader announce itself in `reading` for the O(1) window in which it clones, and a
/// storing writer wait out that window before dropping the value it replaced. Any reader that could
/// still see the old value is counted, and the swap is ordered before the writer's check, so the
/// writer drops only once every such reader has already bumped the strong count -- a clone can
/// never touch freed memory.
///
/// That window is a handful of instructions -- a pointer read and a strong-count bump -- with no
/// lock, allocation, or syscall in it, so [`store`](AtomicArc::store) waits it out with a bounded
/// spin rather than parking. Nothing here blocks, so there is no async variant.
pub struct AtomicArc<A: RawArc> {
    /// The live handle as a raw pointer from [`RawArc::into_raw`]; always non-null.
    ptr: AtomicPtr<A::Target>,
    /// Readers currently between announcing themselves and finishing their clone of `ptr`.
    reading: AtomicUsize,
    _marker: PhantomData<A>,
}

// SAFETY: the cell owns and hands out `A` (an `Arc`) across threads, so it is `Send`/`Sync` under
// exactly the bounds `A` already carries.
unsafe impl<A: RawArc + Send + Sync> Send for AtomicArc<A> {}
unsafe impl<A: RawArc + Send + Sync> Sync for AtomicArc<A> {}

impl<A: RawArc> AtomicArc<A> {
    /// Creates a cell holding `initial`.
    pub fn new(initial: A) -> Self {
        Self {
            ptr: AtomicPtr::new(A::into_raw(initial) as *mut A::Target),
            reading: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    /// The latest published value, as an owned clone. Never blocks.
    pub fn load(&self) -> A {
        // Announce before reading the pointer: a writer that swaps after this cannot free the value
        // we are about to clone until we leave `reading`.
        self.reading.fetch_add(1, Ordering::SeqCst);
        let ptr = self.ptr.load(Ordering::SeqCst);
        // SAFETY: `ptr` came from `RawArc::into_raw` and, because we are counted in `reading`, a
        // concurrent `store` has not yet reclaimed it. Rebuilding the handle to clone it and then
        // forgetting the rebuild leaves the cell's own count intact and hands us a fresh one.
        let value = unsafe {
            let owned = A::from_raw(ptr);
            let cloned = owned.clone();
            core::mem::forget(owned);
            cloned
        };
        self.reading.fetch_sub(1, Ordering::SeqCst);
        value
    }

    /// Publishes `value`, then reclaims the value it replaced once no reader can still be cloning
    /// it. The wait is a bounded spin over the readers' O(1) clone windows, not a park.
    pub fn store(&self, value: A) {
        let old = self.swap_in(value);
        // Any reader that loaded `old` incremented `reading` before its load, and `swap_in` is
        // ordered before this check, so once `reading` reads zero every such reader has finished
        // bumping `old`'s strong count -- reclaiming our reference cannot free a value a reader is
        // still cloning.
        let mut backoff = Backoff::new();
        while self.reading.load(Ordering::SeqCst) != 0 {
            backoff.spin();
        }
        // SAFETY: no reader can still be cloning `old`; reclaim the cell's reference exactly once.
        drop(unsafe { A::from_raw(old) });
    }

    /// Swaps `value` in and returns the displaced raw pointer (one owned strong count).
    #[inline]
    fn swap_in(&self, value: A) -> *const A::Target {
        self.ptr.swap(A::into_raw(value) as *mut A::Target, Ordering::SeqCst)
    }
}

impl<A: RawArc> Drop for AtomicArc<A> {
    fn drop(&mut self) {
        // SAFETY: `&mut self` means no reader is in flight; reclaim the one stored reference.
        drop(unsafe { A::from_raw(*self.ptr.get_mut()) });
    }
}

impl<A: RawArc> core::fmt::Debug for AtomicArc<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AtomicArc")
            .field("reading", &self.reading.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::*;

    #[test]
    fn load_sees_initial_then_stores() {
        let cell = AtomicArc::new(Arc::new(1u32));
        assert_eq!(*cell.load(), 1);
        cell.store(Arc::new(2));
        assert_eq!(*cell.load(), 2);
    }

    #[test]
    fn a_held_read_does_not_block_a_store() {
        let cell = AtomicArc::new(Arc::new(1u32));
        // A reader's snapshot is an owned clone, so holding it never blocks a writer, and the
        // writer's replacement is visible at once.
        let snapshot = cell.load();
        cell.store(Arc::new(2));
        assert_eq!(*snapshot, 1, "the snapshot keeps the value it was taken at");
        assert_eq!(*cell.load(), 2, "the next load sees the store");
    }

    #[test]
    fn store_reclaims_the_replaced_value() {
        let first = Arc::new(1u32);
        let watch = Arc::downgrade(&first);
        let cell = AtomicArc::new(first);
        assert!(watch.upgrade().is_some(), "the cell holds the value");
        cell.store(Arc::new(2));
        assert!(watch.upgrade().is_none(), "the replaced value was reclaimed");
    }

    #[test]
    fn concurrent_readers_and_writers_stay_sound() {
        let cell = Arc::new(AtomicArc::new(Arc::new(0u64)));
        std::thread::scope(|s| {
            for _ in 0..4 {
                let cell = cell.clone();
                s.spawn(move || {
                    for _ in 0..50_000 {
                        // A read that saw a torn or freed value would fault or trip the checker.
                        assert!(*cell.load() < 1_000_000);
                    }
                });
            }
            for _ in 0..2 {
                let cell = cell.clone();
                s.spawn(move || {
                    for i in 0..50_000 {
                        cell.store(Arc::new(i));
                    }
                });
            }
        });
    }
}
