//! A sequence lock ([`SeqLock`]): lock-free reads of small `Copy`, read-mostly data.
//!
//! A `SeqLock` is the counterpart to [`Lock`](crate::Lock) for data that is read far more often
//! than it is written and is cheap to copy (a config snapshot, a pair of counters, a transform
//! matrix). Readers never take a lock and never write to shared memory — they optimistically copy
//! the value and then check a sequence counter to confirm no writer interfered, retrying if one
//! did. This means **readers never block writers** (and vice versa), at the cost of readers
//! potentially retrying under heavy write contention. Writers are serialised against each other.
//!
//! The sequence counter is even while the data is quiescent and odd while a write is in progress. A
//! reader that observes an odd count, or a count that changed across its read, discards the copy
//! and retries.
//!
//! # Soundness
//!
//! Like every sequence lock, a reader may copy the data *while* a writer is modifying it (the stale
//! copy is then thrown away). This is a benign data race in practice — the read is done with
//! [`read_volatile`](core::ptr::read_volatile) so the compiler cannot tear or elide it — but it is
//! technically undefined behaviour under the C/C++/Rust memory model. It is sound on every real
//! architecture, and is the same trade-off made by the established `seqlock` crate. Prefer
//! [`Lock`](crate::Lock) when this caveat is unacceptable.

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicUsize, Ordering, fence};

/// Backoff cap for the (rare) case of writers contending with each other.
const WRITE_SPIN_CAP: usize = 64;

/// A sequence lock over a `Copy` value, providing lock-free reads and serialised writes.
///
/// See the [module documentation](self) for the design and the soundness caveat.
pub struct SeqLock<T> {
    /// Even = quiescent, odd = a write is in progress. Monotonically increasing (wrapping).
    seq: AtomicUsize,
    data: UnsafeCell<T>,
}

// SAFETY: reads only ever hand out *copies* of `T` (never a shared reference into the cell) and the
// odd-sequence write lock serialises writers, so sharing a `&SeqLock` across threads is sound
// whenever `T` may itself cross threads. `T: Sync` is not required because readers never observe a
// borrow of the stored value.
unsafe impl<T: Send> Send for SeqLock<T> {}
unsafe impl<T: Send> Sync for SeqLock<T> {}

impl<T> SeqLock<T> {
    /// Creates a new `SeqLock` holding `value`.
    pub const fn new(value: T) -> Self {
        Self { seq: AtomicUsize::new(0), data: UnsafeCell::new(value) }
    }

    /// Returns a mutable reference to the value. No synchronisation is needed because the borrow is
    /// exclusive.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Consumes the lock, returning the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Acquires the write lock, blocking (spinning) until any other writer releases, and returns a
    /// guard granting mutable access. Readers observe the new value once the guard is dropped.
    #[inline]
    pub fn write(&self) -> SeqLockWriteGuard<'_, T> {
        // Fast path: flip even -> odd in a single unconditional RMW. If the low bit was already set
        // the value is unchanged (a no-op) and another writer holds the lock, so we spin instead.
        // This is cheaper than a load + compare_exchange on the common uncontended path.
        let old = self.seq.fetch_or(1, Ordering::Acquire);
        if (old & 1) == 0 {
            // Keep the data writes from being reordered ahead of the odd-sequence store.
            fence(Ordering::Release);
            return SeqLockWriteGuard { lock: self, seq: old.wrapping_add(1) };
        }
        self.write_contended()
    }

    #[cold]
    #[inline(never)]
    fn write_contended(&self) -> SeqLockWriteGuard<'_, T> {
        let mut backoff = 1;
        loop {
            let seq = self.seq.load(Ordering::Relaxed);
            if (seq & 1) == 0
                && self
                    .seq
                    .compare_exchange_weak(
                        seq,
                        seq.wrapping_add(1),
                        Ordering::Acquire,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                fence(Ordering::Release);
                return SeqLockWriteGuard { lock: self, seq: seq.wrapping_add(1) };
            }
            for _ in 0..backoff {
                core::hint::spin_loop();
            }
            if backoff < WRITE_SPIN_CAP {
                backoff <<= 1;
            }
        }
    }

    /// Attempts to acquire the write lock without spinning. Returns `None` if another writer holds
    /// it.
    #[inline]
    pub fn try_write(&self) -> Option<SeqLockWriteGuard<'_, T>> {
        let seq = self.seq.load(Ordering::Relaxed);
        if (seq & 1) == 0
            && self
                .seq
                .compare_exchange(seq, seq.wrapping_add(1), Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            fence(Ordering::Release);
            Some(SeqLockWriteGuard { lock: self, seq: seq.wrapping_add(1) })
        } else {
            None
        }
    }
}

impl<T: Copy> SeqLock<T> {
    /// Reads a consistent copy of the value, retrying (spinning) if a writer interferes. Never
    /// blocks a writer.
    #[inline]
    pub fn read(&self) -> T {
        loop {
            if let Some(value) = self.try_read() {
                return value;
            }
            core::hint::spin_loop();
        }
    }

    /// Attempts a single lock-free read. Returns `None` if a write was in progress or landed during
    /// the read (i.e. the caller should retry).
    #[inline]
    pub fn try_read(&self) -> Option<T> {
        let seq1 = self.seq.load(Ordering::Acquire);
        if (seq1 & 1) != 0 {
            // A writer is mid-update.
            return None;
        }
        // SAFETY: the volatile read stops the compiler from tearing, eliding, or reordering it past
        // the second sequence load. If a writer raced this copy, `seq2 != seq1` (or the odd bit was
        // set) and we report the miss rather than returning torn data. See the module soundness
        // note.
        let value = unsafe { core::ptr::read_volatile(self.data.get()) };
        // Order the data read before the confirming sequence load, pairing with the writer's fence.
        fence(Ordering::Acquire);
        if self.seq.load(Ordering::Relaxed) == seq1 { Some(value) } else { None }
    }

    /// Overwrites the value (a convenience for `*lock.write() = value`).
    #[inline]
    pub fn set(&self, value: T) {
        *self.write() = value;
    }
}

impl<T: Default> Default for SeqLock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Copy + core::fmt::Debug> core::fmt::Debug for SeqLock<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SeqLock").field("data", &self.read()).finish()
    }
}

/// RAII guard granting exclusive write access to a [`SeqLock`]. The updated value becomes visible
/// to readers when the guard is dropped.
pub struct SeqLockWriteGuard<'a, T> {
    lock: &'a SeqLock<T>,
    /// The odd sequence value this guard holds; releasing bumps it to the next even value.
    seq: usize,
}

impl<T> Deref for SeqLockWriteGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the guard holds the odd-sequence write lock, so it has exclusive access.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for SeqLockWriteGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the guard holds the odd-sequence write lock, so it has exclusive access.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for SeqLockWriteGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        // Publish the writes (Release) and return the sequence to even, allowing readers through.
        self.lock.seq.store(self.seq.wrapping_add(1), Ordering::Release);
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for SeqLockWriteGuard<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::vec::Vec;

    use super::*;

    #[test]
    fn basic_read_write() {
        let lock = SeqLock::new(5u32);
        assert_eq!(lock.read(), 5);
        *lock.write() = 10;
        assert_eq!(lock.read(), 10);
        lock.set(20);
        assert_eq!(lock.read(), 20);
    }

    #[test]
    fn try_write_and_try_read_while_writing() {
        let lock = SeqLock::new(1u32);
        let w = lock.write();
        // A held write guard means an odd sequence: neither another writer nor a reader can
        // proceed.
        assert!(lock.try_write().is_none());
        assert!(lock.try_read().is_none());
        drop(w);
        assert!(lock.try_write().is_some());
        assert_eq!(lock.try_read(), Some(1));
    }

    #[test]
    fn get_mut_and_into_inner() {
        let mut lock = SeqLock::new(3u32);
        *lock.get_mut() += 1;
        assert_eq!(lock.read(), 4);
        assert_eq!(lock.into_inner(), 4);
    }

    #[test]
    fn writers_are_serialised() {
        const THREADS: usize = 8;
        const INCREMENTS: usize = 1000;
        let lock = Arc::new(SeqLock::new(0u64));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                std::thread::spawn(move || {
                    for _ in 0..INCREMENTS {
                        *lock.write() += 1;
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(lock.read(), (THREADS * INCREMENTS) as u64);
    }

    #[test]
    fn concurrent_reads_never_tear() {
        // Two fields are always written to the same value; a torn read would observe them unequal.
        #[derive(Clone, Copy)]
        struct Pair(u64, u64);

        const WRITES: u64 = 100_000;
        let lock = Arc::new(SeqLock::new(Pair(0, 0)));
        let stop = Arc::new(AtomicBool::new(false));

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let lock = Arc::clone(&lock);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let p = lock.read();
                        assert_eq!(p.0, p.1, "torn read observed");
                    }
                })
            })
            .collect();

        let writer = {
            let lock = Arc::clone(&lock);
            std::thread::spawn(move || {
                for i in 1..=WRITES {
                    let mut w = lock.write();
                    w.0 = i;
                    w.1 = i;
                }
            })
        };

        writer.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }

        let final_value = lock.read();
        assert_eq!(final_value.0, WRITES);
        assert_eq!(final_value.1, WRITES);
    }
}
