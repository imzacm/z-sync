use alloc::boxed::Box;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::waker_queue::WakerQueueLock;

/// Strategy controlling where a primitive keeps its async waker queue.
///
/// This is a compile-time choice, analogous to [`ParkStrategy`](crate::ParkStrategy): it lets a
/// caller trade the size of the primitive against how async waiters are stored, without affecting
/// the blocking path at all.
///
/// - [`InlineWakers`] keeps the queue inline (larger primitive, never allocates on the async path).
/// - [`BoxedWakers`] keeps the queue behind a lazily-allocated pointer (small primitive, one
///   allocation on first async use — and none at all for blocking-only usage).
pub trait WakerStorage<const CAP: usize> {
    /// The initial, empty storage. Being an associated `const` (rather than a method) is what lets
    /// primitives keep a `const fn` constructor.
    const INIT: Self;

    /// Returns the waker queue, allocating it on first use if the strategy is lazy.
    fn queue(&self) -> &WakerQueueLock<CAP>;
}

/// Stores the waker queue inline within the primitive.
///
/// The primitive is larger by `size_of::<WakerQueueLock<CAP>>()`, but async waiters never trigger a
/// heap allocation and every queue access avoids a pointer indirection. This is the default.
#[derive(Debug)]
pub struct InlineWakers<const CAP: usize> {
    queue: WakerQueueLock<CAP>,
}

impl<const CAP: usize> WakerStorage<CAP> for InlineWakers<CAP> {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self { queue: WakerQueueLock::new() };

    #[inline(always)]
    fn queue(&self) -> &WakerQueueLock<CAP> {
        &self.queue
    }
}

/// Stores the waker queue behind a lazily-allocated pointer.
///
/// Keeps the primitive small (a single pointer). The queue is heap-allocated on the first async use
/// and is never allocated at all for blocking-only usage.
#[derive(Debug)]
pub struct BoxedWakers<const CAP: usize> {
    queue: AtomicPtr<WakerQueueLock<CAP>>,
}

impl<const CAP: usize> WakerStorage<CAP> for BoxedWakers<CAP> {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self { queue: AtomicPtr::new(core::ptr::null_mut()) };

    #[inline(always)]
    fn queue(&self) -> &WakerQueueLock<CAP> {
        let ptr = self.queue.load(Ordering::Acquire);
        if !ptr.is_null() { unsafe { &*ptr } } else { self.init() }
    }
}

impl<const CAP: usize> BoxedWakers<CAP> {
    #[cold]
    fn init(&self) -> &WakerQueueLock<CAP> {
        let queue = Box::into_raw(Box::new(WakerQueueLock::new()));
        match self.queue.compare_exchange(
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
}

impl<const CAP: usize> Drop for BoxedWakers<CAP> {
    fn drop(&mut self) {
        let ptr = self.queue.load(Ordering::Relaxed);
        if !ptr.is_null() {
            unsafe { drop(Box::from_raw(ptr)) };
        }
    }
}
