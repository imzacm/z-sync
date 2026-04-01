#[cfg(not(feature = "thin-vec"))]
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::num::NonZeroU32;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::Waker;

/// Allow creating in const functions.
#[cfg_attr(feature = "thin-vec", derive(Debug))]
#[cfg(feature = "thin-vec")]
struct Vec<T> {
    inner: core::cell::LazyCell<thin_vec::ThinVec<T>>,
}

#[cfg(feature = "thin-vec")]
impl<T> Vec<T> {
    const fn new() -> Self {
        Self { inner: core::cell::LazyCell::new(thin_vec::ThinVec::new) }
    }
}

#[cfg(feature = "thin-vec")]
impl<T> core::ops::Deref for Vec<T> {
    type Target = thin_vec::ThinVec<T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(feature = "thin-vec")]
impl<T> core::ops::DerefMut for Vec<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// An allocation-free, generational doubly-linked list node.
#[derive(Debug)]
pub struct WakerNode {
    /// Prevents the ABA problem if a slot is rapidly reused.
    generation: u16,
    waker: Option<Waker>,
    prev: u16,
    next: u16,
}

impl WakerNode {
    const NULL_NODE: u16 = u16::MAX;

    const fn new(prev: u16, next: u16) -> Self {
        Self { generation: 0, waker: None, prev, next }
    }

    pub const fn generation(&self) -> u16 {
        self.generation
    }

    pub fn waker(&self) -> Option<&Waker> {
        self.waker.as_ref()
    }

    pub fn waker_mut(&mut self) -> &mut Option<Waker> {
        &mut self.waker
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WakerTicket(NonZeroU32);

impl WakerTicket {
    #[inline(always)]
    pub const fn new(index: u16, generation: u16) -> Self {
        // Shift index to the top 32 bits, generation to the bottom 32.
        let value = ((index as u32) << 16) | (generation as u32);
        // Add 1 so the bit pattern is never zero, allowing Option to use 0 as `None`
        let value = unsafe { NonZeroU32::new_unchecked(value + 1) };
        Self(value)
    }

    #[inline(always)]
    pub const fn index(self) -> u16 {
        ((self.0.get() - 1) >> 16) as u16
    }

    #[inline(always)]
    pub const fn generation(self) -> u16 {
        (self.0.get() - 1) as u16
    }
}

/// A pre-allocated queue for async wakers supporting O(1) push, pop, and removal.
/// Spills over into a dynamic vector if the array fills up.
#[derive(Debug)]
pub struct WakerQueue<const ARRAY_CAPACITY: usize> {
    nodes_array: [WakerNode; ARRAY_CAPACITY],
    nodes_vec: Vec<WakerNode>,
    head: u16,
    tail: u16,
    free_head: u16,
}

impl<const ARRAY_CAPACITY: usize> Default for WakerQueue<ARRAY_CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const ARRAY_CAPACITY: usize> WakerQueue<ARRAY_CAPACITY> {
    pub const fn new() -> Self {
        assert!(
            ARRAY_CAPACITY < u16::MAX as usize,
            "WakerQueue capacity must be less than `u16::MAX`"
        );

        let mut nodes_array: [MaybeUninit<WakerNode>; ARRAY_CAPACITY] =
            [const { MaybeUninit::uninit() }; ARRAY_CAPACITY];
        let mut index = 0;
        while index < ARRAY_CAPACITY {
            let next = if index == ARRAY_CAPACITY - 1 {
                WakerNode::NULL_NODE
            } else {
                (index + 1) as u16
            };
            nodes_array[index].write(WakerNode::new(WakerNode::NULL_NODE, next));

            index += 1;
        }

        let nodes_array =
            unsafe { nodes_array.as_ptr().cast::<[WakerNode; ARRAY_CAPACITY]>().read() };

        Self {
            nodes_array,
            nodes_vec: Vec::new(),
            head: WakerNode::NULL_NODE,
            tail: WakerNode::NULL_NODE,
            free_head: 0,
        }
    }

    /// Helper to seamlessly index into either the inline array or the fallback vector.
    #[inline(always)]
    pub fn node_mut(&mut self, index: u16) -> &mut WakerNode {
        unsafe { &mut *self.node_mut_ptr(index) }
    }

    #[inline(always)]
    pub fn node_mut_ptr(&mut self, index: u16) -> *mut WakerNode {
        let index = index as usize;
        if index < ARRAY_CAPACITY {
            unsafe { self.nodes_array.get_unchecked_mut(index) }
        } else {
            debug_assert!((index - ARRAY_CAPACITY) < self.nodes_vec.len(), "index out of bounds");
            unsafe { self.nodes_vec.as_mut_ptr().add(index - ARRAY_CAPACITY) }
        }
    }

    /// Pushes a waker to the back of the queue. Returns an (index, generation) ticket.
    pub fn push(&mut self, waker: Waker) -> WakerTicket {
        let (index, generation) = {
            let tail = self.tail;
            if self.free_head != WakerNode::NULL_NODE {
                let index = self.free_head;
                // Use ptr so we aren't mutably borrowing the whole of self.
                let old_node = unsafe { &mut *self.node_mut_ptr(index) };
                self.free_head = old_node.next;

                old_node.waker = Some(waker);
                old_node.prev = tail;
                old_node.next = WakerNode::NULL_NODE;

                (index, old_node.generation)
            } else {
                let index = (ARRAY_CAPACITY + self.nodes_vec.len()) as u16;
                let mut node = WakerNode::new(tail, WakerNode::NULL_NODE);
                node.waker = Some(waker);
                let generation = node.generation;
                self.nodes_vec.push(node);
                (index, generation)
            }
        };

        // Link it into the active queue
        if self.tail != WakerNode::NULL_NODE {
            self.node_mut(self.tail).next = index;
        } else {
            self.head = index;
        }
        self.tail = index;

        WakerTicket::new(index, generation)
    }

    /// Safely unlinks a node (used when futures are dropped).
    ///
    /// Returns true if the node was successfully removed, false if it was already popped.
    pub fn remove(&mut self, ticket: WakerTicket) -> bool {
        let index = ticket.index();

        // Use ptr so we aren't mutably borrowing the whole of self.
        let node = unsafe { &mut *self.node_mut_ptr(index) };

        // If generation mismatches, this slot was already popped and reused.
        if node.generation != ticket.generation() {
            return false;
        }

        // Invalidate the node
        node.generation = node.generation.wrapping_add(1);
        node.waker = None;

        let prev = node.prev;
        let next = node.next;

        // Push to free list immediately while we have the reference
        // Note: This seamlessly chains free array slots AND free vector slots together
        node.next = self.free_head;
        self.free_head = index;

        // Unlink from neighbors
        if node.prev != WakerNode::NULL_NODE {
            self.node_mut(prev).next = next;
        } else {
            self.head = next;
        }

        if next != WakerNode::NULL_NODE {
            self.node_mut(next).prev = prev;
        } else {
            self.tail = prev;
        }

        true
    }

    /// Pops the front waker and removes it from the queue.
    pub fn pop_and_take(&mut self) -> Option<Waker> {
        if self.head == WakerNode::NULL_NODE {
            return None;
        }

        let index = self.head;
        let (waker, next) = {
            let free_head = self.free_head;
            let node = self.node_mut(index);
            node.generation = node.generation.wrapping_add(1);
            let waker = node.waker.take().unwrap();
            let next = node.next;

            // Push directly to free list
            node.next = free_head;
            (waker, next)
        };

        // Re-link queue
        self.head = next;
        if next != WakerNode::NULL_NODE {
            self.node_mut(next).prev = WakerNode::NULL_NODE;
        } else {
            self.tail = WakerNode::NULL_NODE;
        }
        self.free_head = index;

        Some(waker)
    }
}

/// An exponential backoff spinlock tailored for microscopic critical sections.
#[derive(Debug)]
pub struct WakerQueueLock<const CAP: usize> {
    locked: AtomicBool,
    queue: UnsafeCell<WakerQueue<CAP>>,
}

// SAFETY: The queue is only ever accessed while `locked` is true.
unsafe impl<const CAP: usize> Sync for WakerQueueLock<CAP> {}
unsafe impl<const CAP: usize> Send for WakerQueueLock<CAP> {}

impl<const CAP: usize> Default for WakerQueueLock<CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize> WakerQueueLock<CAP> {
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
            queue: UnsafeCell::new(WakerQueue::new()),
        }
    }

    #[inline(always)]
    pub fn lock(&self) -> WakerQueueGuard<'_, CAP> {
        if self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return WakerQueueGuard { lock: self };
        }

        let mut backoff = 1;
        loop {
            // This stays cleanly inside the CPU's local L1 cache until the lock holder releases it.
            while self.locked.load(Ordering::Relaxed) {
                for _ in 0..backoff {
                    core::hint::spin_loop();
                }
                if backoff < 64 {
                    backoff <<= 1; // Bitwise shift is microscopically faster than *= 2
                } else if cfg!(not(any(target_arch = "x86", target_arch = "x86_64"))) {
                    #[cfg(feature = "std")]
                    std::thread::yield_now();
                }
            }

            // 3. The lock appears free, attempt to grab it
            if self
                .locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return WakerQueueGuard { lock: self };
            }
        }
    }
}

pub struct WakerQueueGuard<'a, const CAP: usize> {
    lock: &'a WakerQueueLock<CAP>,
}

impl<'a, const CAP: usize> core::ops::Deref for WakerQueueGuard<'a, CAP> {
    type Target = WakerQueue<CAP>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.queue.get() }
    }
}

impl<'a, const CAP: usize> core::ops::DerefMut for WakerQueueGuard<'a, CAP> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.queue.get() }
    }
}

impl<'a, const CAP: usize> Drop for WakerQueueGuard<'a, CAP> {
    #[inline(always)]
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}
