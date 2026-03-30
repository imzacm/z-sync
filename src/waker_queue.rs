#[cfg(not(feature = "thin-vec"))]
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::num::NonZeroU32;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::Waker;
use std::mem::MaybeUninit;

/// Allow creating in const functions.
#[cfg_attr(feature = "thin-vec", derive(Debug))]
#[cfg(feature = "thin-vec")]
struct Vec<T> {
    inner: core::cell::OnceCell<thin_vec::ThinVec<T>>,
}

#[cfg(feature = "thin-vec")]
impl<T> Vec<T> {
    const fn new() -> Self {
        Self { inner: core::cell::OnceCell::new() }
    }

    #[inline(always)]
    fn get_ref(&self) -> &thin_vec::ThinVec<T> {
        self.inner.get_or_init(thin_vec::ThinVec::new)
    }

    #[inline(always)]
    fn get_mut(&mut self) -> &mut thin_vec::ThinVec<T> {
        self.get_ref();
        unsafe { self.inner.get_mut().unwrap_unchecked() }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.get_ref().len()
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.get_mut().push(value);
    }
}

#[cfg(feature = "thin-vec")]
impl<T> core::ops::Index<usize> for Vec<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.get_ref()[index]
    }
}

#[cfg(feature = "thin-vec")]
impl<T> core::ops::IndexMut<usize> for Vec<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.get_mut()[index]
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
        let index = index as usize;
        if index < ARRAY_CAPACITY {
            unsafe { self.nodes_array.get_unchecked_mut(index) }
        } else {
            &mut self.nodes_vec[index - ARRAY_CAPACITY]
        }
    }

    /// Pushes a waker to the back of the queue. Returns an (index, generation) ticket.
    pub fn push(&mut self, waker: Waker) -> WakerTicket {
        // 1. Claim a free slot or allocate a new one in the vector
        let index = if self.free_head != WakerNode::NULL_NODE {
            let index = self.free_head;
            self.free_head = self.node_mut(index).next;
            index
        } else {
            let index = (ARRAY_CAPACITY + self.nodes_vec.len()) as u16;
            self.nodes_vec.push(WakerNode::new(WakerNode::NULL_NODE, WakerNode::NULL_NODE));
            index
        };

        // 2. Initialize the claimed node (tightly scoped for borrow checker)
        let generation = {
            let tail = self.tail;
            let node = self.node_mut(index);
            node.waker = Some(waker);
            node.prev = tail;
            node.next = WakerNode::NULL_NODE;
            node.generation
        };

        // 3. Link it into the active queue
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

        // Scope the borrow to extract current state so we can mutate neighbors later
        let (node_gen, prev, next) = {
            let node = self.node_mut(index);
            (node.generation, node.prev, node.next)
        };

        // If generation mismatches, this slot was already popped and reused.
        if node_gen != ticket.generation() {
            return false;
        }

        // Invalidate the node
        {
            let node = self.node_mut(index);
            node.generation = node.generation.wrapping_add(1);
            node.waker = None;
        }

        // Unlink from neighbors
        if prev != WakerNode::NULL_NODE {
            self.node_mut(prev).next = next;
        } else {
            self.head = next;
        }

        if next != WakerNode::NULL_NODE {
            self.node_mut(next).prev = prev;
        } else {
            self.tail = prev;
        }

        // Push the slot back onto the free list
        // Note: This seamlessly chains free array slots AND free vector slots together!
        {
            let head = self.free_head;
            let node = self.node_mut(index);
            node.next = head;
            self.free_head = index;
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
            let waker = node.waker.take();
            let waker = unsafe { waker.unwrap_unchecked() };
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
