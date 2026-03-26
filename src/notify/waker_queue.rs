#[cfg(not(feature = "thin-vec"))]
use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::task::Waker;

#[cfg(feature = "thin-vec")]
use thin_vec::ThinVec as Vec;

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

impl<const ARRAY_CAPACITY: usize> WakerQueue<ARRAY_CAPACITY> {
    pub fn new() -> Self {
        assert!(
            ARRAY_CAPACITY < u16::MAX as usize,
            "WakerQueue capacity must be less than `u16::MAX`"
        );

        Self {
            nodes_array: core::array::from_fn(|index| {
                let next = if index == ARRAY_CAPACITY - 1 {
                    WakerNode::NULL_NODE
                } else {
                    (index + 1) as u16
                };
                WakerNode::new(WakerNode::NULL_NODE, next)
            }),
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
