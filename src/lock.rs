use core::sync::atomic::{AtomicU64, Ordering};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use crossbeam_utils::CachePadded;

use crate::park_strategy::{DefaultParkStrategy, ParkStrategy};
use crate::waker_queue::{WakerQueueLock, WakerTicket};

const ASYNC_CAPACITY: usize = 4;

/// An efficient multi-purpose blocking and async lock supporting Mutex and RwLock style usage.
#[derive(Debug)]
pub struct Lock<T, P = DefaultParkStrategy> {
    _marker: core::marker::PhantomData<P>,
    /// Bit layout:
    /// - 0..16:   read async wakers count (u16)
    /// - 16..24:  read parked threads count (u8)
    /// - 24..32:  write async wakers count (u8)
    /// - 32..40:  write parked threads count (u8)
    /// - 40..48:  writer count (u8)
    /// - 48..64:  readers count (u16)
    state: CachePadded<AtomicU64>,
    read_wakers: WakerQueueLock<ASYNC_CAPACITY>,
    write_wakers: WakerQueueLock<ASYNC_CAPACITY>,
    data: CachePadded<UnsafeCell<T>>,
}

impl<T> Lock<T, DefaultParkStrategy> {
    pub fn new(data: T) -> Self {
        Self::with_park_strategy(data)
    }
}

impl<T, P> Default for Lock<T, P>
where
    T: Default,
    P: ParkStrategy,
{
    fn default() -> Self {
        Self::with_park_strategy(T::default())
    }
}

impl<T, P: ParkStrategy> Lock<T, P> {
    pub fn with_park_strategy(data: T) -> Self {
        Self {
            _marker: core::marker::PhantomData,
            state: CachePadded::new(AtomicU64::new(0)),
            read_wakers: WakerQueueLock::new(),
            write_wakers: WakerQueueLock::new(),
            data: CachePadded::new(UnsafeCell::new(data)),
        }
    }

    pub fn try_read(&self) -> Option<ReadGuard<'_, T, P>> {
        let old_state = self.add_reader(Ordering::Acquire);
        // This will drop on None, so we don't need to worry about it.
        let guard = ReadGuard { lock: self };

        if old_state.has_any_write_state() {
            return None;
        }

        Some(guard)
    }

    fn spin_try_read(&self) -> Option<ReadGuard<'_, T, P>> {
        let mut backoff = 1;
        for _ in 0..64 {
            let state = self.load_state(Ordering::Relaxed);

            if !state.has_any_write_state()
                && let Some(guard) = self.try_read()
            {
                return Some(guard);
            }
            for _ in 0..backoff {
                core::hint::spin_loop();
            }
            if backoff < 32 {
                backoff <<= 1;
            }
        }
        None
    }

    pub fn try_write(&self) -> Option<WriteGuard<'_, T, P>> {
        let mut state = match self.state.compare_exchange_weak(
            0,
            1 << Self::WRITER_SHIFT,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Some(WriteGuard { lock: self }),
            Err(v) => LockState(v),
        };

        loop {
            // Instantly check if any readers or writers exist
            if state.has_readers_or_writers() {
                return None;
            }

            assert!(state.writers() < u8::MAX, "Writer overflow");

            let new_state = state.add_writer();
            match self.state.compare_exchange_weak(
                state.0,
                new_state.0,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(WriteGuard { lock: self }),
                Err(v) => state = LockState(v),
            }
        }
    }

    fn spin_try_write(&self) -> Option<WriteGuard<'_, T, P>> {
        let mut backoff = 1;
        for _ in 0..64 {
            let state = self.load_state(Ordering::Relaxed);

            if !state.has_readers_or_writers()
                && let Some(guard) = self.try_write()
            {
                return Some(guard);
            }

            for _ in 0..backoff {
                core::hint::spin_loop();
            }
            if backoff < 32 {
                backoff <<= 1;
            }
        }
        None
    }

    pub fn read(&self) -> ReadGuard<'_, T, P> {
        if let Some(guard) = self.try_read() {
            return guard;
        }

        if let Some(guard) = self.spin_try_read() {
            return guard;
        }

        self.add_read_parker(Ordering::Relaxed);

        loop {
            P::park(self.reader_parking_key(), || {
                let s = self.load_state(Ordering::Relaxed);
                s.writers() > 0 || s.write_parked() > 0 || s.write_wakers() > 0
            });

            if let Some(guard) = self.try_read() {
                self.sub_read_parker(Ordering::Relaxed);
                return guard;
            }
        }
    }

    pub fn write(&self) -> WriteGuard<'_, T, P> {
        if let Some(guard) = self.try_write() {
            return guard;
        }

        if let Some(guard) = self.spin_try_write() {
            return guard;
        }

        self.add_write_parker(Ordering::Relaxed);

        loop {
            P::park(self.writer_parking_key(), || {
                let s = self.load_state(Ordering::Relaxed);
                s.writers() > 0 || s.readers() > 0
            });

            if let Some(guard) = self.try_write() {
                self.sub_write_parker(Ordering::Relaxed);
                return guard;
            }
        }
    }

    #[inline(always)]
    pub fn read_async(&self) -> ReadFuture<'_, T, P> {
        ReadFuture { lock: self, waker_node_ticket: None }
    }

    #[inline(always)]
    pub fn write_async(&self) -> WriteFuture<'_, T, P> {
        WriteFuture { lock: self, waker_node_ticket: None }
    }

    fn common_dropped<const IS_READER: bool>(&self) {
        let state = if IS_READER {
            self.sub_reader(Ordering::Release).sub_reader()
        } else {
            self.sub_writer(Ordering::Release).sub_writer()
        };

        if state.has_readers_or_writers() {
            return;
        }

        if state.has_write_waiters() {
            if state.write_wakers() > 0 {
                self.wake_one_in_queue::<false>();
            } else if state.write_parked() > 0 {
                P::unpark_one(self.writer_parking_key());
            }
        } else if state.has_read_waiters() {
            if state.read_wakers() > 0 {
                self.wake_all_in_queue::<true>();
            }
            if state.read_parked() > 0 {
                P::unpark_all(self.reader_parking_key());
            }
        }
    }

    fn wake_one_in_queue<const IS_READER: bool>(&self) {
        let queue = if IS_READER { &self.read_wakers } else { &self.write_wakers };

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

        let queue = if IS_READER { &self.read_wakers } else { &self.write_wakers };

        let mut batch_sub = BatchStateSub::new();

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
                batch_sub = batch_sub.sub_read_waker(popped as u8);
            } else {
                batch_sub = batch_sub.sub_write_waker(popped as u8);
            }

            for waker in &mut wakers[..popped] {
                // SAFETY: We explicitly initialized exactly `popped` elements
                // inside the mutex lock above.
                unsafe {
                    waker.assume_init_read().wake();
                }
            }
        }

        if batch_sub.0 != 0 {
            self.state.fetch_sub(batch_sub.0, Ordering::Release);
        }
    }
}

impl<T, P> Lock<T, P> {
    const READ_WAKER_SHIFT: u32 = 0;
    const READ_PARKER_SHIFT: u32 = 16;
    const WRITE_WAKER_SHIFT: u32 = 24;
    const WRITE_PARKER_SHIFT: u32 = 32;
    const WRITER_SHIFT: u32 = 40;
    const READER_SHIFT: u32 = 48;
    // Bits 63..40 (24 bits): Covers both Readers and Writers
    const READERS_AND_WRITERS_MASK: u64 = 0xFFFF_FF00_0000_0000;
    // Bits 47..24 (24 bits): Covers Writers, Write Parked, and Write Wakers
    const ANY_WRITE_STATE_MASK: u64 = 0x0000_FFFF_FF00_0000;
    // Bits 39..24 (16 bits): Covers Write Wakers and Write Parked
    const WRITE_WAITERS_MASK: u64 = 0x0000_00FF_FF00_0000;

    // Bits 23..0 (24 bits): Covers Read Wakers and Read Parked
    const READ_WAITERS_MASK: u64 = 0x0000_0000_00FF_FFFF;

    #[inline(always)]
    fn add_read_waker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_add(1 << Self::READ_WAKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn add_read_parker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_add(1 << Self::READ_PARKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn add_write_waker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_add(1 << Self::WRITE_WAKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn add_write_parker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_add(1 << Self::WRITE_PARKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn add_writer(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_add(1 << Self::WRITER_SHIFT, ordering))
    }

    #[inline(always)]
    fn add_reader(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_add(1 << Self::READER_SHIFT, ordering))
    }

    #[inline(always)]
    fn sub_read_waker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_sub(1 << Self::READ_WAKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn sub_read_parker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_sub(1 << Self::READ_PARKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn sub_write_waker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_sub(1 << Self::WRITE_WAKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn sub_write_parker(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_sub(1 << Self::WRITE_PARKER_SHIFT, ordering))
    }

    #[inline(always)]
    fn sub_writer(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_sub(1 << Self::WRITER_SHIFT, ordering))
    }

    #[inline(always)]
    fn sub_reader(&self, ordering: Ordering) -> LockState {
        LockState(self.state.fetch_sub(1 << Self::READER_SHIFT, ordering))
    }

    #[inline(always)]
    fn load_state(&self, ordering: Ordering) -> LockState {
        LockState(self.state.load(ordering))
    }

    #[inline(always)]
    fn reader_parking_key(&self) -> usize {
        core::ptr::from_ref(&self.read_wakers) as usize
    }

    #[inline(always)]
    fn writer_parking_key(&self) -> usize {
        core::ptr::from_ref(&self.write_wakers) as usize
    }
}

unsafe impl<T: Send> Send for Lock<T> {}
unsafe impl<T: Send> Sync for Lock<T> {}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct LockState(u64);

impl LockState {
    #[inline(always)]
    const fn readers(self) -> u16 {
        (self.0 >> 48) as u16
    }

    #[inline(always)]
    const fn writers(self) -> u8 {
        (self.0 >> 40) as u8
    }

    #[inline(always)]
    const fn write_parked(self) -> u8 {
        (self.0 >> 32) as u8
    }

    #[inline(always)]
    const fn write_wakers(self) -> u8 {
        (self.0 >> 24) as u8
    }

    #[inline(always)]
    const fn read_parked(self) -> u8 {
        (self.0 >> 16) as u8
    }

    #[inline(always)]
    const fn read_wakers(self) -> u16 {
        self.0 as u16
    }

    #[inline(always)]
    pub fn add_read_waker(self) -> Self {
        Self(self.0 + (1 << Lock::<()>::READ_WAKER_SHIFT))
    }

    #[inline(always)]
    pub fn sub_read_waker(self) -> Self {
        Self(self.0 - (1 << Lock::<()>::READ_WAKER_SHIFT))
    }

    #[inline(always)]
    pub fn add_read_parker(self) -> Self {
        Self(self.0 + (1 << Lock::<()>::READ_PARKER_SHIFT))
    }

    #[inline(always)]
    pub fn sub_read_parker(self) -> Self {
        Self(self.0 - (1 << Lock::<()>::READ_PARKER_SHIFT))
    }

    #[inline(always)]
    pub fn add_write_waker(self) -> Self {
        Self(self.0 + (1 << Lock::<()>::WRITE_WAKER_SHIFT))
    }

    #[inline(always)]
    pub fn sub_write_waker(self) -> Self {
        Self(self.0 - (1 << Lock::<()>::WRITE_WAKER_SHIFT))
    }

    #[inline(always)]
    pub fn add_write_parker(self) -> Self {
        Self(self.0 + (1 << Lock::<()>::WRITE_PARKER_SHIFT))
    }

    #[inline(always)]
    pub fn sub_write_parker(self) -> Self {
        Self(self.0 - (1 << Lock::<()>::WRITE_PARKER_SHIFT))
    }

    #[inline(always)]
    pub fn add_writer(self) -> Self {
        Self(self.0 + (1 << Lock::<()>::WRITER_SHIFT))
    }

    #[inline(always)]
    pub fn sub_writer(self) -> Self {
        Self(self.0 - (1 << Lock::<()>::WRITER_SHIFT))
    }

    #[inline(always)]
    pub fn add_reader(self) -> Self {
        Self(self.0 + (1 << Lock::<()>::READER_SHIFT))
    }

    #[inline(always)]
    pub fn sub_reader(self) -> Self {
        Self(self.0 - (1 << Lock::<()>::READER_SHIFT))
    }

    #[inline(always)]
    pub const fn has_readers_or_writers(self) -> bool {
        (self.0 & Lock::<()>::READERS_AND_WRITERS_MASK) != 0
    }

    #[inline(always)]
    pub const fn has_any_write_state(self) -> bool {
        (self.0 & Lock::<()>::ANY_WRITE_STATE_MASK) != 0
    }

    #[inline(always)]
    pub const fn has_write_waiters(self) -> bool {
        (self.0 & Lock::<()>::WRITE_WAITERS_MASK) != 0
    }

    #[inline(always)]
    pub const fn has_read_waiters(self) -> bool {
        (self.0 & Lock::<()>::READ_WAITERS_MASK) != 0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct BatchStateSub(u64);

impl BatchStateSub {
    const fn new() -> Self {
        Self(0)
    }

    const fn sub_readers(self, n: u16) -> Self {
        Self(self.0 + ((n as u64) << Lock::<()>::READER_SHIFT))
    }

    const fn sub_writers(self, n: u16) -> Self {
        Self(self.0 + ((n as u64) << Lock::<()>::WRITER_SHIFT))
    }

    const fn sub_read_waker(self, n: u8) -> Self {
        Self(self.0 + ((n as u64) << Lock::<()>::READ_WAKER_SHIFT))
    }

    const fn sub_read_parker(self, n: u8) -> Self {
        Self(self.0 + ((n as u64) << Lock::<()>::READ_PARKER_SHIFT))
    }

    const fn sub_write_waker(self, n: u8) -> Self {
        Self(self.0 + ((n as u64) << Lock::<()>::WRITE_WAKER_SHIFT))
    }

    const fn sub_write_parker(self, n: u8) -> Self {
        Self(self.0 + ((n as u64) << Lock::<()>::WRITE_PARKER_SHIFT))
    }
}

#[derive(Debug)]
pub struct ReadGuard<'a, T, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, P>,
}

impl<T, P: ParkStrategy> Drop for ReadGuard<'_, T, P> {
    fn drop(&mut self) {
        self.lock.common_dropped::<true>();
    }
}

impl<T, P: ParkStrategy> Deref for ReadGuard<'_, T, P> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

#[derive(Debug)]
pub struct WriteGuard<'a, T, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, P>,
}

impl<T, P: ParkStrategy> Drop for WriteGuard<'_, T, P> {
    fn drop(&mut self) {
        self.lock.common_dropped::<false>();
    }
}

impl<T, P: ParkStrategy> Deref for WriteGuard<'_, T, P> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T, P: ParkStrategy> DerefMut for WriteGuard<'_, T, P> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

#[derive(Debug)]
pub struct ReadFuture<'a, T, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, P>,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, T, P: ParkStrategy> Future for ReadFuture<'a, T, P> {
    type Output = ReadGuard<'a, T, P>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();

        if let Some(guard) = this.lock.try_read() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.read_wakers.lock().remove(ticket)
            {
                this.lock.sub_read_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        {
            let mut queue = this.lock.read_wakers.lock();

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
                && this.lock.read_wakers.lock().remove(ticket)
            {
                this.lock.sub_read_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        Poll::Pending
    }
}

impl<T, P: ParkStrategy> Drop for ReadFuture<'_, T, P> {
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.read_wakers.lock().remove(ticket)
        {
            self.lock.sub_read_waker(Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
pub struct WriteFuture<'a, T, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, P>,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, T, P: ParkStrategy> Future for WriteFuture<'a, T, P> {
    type Output = WriteGuard<'a, T, P>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();

        if let Some(guard) = this.lock.try_write() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && this.lock.write_wakers.lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        {
            let mut queue = this.lock.write_wakers.lock();

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
                && this.lock.write_wakers.lock().remove(ticket)
            {
                this.lock.sub_write_waker(Ordering::Relaxed);
            }
            return Poll::Ready(guard);
        }

        Poll::Pending
    }
}

impl<T, P: ParkStrategy> Drop for WriteFuture<'_, T, P> {
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.write_wakers.lock().remove(ticket)
        {
            self.lock.sub_write_waker(Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;
    use std::vec::Vec;

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
        assert_eq!(lock.load_state(Ordering::Relaxed), LockState(0));

        {
            let _w = lock.write();
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), LockState(0));
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
    async fn state_counters_return_to_zero_after_async_guards_dropped() {
        let lock = Lock::new(());

        {
            let _r1 = lock.read_async().await;
            let _r2 = lock.read_async().await;
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), LockState(0));

        {
            let _w = lock.write_async().await;
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), LockState(0));
    }
}
