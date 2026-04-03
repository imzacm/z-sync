mod state;

use alloc::boxed::Box;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::sync::atomic::{AtomicPtr, Ordering};
use core::task::{Context, Poll, Waker};

use num_traits::{ConstZero, NumCast};

pub use self::state::*;
use crate::NotifyState;
use crate::park_strategy::{DefaultParkStrategy, ParkStrategy};
use crate::waker_queue::{WakerQueueLock, WakerTicket};

pub type Lock16<T, P = DefaultParkStrategy> = Lock<T, LockStateU16, P>;
pub type Lock32<T, P = DefaultParkStrategy> = Lock<T, LockStateU32, P>;
pub type Lock64<T, P = DefaultParkStrategy> = Lock<T, LockStateU64, P>;

const ASYNC_CAPACITY: usize = 4;

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
#[derive(Debug)]
pub struct Lock<T, S: LockState, P = DefaultParkStrategy> {
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
    read_wakers: AtomicPtr<WakerQueueLock<ASYNC_CAPACITY>>,
    write_wakers: AtomicPtr<WakerQueueLock<ASYNC_CAPACITY>>,
}

impl<T, S: LockState> Lock<T, S, DefaultParkStrategy> {
    pub const fn new(data: T) -> Self {
        Self::with_park_strategy(data)
    }
}

impl<T, S: LockState, P> Default for Lock<T, S, P>
where
    T: Default,
    P: ParkStrategy,
{
    fn default() -> Self {
        Self::with_park_strategy(T::default())
    }
}

impl<T, S: LockState, P: ParkStrategy> Lock<T, S, P> {
    pub const fn with_park_strategy(data: T) -> Self {
        Self {
            _marker: core::marker::PhantomData,
            state: S::INITIAL_ATOMIC,
            data: UnsafeCell::new(data),
            read_wakers: AtomicPtr::new(core::ptr::null_mut()),
            write_wakers: AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    #[inline(always)]
    pub fn into_observable<N: NotifyState>(self) -> crate::ObservableLock<T, S, N, P>
    where
        T: Clone,
    {
        crate::ObservableLock::from_lock(self)
    }

    #[cold]
    fn init_waker_queue(
        ptr: &AtomicPtr<WakerQueueLock<ASYNC_CAPACITY>>,
    ) -> &WakerQueueLock<ASYNC_CAPACITY> {
        let queue = Box::into_raw(Box::new(WakerQueueLock::new()));
        match ptr.compare_exchange(
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

    #[inline(always)]
    fn get_read_wakers(&self) -> &WakerQueueLock<ASYNC_CAPACITY> {
        let ptr = self.read_wakers.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe { &*ptr }
        } else {
            Self::init_waker_queue(&self.read_wakers)
        }
    }

    #[inline(always)]
    fn get_write_wakers(&self) -> &WakerQueueLock<ASYNC_CAPACITY> {
        let ptr = self.write_wakers.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe { &*ptr }
        } else {
            Self::init_waker_queue(&self.write_wakers)
        }
    }

    pub fn try_read(&self) -> Option<ReadGuard<'_, T, S, P>> {
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

    fn spin_try_read(&self) -> Option<ReadGuard<'_, T, S, P>> {
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

    pub fn try_write(&self) -> Option<WriteGuard<'_, T, S, P>> {
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
            // Instantly check if any readers or writers exist
            if state.has_readers_or_writers() {
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

    fn spin_try_write(&self) -> Option<WriteGuard<'_, T, S, P>> {
        let mut backoff = 1;
        for _ in 0..WRITE_SPIN_MAX {
            let state = self.load_state(Ordering::Relaxed);

            if !state.has_readers_or_writers()
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
                && let Some(guard) = self.try_write()
            {
                return Some(guard);
            }
            std::thread::yield_now();
        }

        None
    }

    #[inline(always)]
    pub fn read(&self) -> ReadGuard<'_, T, S, P> {
        if let Some(guard) = self.try_read() {
            return guard;
        }
        self.read_slow()
    }

    #[cold]
    #[inline(never)]
    fn read_slow(&self) -> ReadGuard<'_, T, S, P> {
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
    pub fn write(&self) -> WriteGuard<'_, T, S, P> {
        if let Some(guard) = self.try_write() {
            return guard;
        }
        self.write_slow()
    }

    #[cold]
    #[inline(never)]
    fn write_slow(&self) -> WriteGuard<'_, T, S, P> {
        if let Some(guard) = self.spin_try_write() {
            return guard;
        }

        self.add_write_parker(Ordering::Relaxed);

        loop {
            P::park(self.writer_parking_key(), || {
                let s = self.load_state(Ordering::Relaxed);
                s.writers() > S::Writers::ZERO || s.readers() > S::Readers::ZERO
            });

            if let Some(guard) = self.try_write() {
                self.sub_write_parker(Ordering::Relaxed);
                return guard;
            }
        }
    }

    #[inline(always)]
    pub fn read_async(&self) -> ReadFuture<'_, T, S, P> {
        ReadFuture { lock: self, waker_node_ticket: None }
    }

    #[inline(always)]
    pub fn write_async(&self) -> WriteFuture<'_, T, S, P> {
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
            if state.write_wakers() > S::WriteWakers::ZERO {
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
}

impl<T, S: LockState, P> Lock<T, S, P> {
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

unsafe impl<T: Send, S: LockState> Send for Lock<T, S> {}
unsafe impl<T: Send, S: LockState> Sync for Lock<T, S> {}

impl<T, S: LockState, P> Drop for Lock<T, S, P> {
    fn drop(&mut self) {
        let read_wakers = self.read_wakers.load(Ordering::Relaxed);
        if !read_wakers.is_null() {
            unsafe { drop(Box::from_raw(read_wakers)) };
        }

        let write_wakers = self.write_wakers.load(Ordering::Relaxed);
        if !write_wakers.is_null() {
            unsafe { drop(Box::from_raw(write_wakers)) };
        }
    }
}

#[derive(Debug)]
pub struct ReadGuard<'a, T, S: LockState, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, S, P>,
}

impl<'a, T, S: LockState, P: ParkStrategy> ReadGuard<'a, T, S, P> {
    #[inline(always)]
    pub fn map<U, F>(guard: ReadGuard<'a, T, S, P>, f: F) -> MappedReadGuard<'a, T, U, S, P>
    where
        F: FnOnce(&T) -> &U,
        U: ?Sized,
    {
        let value = unsafe { &*guard.lock.data.get() };
        MappedReadGuard { _guard: guard, value: f(value) }
    }
}

impl<T, S: LockState, P: ParkStrategy> Drop for ReadGuard<'_, T, S, P> {
    fn drop(&mut self) {
        self.lock.common_dropped::<true>();
    }
}

impl<T, S: LockState, P: ParkStrategy> Deref for ReadGuard<'_, T, S, P> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

#[derive(Debug)]
pub struct MappedReadGuard<'a, T, U: ?Sized, S: LockState, P: ParkStrategy = DefaultParkStrategy> {
    _guard: ReadGuard<'a, T, S, P>,
    value: &'a U,
}

impl<'a, T, U: ?Sized, S: LockState, P: ParkStrategy> Deref for MappedReadGuard<'a, T, U, S, P> {
    type Target = U;

    fn deref(&self) -> &U {
        self.value
    }
}

#[derive(Debug)]
pub struct WriteGuard<'a, T, S: LockState, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, S, P>,
}

impl<'a, T, S: LockState, P: ParkStrategy> WriteGuard<'a, T, S, P> {
    pub(crate) unsafe fn get_lock(guard: &Self) -> &'a Lock<T, S, P> {
        guard.lock
    }
}

impl<T, S: LockState, P: ParkStrategy> Drop for WriteGuard<'_, T, S, P> {
    fn drop(&mut self) {
        self.lock.common_dropped::<false>();
    }
}

impl<T, S: LockState, P: ParkStrategy> Deref for WriteGuard<'_, T, S, P> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T, S: LockState, P: ParkStrategy> DerefMut for WriteGuard<'_, T, S, P> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

#[derive(Debug)]
pub struct ReadFuture<'a, T, S: LockState, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, S, P>,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, T, S: LockState, P: ParkStrategy> Future for ReadFuture<'a, T, S, P> {
    type Output = ReadGuard<'a, T, S, P>;

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

impl<T, S: LockState, P: ParkStrategy> Drop for ReadFuture<'_, T, S, P> {
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.get_read_wakers().lock().remove(ticket)
        {
            self.lock.sub_read_waker(Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
pub struct WriteFuture<'a, T, S: LockState, P: ParkStrategy = DefaultParkStrategy> {
    lock: &'a Lock<T, S, P>,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, T, S: LockState, P: ParkStrategy> Future for WriteFuture<'a, T, S, P> {
    type Output = WriteGuard<'a, T, S, P>;

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

impl<T, S: LockState, P: ParkStrategy> Drop for WriteFuture<'_, T, S, P> {
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.lock.get_write_wakers().lock().remove(ticket)
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
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());

        {
            let _w = lock.write_async().await;
        }
        assert_eq!(lock.load_state(Ordering::Relaxed), crate::LockStateU64::empty());
    }
}
