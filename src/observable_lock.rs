use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::task::{Context, Poll};

use crate::lock::{ReadFuture, ReadGuard, WriteFuture, WriteGuard};
use crate::park_strategy::DefaultParkStrategy;
use crate::{
    Lock, LockState, LockStateU16, LockStateU32, LockStateU64, Notify, NotifyState, NotifyStateU16,
    NotifyStateU32, NotifyStateU64, ParkStrategy,
};

pub type ObservableLock16<T, NS = NotifyStateU16, P = DefaultParkStrategy> =
    ObservableLock<T, LockStateU16, NS, P>;
pub type ObservableLock32<T, NS = NotifyStateU32, P = DefaultParkStrategy> =
    ObservableLock<T, LockStateU32, NS, P>;
pub type ObservableLock64<T, NS = NotifyStateU64, P = DefaultParkStrategy> =
    ObservableLock<T, LockStateU64, NS, P>;

/// Emits a notification when a write guard is dropped.
#[derive(Debug)]
pub struct ObservableLock<
    T,
    LS: LockState = LockStateU32,
    NS: NotifyState = NotifyStateU32,
    P = DefaultParkStrategy,
> {
    lock: Lock<T, LS, P>,
    notify: Notify<NS, P>,
    // TODO: Make this state configurable.
    latest_value: Lock<T, LockStateU16, P>,
}

impl<T: Clone, LS: LockState, NS: NotifyState> ObservableLock<T, LS, NS, DefaultParkStrategy> {
    pub fn new(data: T) -> Self {
        Self::with_park_strategy(data)
    }
}

impl<T: Clone, LS: LockState, NS: NotifyState, P: ParkStrategy> Default
    for ObservableLock<T, LS, NS, P>
where
    T: Default,
{
    fn default() -> Self {
        Self::with_park_strategy(T::default())
    }
}

impl<T: Clone, LS: LockState, NS: NotifyState, P: ParkStrategy> ObservableLock<T, LS, NS, P> {
    /// # Safety
    ///
    /// The caller must guarantee that `lock` points to the `lock` field
    /// of a valid, live `ObservableLock<T, P>` and that the returned
    /// reference does not outlive that `ObservableLock`.
    unsafe fn from_lock_ref(lock: &Lock<T, LS, P>) -> &Self {
        let lock_ptr = lock as *const Lock<T, LS, P>;
        let offset = core::mem::offset_of!(Self, lock);
        let base_ptr = unsafe { lock_ptr.cast::<u8>().sub(offset).cast::<Self>() };
        unsafe { &*base_ptr }
    }

    pub fn with_park_strategy(data: T) -> Self {
        Self {
            lock: Lock::with_park_strategy(data.clone()),
            notify: Notify::with_park_strategy(),
            latest_value: Lock::with_park_strategy(data),
        }
    }

    pub fn from_lock(lock: Lock<T, LS, P>) -> Self {
        // We own the lock, so we know nobody is referencing it.
        let data = lock.try_read().unwrap().clone();
        Self {
            lock,
            notify: Notify::with_park_strategy(),
            latest_value: Lock::with_park_strategy(data),
        }
    }

    #[inline(always)]
    pub fn observe(&self) -> crate::notify::NotifyListener<'_, NS, P> {
        self.notify.listener()
    }

    #[inline(always)]
    pub fn latest_value(&self) -> ReadGuard<'_, T, LockStateU16, P> {
        self.latest_value.read()
    }

    #[inline(always)]
    pub fn latest_value_async(&self) -> ReadFuture<'_, T, LockStateU16, P> {
        self.latest_value.read_async()
    }

    #[inline(always)]
    pub fn into_lock(self) -> Lock<T, LS, P> {
        self.lock
    }

    #[inline(always)]
    pub fn try_read(&self) -> Option<ReadGuard<'_, T, LS, P>> {
        self.lock.try_read()
    }

    #[inline(always)]
    pub fn read(&self) -> ReadGuard<'_, T, LS, P> {
        self.lock.read()
    }

    #[inline(always)]
    pub fn read_async(&self) -> ReadFuture<'_, T, LS, P> {
        self.lock.read_async()
    }

    #[inline(always)]
    pub fn try_write(&self) -> Option<ObservableLockWriteGuard<'_, T, LS, NS, P>> {
        let guard = self.lock.try_write()?;
        Some(ObservableLockWriteGuard {
            guard: ManuallyDrop::new(guard),
            _marker: core::marker::PhantomData,
        })
    }

    #[inline(always)]
    pub fn write(&self) -> ObservableLockWriteGuard<'_, T, LS, NS, P> {
        ObservableLockWriteGuard {
            guard: ManuallyDrop::new(self.lock.write()),
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn write_async(&self) -> ObservableLockWriteFuture<'_, T, LS, NS, P> {
        ObservableLockWriteFuture {
            future: self.lock.write_async(),
            _marker: core::marker::PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct ObservableLockWriteGuard<
    'a,
    T: Clone,
    LS: LockState,
    NS: NotifyState,
    P: ParkStrategy = DefaultParkStrategy,
> {
    guard: ManuallyDrop<WriteGuard<'a, T, LS, P>>,
    _marker: core::marker::PhantomData<NS>,
}

impl<T: Clone, LS: LockState, NS: NotifyState, P: ParkStrategy> Drop
    for ObservableLockWriteGuard<'_, T, LS, NS, P>
{
    fn drop(&mut self) {
        let lock = unsafe {
            let lock = WriteGuard::get_lock(&self.guard);
            ObservableLock::<T, LS, NS, P>::from_lock_ref(lock)
        };

        let new_value: T = self.guard.clone();
        unsafe {
            ManuallyDrop::drop(&mut self.guard);
        }

        {
            let mut lock = lock.latest_value.write();
            *lock = new_value;
        }

        lock.notify.notify(usize::MAX);
    }
}

impl<T: Clone, LS: LockState, NS: NotifyState, P: ParkStrategy> Deref
    for ObservableLockWriteGuard<'_, T, LS, NS, P>
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T: Clone, LS: LockState, NS: NotifyState, P: ParkStrategy> DerefMut
    for ObservableLockWriteGuard<'_, T, LS, NS, P>
{
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

pin_project_lite::pin_project! {
    #[derive(Debug)]
    pub struct ObservableLockWriteFuture<'a, T, LS: LockState, NS: NotifyState, P: ParkStrategy = DefaultParkStrategy> {
        #[pin]
        future: WriteFuture<'a, T, LS, P>,
        _marker: core::marker::PhantomData<NS>,
    }
}

impl<'a, T: Clone, LS: LockState, NS: NotifyState, P: ParkStrategy> Future
    for ObservableLockWriteFuture<'a, T, LS, NS, P>
{
    type Output = ObservableLockWriteGuard<'a, T, LS, NS, P>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.future.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(guard) => Poll::Ready(ObservableLockWriteGuard {
                guard: ManuallyDrop::new(guard),
                _marker: core::marker::PhantomData,
            }),
        }
    }
}
