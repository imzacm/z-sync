use alloc::sync::Arc;
#[cfg(not(feature = "triomphe-arc"))]
use alloc::sync::Arc as DefaultArc;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::task::{Context, Poll};

// The shared pointer `ObservableLock` publishes its value as by default: a `triomphe::Arc` with the
// `triomphe-arc` feature, else the standard-library `Arc`, matching the rest of the crate's
// owned handles. Callers who want the other one name it through the `A` parameter, for which
// `ArcObservableLock` and `TriompheArcObservableLock` below are the ready-made spellings.
#[cfg(feature = "triomphe-arc")]
use triomphe::Arc as DefaultArc;
#[cfg(feature = "triomphe-arc")]
use triomphe::Arc as TriompheArc;

use crate::atomic_arc::{AtomicArc, RawArc};
use crate::lock::{ReadFuture, ReadGuard, WriteFuture, WriteGuard};
use crate::park_strategy::DefaultParkStrategy;
use crate::{
    Lock, LockState, LockStateU16, LockStateU32, LockStateU64, Notify, NotifyState, NotifyStateU16,
    NotifyStateU32, NotifyStateU64, ParkStrategy,
};

pub type ObservableLock16<T, A = DefaultArc<T>, NS = NotifyStateU16, P = DefaultParkStrategy> =
    ObservableLock<T, A, LockStateU16, NS, P>;
pub type ObservableLock32<T, A = DefaultArc<T>, NS = NotifyStateU32, P = DefaultParkStrategy> =
    ObservableLock<T, A, LockStateU32, NS, P>;
pub type ObservableLock64<T, A = DefaultArc<T>, NS = NotifyStateU64, P = DefaultParkStrategy> =
    ObservableLock<T, A, LockStateU64, NS, P>;

/// An [`ObservableLock`] publishing its value through a std [`Arc`].
pub type ArcObservableLock<T, LS = LockStateU32, NS = NotifyStateU32, P = DefaultParkStrategy> =
    ObservableLock<T, Arc<T>, LS, NS, P>;
/// An [`ObservableLock`] publishing its value through a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type TriompheArcObservableLock<
    T,
    LS = LockStateU32,
    NS = NotifyStateU32,
    P = DefaultParkStrategy,
> = ObservableLock<T, TriompheArc<T>, LS, NS, P>;

/// Emits a notification when a write guard is dropped.
///
/// `A` is the shared pointer the latest value is published as — any [`RawArc`], which the crate
/// implements for the standard-library [`Arc`] and, under the `triomphe-arc` feature, for
/// [`triomphe::Arc`]. It defaults to whichever of the two that feature selects.
#[derive(Debug)]
pub struct ObservableLock<
    T,
    A: RawArc<Target = T> = DefaultArc<T>,
    LS: LockState = LockStateU32,
    NS: NotifyState = NotifyStateU32,
    P = DefaultParkStrategy,
> {
    lock: Lock<T, LS, P>,
    notify: Notify<NS, P>,
    latest_value: AtomicArc<A>,
}

impl<T: Clone, A: RawArc<Target = T>, LS: LockState, NS: NotifyState>
    ObservableLock<T, A, LS, NS, DefaultParkStrategy>
{
    pub fn new(data: T) -> Self {
        Self::with_park_strategy(data)
    }
}

impl<T: Clone, A: RawArc<Target = T>, LS: LockState, NS: NotifyState, P: ParkStrategy> Default
    for ObservableLock<T, A, LS, NS, P>
where
    T: Default,
{
    fn default() -> Self {
        Self::with_park_strategy(T::default())
    }
}

impl<T: Clone, A: RawArc<Target = T>, LS: LockState, NS: NotifyState, P: ParkStrategy>
    ObservableLock<T, A, LS, NS, P>
{
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
            latest_value: AtomicArc::new(A::new(data)),
        }
    }

    pub fn from_lock(lock: Lock<T, LS, P>) -> Self {
        // We own the lock, so we know nobody is referencing it.
        let data = lock.try_read().unwrap().clone();
        Self {
            lock,
            notify: Notify::with_park_strategy(),
            latest_value: AtomicArc::new(A::new(data)),
        }
    }

    #[inline(always)]
    pub fn observe(&self) -> crate::notify::NotifyListener<'_, NS, P> {
        self.notify.listener()
    }

    /// The most recently published value, as a cheap shared-pointer clone. Never blocks -- readers
    /// do not contend with writers, however busy the lock is.
    #[inline(always)]
    pub fn latest_value(&self) -> A {
        self.latest_value.load()
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
    pub fn try_write(&self) -> Option<ObservableLockWriteGuard<'_, T, A, LS, NS, P>> {
        let guard = self.lock.try_write()?;
        Some(ObservableLockWriteGuard { guard: ManuallyDrop::new(guard), _marker: PhantomData })
    }

    #[inline(always)]
    pub fn write(&self) -> ObservableLockWriteGuard<'_, T, A, LS, NS, P> {
        ObservableLockWriteGuard {
            guard: ManuallyDrop::new(self.lock.write()),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn write_async(&self) -> ObservableLockWriteFuture<'_, T, A, LS, NS, P> {
        ObservableLockWriteFuture { future: self.lock.write_async(), _marker: PhantomData }
    }
}

/// The guard carries `A` so that its drop -- which is what publishes the new value -- can name the
/// `ObservableLock` it belongs to, and so a guard can never be paired with the wrong pointer type.
#[derive(Debug)]
pub struct ObservableLockWriteGuard<
    'a,
    T: Clone,
    A: RawArc<Target = T>,
    LS: LockState,
    NS: NotifyState,
    P: ParkStrategy = DefaultParkStrategy,
> {
    guard: ManuallyDrop<WriteGuard<'a, T, LS, P>>,
    _marker: PhantomData<(A, NS)>,
}

impl<T: Clone, A: RawArc<Target = T>, LS: LockState, NS: NotifyState, P: ParkStrategy> Drop
    for ObservableLockWriteGuard<'_, T, A, LS, NS, P>
{
    fn drop(&mut self) {
        let lock = unsafe {
            let lock = WriteGuard::get_lock(&self.guard);
            ObservableLock::<T, A, LS, NS, P>::from_lock_ref(lock)
        };

        let new_value: T = self.guard.clone();
        unsafe {
            ManuallyDrop::drop(&mut self.guard);
        }

        lock.latest_value.store(A::new(new_value));
        lock.notify.notify(usize::MAX);
    }
}

impl<T: Clone, A: RawArc<Target = T>, LS: LockState, NS: NotifyState, P: ParkStrategy> Deref
    for ObservableLockWriteGuard<'_, T, A, LS, NS, P>
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T: Clone, A: RawArc<Target = T>, LS: LockState, NS: NotifyState, P: ParkStrategy> DerefMut
    for ObservableLockWriteGuard<'_, T, A, LS, NS, P>
{
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

pin_project_lite::pin_project! {
    #[derive(Debug)]
    pub struct ObservableLockWriteFuture<'a, T, A: RawArc<Target = T>, LS: LockState, NS: NotifyState, P: ParkStrategy = DefaultParkStrategy> {
        #[pin]
        future: WriteFuture<'a, T, LS, P>,
        _marker: PhantomData<(A, NS)>,
    }
}

impl<'a, T: Clone, A: RawArc<Target = T>, LS: LockState, NS: NotifyState, P: ParkStrategy> Future
    for ObservableLockWriteFuture<'a, T, A, LS, NS, P>
{
    type Output = ObservableLockWriteGuard<'a, T, A, LS, NS, P>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.future.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(guard) => Poll::Ready(ObservableLockWriteGuard {
                guard: ManuallyDrop::new(guard),
                _marker: PhantomData,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Publishing works through whichever pointer `A` names, so the two aliases are exercised
    /// against the same expectations.
    macro_rules! publishes_through {
        ($name:ident, $alias:ident) => {
            #[test]
            fn $name() {
                let lock = $alias::<u32>::new(1);
                assert_eq!(*lock.latest_value(), 1);

                *lock.write() = 2;
                assert_eq!(*lock.latest_value(), 2, "the guard's drop published the new value");
                assert_eq!(*lock.read(), 2, "and the lock holds it too");
            }
        };
    }

    publishes_through!(publishes_through_the_std_arc, ArcObservableLock);
    #[cfg(feature = "triomphe-arc")]
    publishes_through!(publishes_through_the_triomphe_arc, TriompheArcObservableLock);

    #[test]
    fn observers_see_the_published_value() {
        let lock = ArcObservableLock::<u32>::new(0);
        std::thread::scope(|s| {
            let listener = lock.observe();
            s.spawn(|| {
                listener.wait();
                assert_eq!(*lock.latest_value(), 7, "the value is published before the notify");
            });
            *lock.write() = 7;
        });
    }
}
