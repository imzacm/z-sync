//! A watch channel: a single latest value observed by many receivers.
//!
//! The state lives in a standalone [`Watch`] the caller owns; [`Watch::split`] yields the borrowed
//! [`Sender`] and a first [`Receiver`] (clone or [`Sender::subscribe`] for more). The value lives in
//! a [`Lock`](crate::Lock) and change notifications reuse [`Notify`](crate::Notify), so receivers
//! can wait from blocking or async code.

use alloc::rc::Rc;
use alloc::sync::Arc;
use core::marker::PhantomData;
use core::ops::Deref;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "triomphe-arc")]
use triomphe::Arc as TriompheArc;

use crate::lock::ReadGuard;
use crate::{Holder, Lock32, LockStateU32, Notify64};

/// A borrowed [`Sender`] (`&Watch`).
pub type RefSender<'a, T> = Sender<T, &'a Watch<T>>;
/// A borrowed [`Receiver`] (`&Watch`).
pub type RefReceiver<'a, T> = Receiver<T, &'a Watch<T>>;
/// An owned [`Sender`] backed by a std [`Arc`].
pub type ArcSender<T> = Sender<T, Arc<Watch<T>>>;
/// An owned [`Receiver`] backed by a std [`Arc`].
pub type ArcReceiver<T> = Receiver<T, Arc<Watch<T>>>;
/// An owned [`Sender`] backed by an [`Rc`].
pub type RcSender<T> = Sender<T, Rc<Watch<T>>>;
/// An owned [`Receiver`] backed by an [`Rc`].
pub type RcReceiver<T> = Receiver<T, Rc<Watch<T>>>;
/// An owned [`Sender`] backed by a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type TriompheArcSender<T> = Sender<T, TriompheArc<Watch<T>>>;
/// An owned [`Receiver`] backed by a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type TriompheArcReceiver<T> = Receiver<T, TriompheArc<Watch<T>>>;

/// Error returned by [`Sender::send`] when every receiver has been dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendError<T>(pub T);

/// Error returned by [`Receiver::changed`] when the sender has been dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecvError;

/// A watch channel. Own one of these and call [`split`](Watch::split) to get the halves.
pub struct Watch<T> {
    value: Lock32<T>,
    /// Incremented on every change; receivers compare it against their last-seen version.
    version: AtomicU64,
    notify: Notify64,
    receivers: AtomicUsize,
    sender_alive: AtomicBool,
}

/// A shared read guard over the watched value.
pub struct Ref<'a, T> {
    guard: ReadGuard<'a, T, LockStateU32>,
}

impl<T> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> Watch<T> {
    /// Creates a watch channel seeded with `initial`.
    pub const fn new(initial: T) -> Self {
        Self {
            value: Lock32::new(initial),
            version: AtomicU64::new(0),
            notify: Notify64::new(),
            receivers: AtomicUsize::new(1),
            sender_alive: AtomicBool::new(true),
        }
    }

    /// Splits into borrowed sender and a first receiver.
    ///
    /// Takes `&mut self` so the halves cannot be created more than once.
    pub fn split(&mut self) -> (RefSender<'_, T>, RefReceiver<'_, T>) {
        let channel: &Self = self;
        Self::make_halves(channel)
    }

    /// Splits into owned halves backed by a std [`Arc`] (movable across threads/tasks).
    pub fn arc_split(self: &Arc<Self>) -> (ArcSender<T>, ArcReceiver<T>) {
        Self::make_halves(Arc::clone(self))
    }

    /// Splits into owned halves backed by an [`Rc`] (single-threaded).
    pub fn rc_split(self: &Rc<Self>) -> (RcSender<T>, RcReceiver<T>) {
        Self::make_halves(Rc::clone(self))
    }

    /// Splits into owned halves backed by a `triomphe::Arc`.
    ///
    /// A free-standing associated function because `triomphe::Arc` cannot be a `self` receiver on
    /// stable.
    #[cfg(feature = "triomphe-arc")]
    pub fn triomphe_arc_split(
        this: &TriompheArc<Self>,
    ) -> (TriompheArcSender<T>, TriompheArcReceiver<T>) {
        Self::make_halves(TriompheArc::clone(this))
    }

    fn make_halves<H: Holder<Self>>(channel: H) -> (Sender<T, H>, Receiver<T, H>) {
        let seen = channel.version.load(Ordering::Acquire);
        (
            Sender { channel: channel.clone(), _marker: PhantomData },
            Receiver { channel, seen, _marker: PhantomData },
        )
    }
}

/// The sending half of a [`Watch`], generic over the [`Holder`] `H`. See the [`RefSender`] /
/// [`ArcSender`] / [`RcSender`] aliases.
pub struct Sender<T, H: Holder<Watch<T>> = Arc<Watch<T>>> {
    channel: H,
    _marker: PhantomData<fn() -> T>,
}

/// A receiving half of a [`Watch`], generic over the [`Holder`] `H`. Clone to observe from multiple
/// places. See the [`RefReceiver`] / [`ArcReceiver`] / [`RcReceiver`] aliases.
pub struct Receiver<T, H: Holder<Watch<T>> = Arc<Watch<T>>> {
    channel: H,
    seen: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<T, H: Holder<Watch<T>>> Sender<T, H> {
    /// Replaces the value and notifies all receivers.
    ///
    /// Returns `Err(value)` if every receiver has been dropped.
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        if self.channel.receivers.load(Ordering::Acquire) == 0 {
            return Err(SendError(value));
        }
        self.send_modify(|slot| *slot = value);
        Ok(())
    }

    /// Modifies the value in place and notifies all receivers, regardless of receiver count.
    pub fn send_modify<F: FnOnce(&mut T)>(&self, f: F) {
        {
            let mut guard = self.channel.value.write();
            f(&mut guard);
            // Bump the version while the write lock is held so readers never observe a value that is
            // newer than the version they read.
            self.channel.version.fetch_add(1, Ordering::Release);
        }
        self.channel.notify.notify(usize::MAX);
    }

    /// Borrows the current value without affecting receivers.
    #[inline]
    pub fn borrow(&self) -> Ref<'_, T> {
        Ref { guard: self.channel.value.read() }
    }

    /// The number of live receivers.
    #[inline]
    pub fn receiver_count(&self) -> usize {
        self.channel.receivers.load(Ordering::Acquire)
    }

    /// Creates a new receiver that observes changes made after this call.
    pub fn subscribe(&self) -> Receiver<T, H> {
        self.channel.receivers.fetch_add(1, Ordering::Release);
        Receiver {
            channel: self.channel.clone(),
            seen: self.channel.version.load(Ordering::Acquire),
            _marker: PhantomData,
        }
    }
}

impl<T, H: Holder<Watch<T>>> Drop for Sender<T, H> {
    fn drop(&mut self) {
        self.channel.sender_alive.store(false, Ordering::Release);
        // Wake receivers so they observe closure.
        self.channel.notify.notify(usize::MAX);
    }
}

impl<T, H: Holder<Watch<T>>> Receiver<T, H> {
    /// Borrows the most recent value without marking it seen.
    #[inline]
    pub fn borrow(&self) -> Ref<'_, T> {
        Ref { guard: self.channel.value.read() }
    }

    /// Borrows the most recent value and marks it seen, so the next [`changed`](Receiver::changed)
    /// waits for a later change.
    pub fn borrow_and_update(&mut self) -> Ref<'_, T> {
        self.seen = self.channel.version.load(Ordering::Acquire);
        Ref { guard: self.channel.value.read() }
    }

    /// Returns `Ok(true)` if the value has changed since it was last seen, `Ok(false)` if not, or
    /// `Err` if the sender has been dropped and there is no unseen change.
    pub fn has_changed(&self) -> Result<bool, RecvError> {
        if self.channel.version.load(Ordering::Acquire) != self.seen {
            return Ok(true);
        }
        if self.channel.sender_alive.load(Ordering::Acquire) {
            Ok(false)
        } else {
            Err(RecvError)
        }
    }

    /// Checks for a change, updating the seen version. Split fields so a listener borrowing `channel`
    /// can be held across the check without conflicting with the mutable `seen` borrow.
    fn poll_changed(channel: &Watch<T>, seen: &mut u64) -> Option<Result<(), RecvError>> {
        let version = channel.version.load(Ordering::Acquire);
        if version != *seen {
            *seen = version;
            return Some(Ok(()));
        }
        if !channel.sender_alive.load(Ordering::Acquire) {
            // Re-check the version: a send may have raced with the sender drop.
            let version = channel.version.load(Ordering::Acquire);
            if version != *seen {
                *seen = version;
                return Some(Ok(()));
            }
            return Some(Err(RecvError));
        }
        None
    }

    /// Blocks until the value changes or the sender is dropped.
    pub fn changed(&mut self) -> Result<(), RecvError> {
        loop {
            if let Some(result) = Self::poll_changed(&*self.channel, &mut self.seen) {
                return result;
            }
            let listener = self.channel.notify.listener();
            if let Some(result) = Self::poll_changed(&*self.channel, &mut self.seen) {
                return result;
            }
            listener.wait();
        }
    }

    /// Resolves once the value changes or the sender is dropped.
    pub async fn changed_async(&mut self) -> Result<(), RecvError> {
        loop {
            if let Some(result) = Self::poll_changed(&*self.channel, &mut self.seen) {
                return result;
            }
            let listener = self.channel.notify.listener();
            if let Some(result) = Self::poll_changed(&*self.channel, &mut self.seen) {
                return result;
            }
            listener.await;
        }
    }
}

impl<T, H: Holder<Watch<T>>> Clone for Receiver<T, H> {
    fn clone(&self) -> Self {
        self.channel.receivers.fetch_add(1, Ordering::Release);
        Receiver { channel: self.channel.clone(), seen: self.seen, _marker: PhantomData }
    }
}

impl<T, H: Holder<Watch<T>>> Drop for Receiver<T, H> {
    fn drop(&mut self) {
        self.channel.receivers.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn borrow_sees_initial_then_updates() {
        let mut channel = Watch::new(1u32);
        let (tx, rx) = channel.split();
        assert_eq!(*rx.borrow(), 1);
        tx.send(2).unwrap();
        assert_eq!(*rx.borrow(), 2);
    }

    #[test]
    fn has_changed_tracks_seen() {
        let mut channel = Watch::new(0u32);
        let (tx, mut rx) = channel.split();
        assert_eq!(rx.has_changed(), Ok(false));
        tx.send(1).unwrap();
        assert_eq!(rx.has_changed(), Ok(true));
        let _ = rx.borrow_and_update();
        assert_eq!(rx.has_changed(), Ok(false));
    }

    #[test]
    fn send_fails_with_no_receivers() {
        let mut channel = Watch::new(0u32);
        let (tx, rx) = channel.split();
        drop(rx);
        assert_eq!(tx.send(1), Err(SendError(1)));
    }

    #[test]
    fn changed_errors_when_sender_dropped() {
        let mut channel = Watch::new(0u32);
        let (tx, mut rx) = channel.split();
        drop(tx);
        assert_eq!(rx.changed(), Err(RecvError));
    }

    #[test]
    fn blocking_changed_waits() {
        let mut channel = Watch::new(0u32);
        let (tx, mut rx) = channel.split();
        std::thread::scope(|s| {
            let h = s.spawn(move || {
                rx.changed().unwrap();
                *rx.borrow()
            });
            std::thread::sleep(Duration::from_millis(20));
            tx.send(99).unwrap();
            assert_eq!(h.join().unwrap(), 99);
        });
    }

    #[test]
    fn multiple_receivers_each_observe() {
        let mut channel = Watch::new(0u32);
        let (tx, mut rx1) = channel.split();
        let mut rx2 = rx1.clone();
        tx.send(5).unwrap();
        rx1.changed().unwrap();
        rx2.changed().unwrap();
        assert_eq!(*rx1.borrow(), 5);
        assert_eq!(*rx2.borrow(), 5);
    }

    #[tokio::test]
    async fn async_changed_waits() {
        let mut channel = Watch::new(0u32);
        let (tx, mut rx) = channel.split();
        let (got, ()) = tokio::join!(
            async {
                rx.changed_async().await.unwrap();
                *rx.borrow()
            },
            async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                tx.send(7).unwrap();
            }
        );
        assert_eq!(got, 7);
    }

    #[test]
    fn arc_split_moves_owned_halves_across_threads() {
        let channel = std::sync::Arc::new(Watch::new(0u32));
        let (tx, mut rx) = channel.arc_split();
        let waiter = std::thread::spawn(move || {
            rx.changed().unwrap();
            *rx.borrow()
        });
        std::thread::sleep(Duration::from_millis(20));
        tx.send(9).unwrap();
        assert_eq!(waiter.join().unwrap(), 9);
    }
}
