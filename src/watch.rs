//! A watch channel: a single latest value observed by many receivers.
//!
//! The state lives in a standalone [`Watch`] the caller owns; [`Watch::split`] yields the borrowed
//! [`Sender`] and a first [`Receiver`] (clone or [`Sender::subscribe`] for more). The value lives in
//! a [`Lock`](crate::Lock) and change notifications reuse [`Notify`](crate::Notify), so receivers
//! can wait from blocking or async code.

use core::ops::Deref;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::lock::ReadGuard;
use crate::{Lock32, LockStateU32, Notify64};

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

    /// Splits into the sender and a first receiver.
    ///
    /// Takes `&mut self` so the halves cannot be created more than once.
    pub fn split(&mut self) -> (Sender<'_, T>, Receiver<'_, T>) {
        let chan: &Self = self;
        let seen = chan.version.load(Ordering::Acquire);
        (Sender { chan }, Receiver { chan, seen })
    }
}

/// The sending half of a [`Watch`].
pub struct Sender<'a, T> {
    chan: &'a Watch<T>,
}

/// A receiving half of a [`Watch`]. Clone to observe from multiple places.
pub struct Receiver<'a, T> {
    chan: &'a Watch<T>,
    seen: u64,
}

impl<'a, T> Sender<'a, T> {
    /// Replaces the value and notifies all receivers.
    ///
    /// Returns `Err(value)` if every receiver has been dropped.
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        if self.chan.receivers.load(Ordering::Acquire) == 0 {
            return Err(SendError(value));
        }
        self.send_modify(|slot| *slot = value);
        Ok(())
    }

    /// Modifies the value in place and notifies all receivers, regardless of receiver count.
    pub fn send_modify<F: FnOnce(&mut T)>(&self, f: F) {
        {
            let mut guard = self.chan.value.write();
            f(&mut guard);
            // Bump the version while the write lock is held so readers never observe a value that is
            // newer than the version they read.
            self.chan.version.fetch_add(1, Ordering::Release);
        }
        self.chan.notify.notify(usize::MAX);
    }

    /// Borrows the current value without affecting receivers.
    #[inline]
    pub fn borrow(&self) -> Ref<'_, T> {
        Ref { guard: self.chan.value.read() }
    }

    /// The number of live receivers.
    #[inline]
    pub fn receiver_count(&self) -> usize {
        self.chan.receivers.load(Ordering::Acquire)
    }

    /// Creates a new receiver that observes changes made after this call.
    pub fn subscribe(&self) -> Receiver<'a, T> {
        self.chan.receivers.fetch_add(1, Ordering::Release);
        Receiver { chan: self.chan, seen: self.chan.version.load(Ordering::Acquire) }
    }
}

impl<T> Drop for Sender<'_, T> {
    fn drop(&mut self) {
        self.chan.sender_alive.store(false, Ordering::Release);
        // Wake receivers so they observe closure.
        self.chan.notify.notify(usize::MAX);
    }
}

impl<T> Receiver<'_, T> {
    /// Borrows the most recent value without marking it seen.
    #[inline]
    pub fn borrow(&self) -> Ref<'_, T> {
        Ref { guard: self.chan.value.read() }
    }

    /// Borrows the most recent value and marks it seen, so the next [`changed`](Receiver::changed)
    /// waits for a later change.
    pub fn borrow_and_update(&mut self) -> Ref<'_, T> {
        self.seen = self.chan.version.load(Ordering::Acquire);
        Ref { guard: self.chan.value.read() }
    }

    /// Returns `Ok(true)` if the value has changed since it was last seen, `Ok(false)` if not, or
    /// `Err` if the sender has been dropped and there is no unseen change.
    pub fn has_changed(&self) -> Result<bool, RecvError> {
        if self.chan.version.load(Ordering::Acquire) != self.seen {
            return Ok(true);
        }
        if self.chan.sender_alive.load(Ordering::Acquire) {
            Ok(false)
        } else {
            Err(RecvError)
        }
    }

    /// Checks for a change, updating the seen version. Split fields so a listener borrowing `chan`
    /// can be held across the check without conflicting with the mutable `seen` borrow.
    fn poll_changed(chan: &Watch<T>, seen: &mut u64) -> Option<Result<(), RecvError>> {
        let version = chan.version.load(Ordering::Acquire);
        if version != *seen {
            *seen = version;
            return Some(Ok(()));
        }
        if !chan.sender_alive.load(Ordering::Acquire) {
            // Re-check the version: a send may have raced with the sender drop.
            let version = chan.version.load(Ordering::Acquire);
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
            if let Some(result) = Self::poll_changed(self.chan, &mut self.seen) {
                return result;
            }
            let listener = self.chan.notify.listener();
            if let Some(result) = Self::poll_changed(self.chan, &mut self.seen) {
                return result;
            }
            listener.wait();
        }
    }

    /// Resolves once the value changes or the sender is dropped.
    pub async fn changed_async(&mut self) -> Result<(), RecvError> {
        loop {
            if let Some(result) = Self::poll_changed(self.chan, &mut self.seen) {
                return result;
            }
            let listener = self.chan.notify.listener();
            if let Some(result) = Self::poll_changed(self.chan, &mut self.seen) {
                return result;
            }
            listener.await;
        }
    }
}

impl<'a, T> Clone for Receiver<'a, T> {
    fn clone(&self) -> Self {
        self.chan.receivers.fetch_add(1, Ordering::Release);
        Receiver { chan: self.chan, seen: self.seen }
    }
}

impl<T> Drop for Receiver<'_, T> {
    fn drop(&mut self) {
        self.chan.receivers.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn borrow_sees_initial_then_updates() {
        let mut chan = Watch::new(1u32);
        let (tx, rx) = chan.split();
        assert_eq!(*rx.borrow(), 1);
        tx.send(2).unwrap();
        assert_eq!(*rx.borrow(), 2);
    }

    #[test]
    fn has_changed_tracks_seen() {
        let mut chan = Watch::new(0u32);
        let (tx, mut rx) = chan.split();
        assert_eq!(rx.has_changed(), Ok(false));
        tx.send(1).unwrap();
        assert_eq!(rx.has_changed(), Ok(true));
        let _ = rx.borrow_and_update();
        assert_eq!(rx.has_changed(), Ok(false));
    }

    #[test]
    fn send_fails_with_no_receivers() {
        let mut chan = Watch::new(0u32);
        let (tx, rx) = chan.split();
        drop(rx);
        assert_eq!(tx.send(1), Err(SendError(1)));
    }

    #[test]
    fn changed_errors_when_sender_dropped() {
        let mut chan = Watch::new(0u32);
        let (tx, mut rx) = chan.split();
        drop(tx);
        assert_eq!(rx.changed(), Err(RecvError));
    }

    #[test]
    fn blocking_changed_waits() {
        let mut chan = Watch::new(0u32);
        let (tx, mut rx) = chan.split();
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
        let mut chan = Watch::new(0u32);
        let (tx, mut rx1) = chan.split();
        let mut rx2 = rx1.clone();
        tx.send(5).unwrap();
        rx1.changed().unwrap();
        rx2.changed().unwrap();
        assert_eq!(*rx1.borrow(), 5);
        assert_eq!(*rx2.borrow(), 5);
    }

    #[tokio::test]
    async fn async_changed_waits() {
        let mut chan = Watch::new(0u32);
        let (tx, mut rx) = chan.split();
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
}
