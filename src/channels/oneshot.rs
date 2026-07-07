//! A one-shot channel: a single value sent from one [`Sender`] to one [`Receiver`].
//!
//! The channel state lives in a standalone [`OneShot`] that the caller owns and stores however they
//! like (on the stack with scoped threads, in a `static`, behind their own `Arc`, ...). Call
//! [`OneShot::split`] to obtain the borrowed [`Sender`]/[`Receiver`] halves. Wakeups reuse
//! [`Notify`](crate::Notify), so the receiver can wait from blocking or async code.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU8, Ordering};

use crate::Notify32;

const EMPTY: u8 = 0;
/// The value has been written and is waiting to be taken.
const SENT: u8 = 1;
/// The value has been taken by the receiver.
const TAKEN: u8 = 2;
/// The sender was dropped without sending.
const TX_CLOSED: u8 = 3;
/// The receiver was dropped.
const RX_CLOSED: u8 = 4;

/// Error returned by [`Receiver::recv`] when the sender was dropped without sending a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecvError;

/// Error returned by [`Receiver::try_recv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    /// No value is available yet, but the sender is still alive.
    Empty,
    /// The sender was dropped without sending a value.
    Closed,
}

/// A one-shot channel. Own one of these and call [`split`](OneShot::split) to get the halves.
pub struct OneShot<T> {
    state: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
    notify: Notify32,
}

// SAFETY: access to `value` is gated by `state` transitions, so the value crosses the
// sender→receiver boundary exactly once.
unsafe impl<T: Send> Send for OneShot<T> {}
unsafe impl<T: Send> Sync for OneShot<T> {}

impl<T> Default for OneShot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OneShot<T> {
    /// Creates an empty one-shot channel.
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(EMPTY),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            notify: Notify32::new(),
        }
    }

    /// Splits into the borrowed sender/receiver halves.
    ///
    /// Takes `&mut self` so the halves cannot be created more than once.
    pub fn split(&mut self) -> (Sender<'_, T>, Receiver<'_, T>) {
        let chan: &Self = self;
        (Sender { chan }, Receiver { chan })
    }
}

impl<T> Drop for OneShot<T> {
    fn drop(&mut self) {
        // A value that was sent but never taken must be dropped here.
        if *self.state.get_mut() == SENT {
            unsafe { (*self.value.get()).assume_init_drop() };
        }
    }
}

/// The sending half of a [`OneShot`].
#[derive(Debug)]
pub struct Sender<'a, T> {
    chan: &'a OneShot<T>,
}

/// The receiving half of a [`OneShot`].
#[derive(Debug)]
pub struct Receiver<'a, T> {
    chan: &'a OneShot<T>,
}

impl<T> core::fmt::Debug for OneShot<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OneShot").field("state", &self.state).finish_non_exhaustive()
    }
}

impl<T> Sender<'_, T> {
    /// Sends `value`, consuming the sender.
    ///
    /// Returns `Err(value)` if the receiver has already been dropped.
    pub fn send(self, value: T) -> Result<(), T> {
        // Write the value before publishing `SENT`; the receiver reads it under Acquire.
        unsafe { (*self.chan.value.get()).write(value) };

        match self
            .chan
            .state
            .compare_exchange(EMPTY, SENT, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => {
                self.chan.notify.notify(1);
                Ok(())
            }
            Err(_) => {
                // The receiver is gone (RX_CLOSED). Reclaim the value we just wrote.
                let value = unsafe { (*self.chan.value.get()).assume_init_read() };
                Err(value)
            }
        }
    }

    /// Returns `true` if the receiver has been dropped, so [`send`](Sender::send) would fail.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.chan.state.load(Ordering::Acquire) == RX_CLOSED
    }
}

impl<T> Drop for Sender<'_, T> {
    fn drop(&mut self) {
        // If we never sent, transition EMPTY→TX_CLOSED and wake the receiver. If a value was sent
        // (SENT) or the receiver already left (RX_CLOSED), the CAS fails and there is nothing to do.
        if self
            .chan
            .state
            .compare_exchange(EMPTY, TX_CLOSED, Ordering::Release, Ordering::Relaxed)
            .is_ok()
        {
            self.chan.notify.notify(1);
        }
    }
}

impl<T> Receiver<'_, T> {
    /// Reads the value if one is available or the sender has closed, without waiting.
    fn poll_value(&self) -> Option<Result<T, RecvError>> {
        match self.chan.state.load(Ordering::Acquire) {
            SENT => {
                let value = unsafe { (*self.chan.value.get()).assume_init_read() };
                self.chan.state.store(TAKEN, Ordering::Release);
                Some(Ok(value))
            }
            TX_CLOSED => Some(Err(RecvError)),
            _ => None,
        }
    }

    /// Attempts to receive the value without waiting.
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        match self.poll_value() {
            Some(Ok(value)) => Ok(value),
            Some(Err(RecvError)) => Err(TryRecvError::Closed),
            None => Err(TryRecvError::Empty),
        }
    }

    /// Blocks the current thread until the value is received or the sender is dropped.
    pub fn recv(self) -> Result<T, RecvError> {
        loop {
            if let Some(result) = self.poll_value() {
                return result;
            }
            let listener = self.chan.notify.listener();
            if let Some(result) = self.poll_value() {
                return result;
            }
            listener.wait();
        }
    }

    /// Resolves once the value is received or the sender is dropped.
    pub async fn recv_async(self) -> Result<T, RecvError> {
        loop {
            if let Some(result) = self.poll_value() {
                return result;
            }
            let listener = self.chan.notify.listener();
            if let Some(result) = self.poll_value() {
                return result;
            }
            listener.await;
        }
    }
}

impl<T> Drop for Receiver<'_, T> {
    fn drop(&mut self) {
        // Signal the sender that we are gone (only meaningful while still EMPTY). A SENT-but-untaken
        // value is left in place for `OneShot::drop` to clean up.
        let _ = self.chan.state.compare_exchange(
            EMPTY,
            RX_CLOSED,
            Ordering::Release,
            Ordering::Relaxed,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use super::*;

    #[test]
    fn send_then_recv() {
        let mut chan = OneShot::new();
        let (tx, rx) = chan.split();
        tx.send(42u32).unwrap();
        assert_eq!(rx.recv().unwrap(), 42);
    }

    #[test]
    fn try_recv_empty_then_value() {
        let mut chan = OneShot::<u32>::new();
        let (tx, rx) = chan.split();
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        tx.send(7).unwrap();
        assert_eq!(rx.try_recv(), Ok(7));
    }

    #[test]
    fn sender_dropped_without_send() {
        let mut chan = OneShot::<u32>::new();
        let (tx, rx) = chan.split();
        drop(tx);
        assert_eq!(rx.recv(), Err(RecvError));
    }

    #[test]
    fn receiver_dropped_before_send() {
        let mut chan = OneShot::new();
        let (tx, rx) = chan.split();
        drop(rx);
        assert_eq!(tx.send(9u32), Err(9));
    }

    #[test]
    fn blocking_recv_waits_for_send() {
        let mut chan = OneShot::<u32>::new();
        let (tx, rx) = chan.split();
        std::thread::scope(|s| {
            let h = s.spawn(move || rx.recv());
            std::thread::sleep(Duration::from_millis(20));
            tx.send(123).unwrap();
            assert_eq!(h.join().unwrap(), Ok(123));
        });
    }

    #[test]
    fn value_dropped_when_never_received() {
        let dropped = Arc::new(AtomicUsize::new(0));
        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        {
            let mut chan = OneShot::new();
            let (tx, rx) = chan.split();
            assert!(tx.send(Guard(Arc::clone(&dropped))).is_ok());
            drop(rx);
        }
        assert_eq!(dropped.load(Ordering::Relaxed), 1, "sent-but-unreceived value must be dropped");
    }

    #[tokio::test]
    async fn async_recv_waits_for_send() {
        let mut chan = OneShot::<u32>::new();
        let (tx, rx) = chan.split();
        let (got, ()) = tokio::join!(rx.recv_async(), async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tx.send(55).unwrap();
        });
        assert_eq!(got, Ok(55));
    }

    #[tokio::test]
    async fn async_recv_sender_dropped() {
        let mut chan = OneShot::<u32>::new();
        let (tx, rx) = chan.split();
        let (got, ()) = tokio::join!(rx.recv_async(), async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(tx);
        });
        assert_eq!(got, Err(RecvError));
    }
}
