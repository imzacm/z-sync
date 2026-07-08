//! A one-shot channel: a single value sent from one [`Sender`] to one [`Receiver`].
//!
//! The channel state lives in a standalone [`OneShot`] that the caller owns. The
//! `Sender`/`Receiver` halves are generic over the [`Holder`](crate::Holder) that keeps the channel
//! alive: [`split`](OneShot::split) yields borrowed halves (`&OneShot`), while
//! [`arc_split`](OneShot::arc_split) / [`rc_split`](OneShot::rc_split) /
//! [`triomphe_arc_split`](OneShot::triomphe_arc_split) yield owned halves that can be moved into a
//! spawned thread or task. An async receiver waits on a single [`AtomicWaker`](crate::AtomicWaker)
//! and a blocking receiver parks its thread, so the single consumer can wait from either world
//! without carrying a full waker queue.

use alloc::rc::Rc;
use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::future::poll_fn;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU8, Ordering};
use core::task::Poll;

#[cfg(feature = "triomphe-arc")]
use triomphe::Arc as TriompheArc;

use crate::Holder;
use crate::atomic_waker::AtomicWaker;
use crate::park_strategy::{DefaultParkStrategy, ParkStrategy};

/// A borrowed [`Sender`] (`&OneShot`).
pub type RefSender<'a, T> = Sender<T, &'a OneShot<T>>;
/// A borrowed [`Receiver`] (`&OneShot`).
pub type RefReceiver<'a, T> = Receiver<T, &'a OneShot<T>>;
/// An owned [`Sender`] backed by a std [`Arc`].
pub type ArcSender<T> = Sender<T, Arc<OneShot<T>>>;
/// An owned [`Receiver`] backed by a std [`Arc`].
pub type ArcReceiver<T> = Receiver<T, Arc<OneShot<T>>>;
/// An owned [`Sender`] backed by an [`Rc`].
pub type RcSender<T> = Sender<T, Rc<OneShot<T>>>;
/// An owned [`Receiver`] backed by an [`Rc`].
pub type RcReceiver<T> = Receiver<T, Rc<OneShot<T>>>;
/// An owned [`Sender`] backed by a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type TriompheArcSender<T> = Sender<T, TriompheArc<OneShot<T>>>;
/// An owned [`Receiver`] backed by a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type TriompheArcReceiver<T> = Receiver<T, TriompheArc<OneShot<T>>>;

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
    /// The async receiver's waker. Blocking receivers park on [`park_key`](OneShot::park_key)
    /// instead; a single receiver only ever uses one of the two.
    waker: AtomicWaker,
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
            waker: AtomicWaker::new(),
        }
    }

    /// The parking-lot key a blocking receiver waits on.
    #[inline]
    fn park_key(&self) -> usize {
        core::ptr::from_ref(self) as usize
    }

    /// Wakes the receiver regardless of how it is waiting: `wake` delivers to an async waker if one
    /// is registered, and `unpark_all` releases a parked blocking thread. A given receiver only
    /// uses one of the two, so the other call is a cheap no-op.
    #[inline]
    fn wake_receiver(&self) {
        self.waker.wake();
        DefaultParkStrategy::unpark_all(self.park_key());
    }

    /// Splits into borrowed sender/receiver halves.
    ///
    /// Takes `&mut self` so the halves cannot be created more than once.
    pub fn split(&mut self) -> (RefSender<'_, T>, RefReceiver<'_, T>) {
        let channel: &Self = self;
        (Sender { channel, _marker: PhantomData }, Receiver { channel, _marker: PhantomData })
    }

    /// Splits into owned halves backed by a std [`Arc`] (movable across threads/tasks).
    pub fn arc_split(self: &Arc<Self>) -> (ArcSender<T>, ArcReceiver<T>) {
        (
            Sender { channel: Arc::clone(self), _marker: PhantomData },
            Receiver { channel: Arc::clone(self), _marker: PhantomData },
        )
    }

    /// Splits into owned halves backed by an [`Rc`] (single-threaded).
    pub fn rc_split(self: &Rc<Self>) -> (RcSender<T>, RcReceiver<T>) {
        (
            Sender { channel: Rc::clone(self), _marker: PhantomData },
            Receiver { channel: Rc::clone(self), _marker: PhantomData },
        )
    }

    /// Splits into owned halves backed by a `triomphe::Arc`.
    ///
    /// A free-standing associated function (`OneShot::triomphe_arc_split(&arc)`) because
    /// `triomphe::Arc` cannot be a `self` receiver on stable.
    #[cfg(feature = "triomphe-arc")]
    pub fn triomphe_arc_split(
        this: &TriompheArc<Self>,
    ) -> (TriompheArcSender<T>, TriompheArcReceiver<T>) {
        (
            Sender { channel: TriompheArc::clone(this), _marker: PhantomData },
            Receiver { channel: TriompheArc::clone(this), _marker: PhantomData },
        )
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

/// The sending half of a [`OneShot`], generic over the [`Holder`] `H` (a `&OneShot`, `Arc`, `Rc`,
/// ...). See the [`RefSender`] / [`ArcSender`] / [`RcSender`] aliases.
#[derive(Debug)]
pub struct Sender<T, H: Holder<OneShot<T>> = Arc<OneShot<T>>> {
    channel: H,
    _marker: PhantomData<fn() -> T>,
}

/// The receiving half of a [`OneShot`], generic over the [`Holder`] `H`. See the [`RefReceiver`] /
/// [`ArcReceiver`] / [`RcReceiver`] aliases.
#[derive(Debug)]
pub struct Receiver<T, H: Holder<OneShot<T>> = Arc<OneShot<T>>> {
    channel: H,
    _marker: PhantomData<fn() -> T>,
}

impl<T> core::fmt::Debug for OneShot<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OneShot").field("state", &self.state).finish_non_exhaustive()
    }
}

impl<T, H: Holder<OneShot<T>>> Sender<T, H> {
    /// Sends `value`, consuming the sender.
    ///
    /// Returns `Err(value)` if the receiver has already been dropped.
    pub fn send(self, value: T) -> Result<(), T> {
        // Write the value before publishing `SENT`; the receiver reads it under Acquire.
        unsafe { (*self.channel.value.get()).write(value) };

        match self
            .channel
            .state
            .compare_exchange(EMPTY, SENT, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => {
                self.channel.wake_receiver();
                Ok(())
            }
            Err(_) => {
                // The receiver is gone (RX_CLOSED). Reclaim the value we just wrote.
                let value = unsafe { (*self.channel.value.get()).assume_init_read() };
                Err(value)
            }
        }
    }

    /// Returns `true` if the receiver has been dropped, so [`send`](Sender::send) would fail.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.channel.state.load(Ordering::Acquire) == RX_CLOSED
    }
}

impl<T, H: Holder<OneShot<T>>> Drop for Sender<T, H> {
    fn drop(&mut self) {
        // If we never sent, transition EMPTY→TX_CLOSED and wake the receiver. If a value was sent
        // (SENT) or the receiver already left (RX_CLOSED), the CAS fails and there is nothing to
        // do.
        if self
            .channel
            .state
            .compare_exchange(EMPTY, TX_CLOSED, Ordering::Release, Ordering::Relaxed)
            .is_ok()
        {
            self.channel.wake_receiver();
        }
    }
}

impl<T, H: Holder<OneShot<T>>> Receiver<T, H> {
    /// Reads the value if one is available or the sender has closed, without waiting.
    fn poll_value(&self) -> Option<Result<T, RecvError>> {
        match self.channel.state.load(Ordering::Acquire) {
            SENT => {
                let value = unsafe { (*self.channel.value.get()).assume_init_read() };
                self.channel.state.store(TAKEN, Ordering::Release);
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
            // Park until the state leaves EMPTY. `park` re-checks the predicate under the parking
            // lock, so a send racing this call is not missed.
            DefaultParkStrategy::park(self.channel.park_key(), || {
                self.channel.state.load(Ordering::Acquire) == EMPTY
            });
        }
    }

    /// Resolves once the value is received or the sender is dropped.
    pub async fn recv_async(self) -> Result<T, RecvError> {
        poll_fn(|cx| {
            if let Some(result) = self.poll_value() {
                return Poll::Ready(result);
            }
            // Register, then re-check: a send between the first check and the registration is
            // caught here, and one after it wakes our registered waker.
            self.channel.waker.register(cx.waker());
            match self.poll_value() {
                Some(result) => Poll::Ready(result),
                None => Poll::Pending,
            }
        })
        .await
    }
}

impl<T, H: Holder<OneShot<T>>> Drop for Receiver<T, H> {
    fn drop(&mut self) {
        // Signal the sender that we are gone (only meaningful while still EMPTY). A
        // SENT-but-untaken value is left in place for `OneShot::drop` to clean up.
        let _ = self.channel.state.compare_exchange(
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
        let mut channel = OneShot::new();
        let (tx, rx) = channel.split();
        tx.send(42u32).unwrap();
        assert_eq!(rx.recv().unwrap(), 42);
    }

    #[test]
    fn try_recv_empty_then_value() {
        let mut channel = OneShot::<u32>::new();
        let (tx, rx) = channel.split();
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        tx.send(7).unwrap();
        assert_eq!(rx.try_recv(), Ok(7));
    }

    #[test]
    fn sender_dropped_without_send() {
        let mut channel = OneShot::<u32>::new();
        let (tx, rx) = channel.split();
        drop(tx);
        assert_eq!(rx.recv(), Err(RecvError));
    }

    #[test]
    fn receiver_dropped_before_send() {
        let mut channel = OneShot::new();
        let (tx, rx) = channel.split();
        drop(rx);
        assert_eq!(tx.send(9u32), Err(9));
    }

    #[test]
    fn blocking_recv_waits_for_send() {
        let mut channel = OneShot::<u32>::new();
        let (tx, rx) = channel.split();
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
            let mut channel = OneShot::new();
            let (tx, rx) = channel.split();
            assert!(tx.send(Guard(Arc::clone(&dropped))).is_ok());
            drop(rx);
        }
        assert_eq!(dropped.load(Ordering::Relaxed), 1, "sent-but-unreceived value must be dropped");
    }

    #[tokio::test]
    async fn async_recv_waits_for_send() {
        let mut channel = OneShot::<u32>::new();
        let (tx, rx) = channel.split();
        let (got, ()) = tokio::join!(rx.recv_async(), async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tx.send(55).unwrap();
        });
        assert_eq!(got, Ok(55));
    }

    #[tokio::test]
    async fn async_recv_sender_dropped() {
        let mut channel = OneShot::<u32>::new();
        let (tx, rx) = channel.split();
        let (got, ()) = tokio::join!(rx.recv_async(), async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(tx);
        });
        assert_eq!(got, Err(RecvError));
    }

    #[test]
    fn arc_split_moves_owned_halves_across_threads() {
        let channel = Arc::new(OneShot::<u32>::new());
        let (tx, rx) = channel.arc_split();
        let sender = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            tx.send(42).unwrap();
        });
        let got = std::thread::spawn(move || rx.recv()).join().unwrap();
        sender.join().unwrap();
        assert_eq!(got, Ok(42));
    }
}
