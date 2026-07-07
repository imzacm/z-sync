//! A broadcast channel: every value sent is observed by every [`Receiver`].
//!
//! The state lives in a standalone [`Broadcast`] the caller owns; [`Broadcast::split`] yields the
//! borrowed [`Sender`]/[`Receiver`] halves (both cloneable). Values are kept in a bounded ring
//! buffer and each receiver clones each value as it reads; a receiver more than `capacity` messages
//! behind is told it [`Lagged`](RecvError::Lagged). Each slot is a [`Lock`](crate::Lock) so
//! receivers read concurrently while a sender overwrites; wakeups reuse [`Notify`](crate::Notify).

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "triomphe-arc")]
use triomphe::Arc as TriompheArc;

use crate::{Holder, Lock16, Lock32, Notify64};

/// A borrowed [`Sender`] (`&Broadcast`).
pub type RefSender<'a, T> = Sender<T, &'a Broadcast<T>>;
/// A borrowed [`Receiver`] (`&Broadcast`).
pub type RefReceiver<'a, T> = Receiver<T, &'a Broadcast<T>>;
/// An owned [`Sender`] backed by a std [`Arc`].
pub type ArcSender<T> = Sender<T, Arc<Broadcast<T>>>;
/// An owned [`Receiver`] backed by a std [`Arc`].
pub type ArcReceiver<T> = Receiver<T, Arc<Broadcast<T>>>;
/// An owned [`Sender`] backed by an [`Rc`].
pub type RcSender<T> = Sender<T, Rc<Broadcast<T>>>;
/// An owned [`Receiver`] backed by an [`Rc`].
pub type RcReceiver<T> = Receiver<T, Rc<Broadcast<T>>>;
/// An owned [`Sender`] backed by a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type TriompheArcSender<T> = Sender<T, TriompheArc<Broadcast<T>>>;
/// An owned [`Receiver`] backed by a `triomphe::Arc`.
#[cfg(feature = "triomphe-arc")]
pub type TriompheArcReceiver<T> = Receiver<T, TriompheArc<Broadcast<T>>>;

const EMPTY_POS: u64 = u64::MAX;

/// Error returned by [`Receiver::recv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecvError {
    /// All senders have been dropped and the receiver has read every remaining message.
    Closed,
    /// The receiver fell behind and `n` messages were skipped; the cursor has advanced to the
    /// oldest retained message.
    Lagged(u64),
}

/// Error returned by [`Receiver::try_recv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    /// No message is currently available, but senders are still alive.
    Empty,
    /// All senders have been dropped and no message remains.
    Closed,
    /// The receiver fell behind and `n` messages were skipped.
    Lagged(u64),
}

/// Error returned by [`Sender::send`] when there are no receivers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendError<T>(pub T);

struct SlotData<T> {
    /// Sequence position stored in this slot, or [`EMPTY_POS`].
    pos: u64,
    value: Option<T>,
}

/// A broadcast channel. Own one of these and call [`split`](Broadcast::split) to get the halves.
pub struct Broadcast<T> {
    buffer: Box<[Lock32<SlotData<T>>]>,
    mask: u64,
    capacity: u64,
    /// Next position to be written; the published high-water mark receivers read up to.
    tail: AtomicU64,
    /// Serialises senders so positions are claimed and published in order.
    send_lock: Lock16<()>,
    notify: Notify64,
    senders: AtomicUsize,
    receivers: AtomicUsize,
}

impl<T> Broadcast<T> {
    /// Creates a broadcast channel that retains up to `capacity` messages (rounded up to a power of
    /// two, minimum 1).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1).next_power_of_two() as u64;
        let buffer: Vec<Lock32<SlotData<T>>> = (0..capacity)
            .map(|_| Lock32::new(SlotData { pos: EMPTY_POS, value: None }))
            .collect();

        Self {
            buffer: buffer.into_boxed_slice(),
            mask: capacity - 1,
            capacity,
            tail: AtomicU64::new(0),
            send_lock: Lock16::new(()),
            notify: Notify64::new(),
            senders: AtomicUsize::new(1),
            receivers: AtomicUsize::new(1),
        }
    }

    /// Splits into borrowed sender/receiver halves.
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
        let next = channel.tail.load(Ordering::Acquire);
        (
            Sender { channel: channel.clone(), _marker: PhantomData },
            Receiver { channel, next, _marker: PhantomData },
        )
    }
}

/// The sending half of a [`Broadcast`], generic over the [`Holder`] `H`. Clone to send from multiple
/// places. See the [`RefSender`] / [`ArcSender`] / [`RcSender`] aliases.
pub struct Sender<T, H: Holder<Broadcast<T>> = Arc<Broadcast<T>>> {
    channel: H,
    _marker: PhantomData<fn() -> T>,
}

/// A receiving half of a [`Broadcast`], generic over the [`Holder`] `H`. Clone (or
/// [`Sender::subscribe`]) for more receivers. See the [`RefReceiver`] / [`ArcReceiver`] /
/// [`RcReceiver`] aliases.
pub struct Receiver<T, H: Holder<Broadcast<T>> = Arc<Broadcast<T>>> {
    channel: H,
    /// Next position this receiver will read.
    next: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<T, H: Holder<Broadcast<T>>> Sender<T, H> {
    /// Sends a value to all receivers, overwriting the oldest message if the buffer is full.
    ///
    /// Returns `Err(value)` if there are no receivers.
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        if self.channel.receivers.load(Ordering::Acquire) == 0 {
            return Err(SendError(value));
        }

        // Serialise senders so positions are claimed and published monotonically.
        let _send = self.channel.send_lock.write();
        let pos = self.channel.tail.load(Ordering::Relaxed);
        let slot = &self.channel.buffer[(pos & self.channel.mask) as usize];
        {
            let mut guard = slot.write();
            guard.pos = pos;
            guard.value = Some(value);
        }
        self.channel.tail.store(pos.wrapping_add(1), Ordering::Release);
        drop(_send);

        self.channel.notify.notify(usize::MAX);
        Ok(())
    }

    /// The number of live receivers.
    #[inline]
    pub fn receiver_count(&self) -> usize {
        self.channel.receivers.load(Ordering::Acquire)
    }

    /// Creates a new receiver that observes messages sent after this call.
    pub fn subscribe(&self) -> Receiver<T, H> {
        self.channel.receivers.fetch_add(1, Ordering::Release);
        Receiver {
            channel: self.channel.clone(),
            next: self.channel.tail.load(Ordering::Acquire),
            _marker: PhantomData,
        }
    }
}

impl<T, H: Holder<Broadcast<T>>> Clone for Sender<T, H> {
    fn clone(&self) -> Self {
        self.channel.senders.fetch_add(1, Ordering::Release);
        Sender { channel: self.channel.clone(), _marker: PhantomData }
    }
}

impl<T, H: Holder<Broadcast<T>>> Drop for Sender<T, H> {
    fn drop(&mut self) {
        if self.channel.senders.fetch_sub(1, Ordering::Release) == 1 {
            // Last sender gone: wake receivers so they observe closure.
            self.channel.notify.notify(usize::MAX);
        }
    }
}

impl<T: Clone, H: Holder<Broadcast<T>>> Receiver<T, H> {
    /// Reads the next message if one is available. Split fields so a listener borrowing `channel` can
    /// be held across the check without conflicting with the mutable `next` borrow.
    fn poll_next(channel: &Broadcast<T>, next: &mut u64) -> Result<Option<T>, TryRecvError> {
        let tail = channel.tail.load(Ordering::Acquire);

        if *next == tail {
            return if channel.senders.load(Ordering::Acquire) == 0 {
                // Re-check tail in case a send raced with the last sender drop.
                if channel.tail.load(Ordering::Acquire) != *next {
                    Ok(None)
                } else {
                    Err(TryRecvError::Closed)
                }
            } else {
                Ok(None)
            };
        }

        let slot = &channel.buffer[(*next & channel.mask) as usize];
        let guard = slot.read();
        if guard.pos == *next {
            let value = guard.value.clone().expect("occupied slot has a value");
            *next = next.wrapping_add(1);
            Ok(Some(value))
        } else {
            // The slot was overwritten: we lagged. Skip to the oldest retained message.
            drop(guard);
            let oldest = tail.wrapping_sub(channel.capacity);
            let skipped = oldest.wrapping_sub(*next);
            *next = oldest;
            Err(TryRecvError::Lagged(skipped))
        }
    }

    /// Attempts to receive the next message without waiting.
    pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
        match Self::poll_next(&*self.channel, &mut self.next) {
            Ok(Some(value)) => Ok(value),
            Ok(None) => Err(TryRecvError::Empty),
            Err(e) => Err(e),
        }
    }

    /// Blocks until the next message is available or the channel closes.
    pub fn recv(&mut self) -> Result<T, RecvError> {
        loop {
            match Self::poll_next(&*self.channel, &mut self.next) {
                Ok(Some(value)) => return Ok(value),
                Err(TryRecvError::Lagged(n)) => return Err(RecvError::Lagged(n)),
                Err(TryRecvError::Closed) => return Err(RecvError::Closed),
                Ok(None) | Err(TryRecvError::Empty) => {}
            }
            let listener = self.channel.notify.listener();
            match Self::poll_next(&*self.channel, &mut self.next) {
                Ok(Some(value)) => return Ok(value),
                Err(TryRecvError::Lagged(n)) => return Err(RecvError::Lagged(n)),
                Err(TryRecvError::Closed) => return Err(RecvError::Closed),
                Ok(None) | Err(TryRecvError::Empty) => {}
            }
            listener.wait();
        }
    }

    /// Resolves once the next message is available or the channel closes.
    pub async fn recv_async(&mut self) -> Result<T, RecvError> {
        loop {
            match Self::poll_next(&*self.channel, &mut self.next) {
                Ok(Some(value)) => return Ok(value),
                Err(TryRecvError::Lagged(n)) => return Err(RecvError::Lagged(n)),
                Err(TryRecvError::Closed) => return Err(RecvError::Closed),
                Ok(None) | Err(TryRecvError::Empty) => {}
            }
            let listener = self.channel.notify.listener();
            match Self::poll_next(&*self.channel, &mut self.next) {
                Ok(Some(value)) => return Ok(value),
                Err(TryRecvError::Lagged(n)) => return Err(RecvError::Lagged(n)),
                Err(TryRecvError::Closed) => return Err(RecvError::Closed),
                Ok(None) | Err(TryRecvError::Empty) => {}
            }
            listener.await;
        }
    }
}

impl<T, H: Holder<Broadcast<T>>> Clone for Receiver<T, H> {
    fn clone(&self) -> Self {
        self.channel.receivers.fetch_add(1, Ordering::Release);
        Receiver { channel: self.channel.clone(), next: self.next, _marker: PhantomData }
    }
}

impl<T, H: Holder<Broadcast<T>>> Drop for Receiver<T, H> {
    fn drop(&mut self) {
        self.channel.receivers.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn send_recv_in_order() {
        let mut channel = Broadcast::new(4);
        let (tx, mut rx) = channel.split();
        tx.send(1u32).unwrap();
        tx.send(2).unwrap();
        assert_eq!(rx.recv(), Ok(1));
        assert_eq!(rx.recv(), Ok(2));
    }

    #[test]
    fn all_receivers_observe_every_message() {
        let mut channel = Broadcast::new(4);
        let (tx, mut rx1) = channel.split();
        let mut rx2 = rx1.clone();
        tx.send(10u32).unwrap();
        tx.send(20).unwrap();
        assert_eq!(rx1.recv(), Ok(10));
        assert_eq!(rx1.recv(), Ok(20));
        assert_eq!(rx2.recv(), Ok(10));
        assert_eq!(rx2.recv(), Ok(20));
    }

    #[test]
    fn lagging_receiver_is_told() {
        // capacity rounds up to 2.
        let mut channel = Broadcast::new(2);
        let (tx, mut rx) = channel.split();
        for i in 0..5u32 {
            tx.send(i).unwrap();
        }
        // Buffer holds the last 2 (positions 3,4). Reading from 0 => lagged by 3.
        assert_eq!(rx.recv(), Err(RecvError::Lagged(3)));
        assert_eq!(rx.recv(), Ok(3));
        assert_eq!(rx.recv(), Ok(4));
    }

    #[test]
    fn closed_when_all_senders_dropped() {
        let mut channel = Broadcast::<u32>::new(4);
        let (tx, mut rx) = channel.split();
        tx.send(1).unwrap();
        drop(tx);
        assert_eq!(rx.recv(), Ok(1));
        assert_eq!(rx.recv(), Err(RecvError::Closed));
    }

    #[test]
    fn send_fails_without_receivers() {
        let mut channel = Broadcast::new(4);
        let (tx, rx) = channel.split();
        drop(rx);
        assert_eq!(tx.send(1u32), Err(SendError(1)));
    }

    #[test]
    fn subscribe_only_sees_future_messages() {
        let mut channel = Broadcast::new(4);
        let (tx, _rx) = channel.split();
        tx.send(1u32).unwrap();
        let mut late = tx.subscribe();
        tx.send(2).unwrap();
        assert_eq!(late.recv(), Ok(2));
    }

    #[test]
    fn blocking_fanout_across_threads() {
        const RECEIVERS: usize = 4;
        const MESSAGES: u32 = 100;

        let mut channel = Broadcast::<u32>::new(256);
        let (tx, rx0) = channel.split();
        let expected: u32 = (0..MESSAGES).sum();

        std::thread::scope(|s| {
            for _ in 0..RECEIVERS {
                let mut rx = rx0.clone();
                s.spawn(move || {
                    let mut sum = 0u32;
                    let mut count = 0;
                    while count < MESSAGES {
                        match rx.recv() {
                            Ok(v) => {
                                sum += v;
                                count += 1;
                            }
                            Err(RecvError::Lagged(_)) => panic!("unexpected lag"),
                            Err(RecvError::Closed) => break,
                        }
                    }
                    assert_eq!(sum, expected);
                });
            }
            drop(rx0);
            for i in 0..MESSAGES {
                tx.send(i).unwrap();
            }
        });
    }

    #[tokio::test]
    async fn async_recv_waits() {
        let mut channel = Broadcast::<u32>::new(4);
        let (tx, mut rx) = channel.split();
        let (got, ()) = tokio::join!(rx.recv_async(), async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tx.send(77).unwrap();
        });
        assert_eq!(got, Ok(77));
    }

    #[test]
    fn arc_split_moves_owned_halves_across_threads() {
        let channel = std::sync::Arc::new(Broadcast::<u32>::new(4));
        let (tx, mut rx) = channel.arc_split();
        let waiter = std::thread::spawn(move || rx.recv());
        std::thread::sleep(Duration::from_millis(20));
        tx.send(5).unwrap();
        assert_eq!(waiter.join().unwrap(), Ok(5));
    }
}
