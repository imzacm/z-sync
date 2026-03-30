use core::pin::Pin;
use core::sync::atomic::Ordering;
use core::task::{Context, Poll};

#[cfg(feature = "std")]
pub use self::timeout::NotifyTimeoutListener;
use super::Notify;
use crate::park_strategy::{DefaultParkStrategy, ParkStrategy};
use crate::waker_queue::WakerTicket;

/// A listener that was created from a [`Notify`].
///
/// Supports both blocking (`.wait()`) and async (`.await`) usage.
#[derive(Debug)]
pub struct NotifyListener<'a, P: ParkStrategy = DefaultParkStrategy> {
    notify: &'a Notify<P>,
    /// The epoch snapshot taken when this listener was created.
    epoch: u32,
    waker_node_ticket: Option<WakerTicket>,
}

impl<'a, P: ParkStrategy> NotifyListener<'a, P> {
    pub(super) fn new(notify: &'a Notify<P>, epoch: u32) -> Self {
        Self { notify, epoch, waker_node_ticket: None }
    }

    #[inline(always)]
    pub fn notification(&self) -> &'a Notify<P> {
        self.notify
    }

    #[inline(always)]
    pub fn is_notification(&self, notify: &Notify<P>) -> bool {
        core::ptr::eq(self.notify, notify)
    }

    /// Returns `true` if a notification has occurred since this listener was created.
    #[inline(always)]
    pub fn is_notified(&self) -> bool {
        self.notify.load_state(Ordering::Acquire).epoch() != self.epoch
    }

    #[cfg(feature = "std")]
    pub fn with_timeout(self, timeout: std::time::Duration) -> NotifyTimeoutListener<'a, P> {
        NotifyTimeoutListener::new(self, timeout)
    }

    /// Blocks the current thread until a notification arrives.
    pub fn wait(self) {
        if self.is_notified() {
            return;
        }

        for _ in 0..64 {
            if self.is_notified() {
                return;
            }
            core::hint::spin_loop();
        }

        self.notify.add_parkers(1, Ordering::SeqCst);

        loop {
            P::park(self.notify.parking_key(), || !self.is_notified());

            if self.is_notified() {
                self.notify.sub_parkers(1, Ordering::Relaxed);
                return;
            }

            // Spurious wakeup — loop back and park again.
        }
    }
}

impl<'a> Future for NotifyListener<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.is_notified() {
            return Poll::Ready(());
        }

        let this = self.as_mut().get_mut();

        // This creates a memory barrier, so even if we don't need the lock (early return), skipping
        // the lock can cause deadlocks (rare).
        let mut queue = this.notify.get_async_wakers().lock();

        if this.is_notified() {
            // We already hold the lock, clean up now so Drop doesn't have to re-lock.
            if let Some(ticket) = this.waker_node_ticket.take()
                && queue.remove(ticket)
            {
                self.notify.sub_wakers(1, Ordering::Relaxed);
            }

            return Poll::Ready(());
        }

        if let Some(ticket) = this.waker_node_ticket {
            let node = queue.node_mut(ticket.index());

            if node.generation() == ticket.generation() {
                if node.waker().is_none_or(|w| !w.will_wake(cx.waker())) {
                    *node.waker_mut() = Some(cx.waker().clone());
                }
            } else {
                // Our slot was popped and recycled by a previous wakeup. We must re-enqueue
                // ourselves to prevent a lost wakeup.
                this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
                this.notify.add_wakers(1, Ordering::SeqCst);
            }
        } else {
            // First time being polled.
            this.waker_node_ticket = Some(queue.push(cx.waker().clone()));
            this.notify.add_wakers(1, Ordering::SeqCst);
        }

        if this.is_notified() {
            if let Some(ticket) = this.waker_node_ticket.take()
                && queue.remove(ticket)
            {
                self.notify.sub_wakers(1, Ordering::Relaxed);
            }
            return Poll::Ready(());
        }

        Poll::Pending
    }
}

impl<'a, P: ParkStrategy> Drop for NotifyListener<'a, P> {
    fn drop(&mut self) {
        if let Some(ticket) = self.waker_node_ticket.take()
            && self.notify.get_async_wakers().lock().remove(ticket)
        {
            self.notify.sub_wakers(1, Ordering::Relaxed);
        }
    }
}

#[cfg(feature = "std")]
mod timeout {
    use std::time::Duration;

    use super::*;

    #[derive(Debug)]
    pub struct NotifyTimeoutListener<'a, P: ParkStrategy = DefaultParkStrategy> {
        listener: NotifyListener<'a, P>,
        timeout: Duration,
    }

    impl<'a, P: ParkStrategy> NotifyTimeoutListener<'a, P> {
        pub(super) fn new(listener: NotifyListener<'a, P>, timeout: Duration) -> Self {
            Self { listener, timeout }
        }

        #[inline(always)]
        pub fn notification(&self) -> &'a Notify<P> {
            self.listener.notification()
        }

        #[inline(always)]
        pub fn is_notification(&self, notify: &Notify<P>) -> bool {
            self.listener.is_notification(notify)
        }

        /// Returns `true` if a notification has occurred since this listener was created.
        #[inline(always)]
        pub fn is_notified(&self) -> bool {
            self.listener.is_notified()
        }

        #[inline(always)]
        pub fn timeout(&self) -> Duration {
            self.timeout
        }

        #[inline(always)]
        pub fn set_timeout(&mut self, timeout: Duration) -> &mut Self {
            self.timeout = timeout;
            self
        }

        #[inline(always)]
        pub fn with_timeout(mut self, timeout: Duration) -> Self {
            self.timeout = timeout;
            self
        }

        pub fn wait(self) -> Result<(), Self> {
            let timeout = std::time::Instant::now() + self.timeout;

            if self.is_notified() {
                return Ok(());
            }

            for _ in 0..64 {
                if std::time::Instant::now() >= timeout {
                    return Err(self);
                }

                if self.is_notified() {
                    return Ok(());
                }
                core::hint::spin_loop();
            }

            self.listener.notify.add_parkers(1, Ordering::SeqCst);

            loop {
                if std::time::Instant::now() >= timeout {
                    return Err(self);
                }

                P::park_timeout(
                    self.listener.notify.parking_key(),
                    || !self.is_notified() && std::time::Instant::now() < timeout,
                    timeout,
                );

                if self.is_notified() {
                    self.listener.notify.sub_parkers(1, Ordering::Relaxed);
                    return Ok(());
                }

                // Spurious wakeup — loop back and park again.
            }
        }
    }
}
