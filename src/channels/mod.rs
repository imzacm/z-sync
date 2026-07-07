//! Channels: single-value and multi-value message passing, unified across blocking and async.
//!
//! Each channel is a standalone, allocation-free core that the caller owns and stores however they
//! like (on the stack with scoped threads, in a `static`, behind their own `Arc`/`Rc`, ...).
//! [`split`](oneshot::OneShot::split) hands out the borrowed sender/receiver halves; wakeups reuse
//! [`Notify`](crate::Notify), so receivers can wait from blocking or async code.
//!
//! - [`oneshot`] — a single value from one sender to one receiver.
//! - [`watch`] — a latest value observed by many receivers.
//! - [`broadcast`] — a bounded ring buffer where every receiver sees every message.

pub mod broadcast;
pub mod oneshot;
pub mod watch;
