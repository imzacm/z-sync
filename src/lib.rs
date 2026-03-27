#![deny(unused_imports, clippy::all)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod lock;
pub mod notify;
pub mod park_strategy;
pub mod waker_queue;

pub use self::lock::Lock;
pub use self::notify::Notify;
pub use self::park_strategy::ParkStrategy;
