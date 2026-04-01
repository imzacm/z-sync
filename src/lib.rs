#![deny(unused_imports, clippy::all)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod lock;
pub mod notify;
pub mod observable_lock;
pub mod park_strategy;
pub mod waker_queue;

pub use self::lock::{
    Lock, Lock16, Lock32, Lock64, LockState, LockStateU16, LockStateU32, LockStateU64,
};
pub use self::notify::{
    Notify, Notify16, Notify32, Notify64, NotifyState, NotifyStateU16, NotifyStateU32,
    NotifyStateU64,
};
pub use self::observable_lock::ObservableLock;
pub use self::park_strategy::ParkStrategy;
