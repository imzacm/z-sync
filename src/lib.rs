#![deny(unused_imports, clippy::all)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod lock;
pub mod notify;
pub mod observable_lock;
pub mod park_strategy;
pub mod semaphore;
pub mod waker_queue;
pub mod waker_storage;

pub use self::lock::{
    Lock, Lock16, Lock32, Lock64, LockState, LockStateU16, LockStateU32, LockStateU64,
};
pub use self::notify::{
    Notify, Notify16, Notify16Boxed, Notify32, Notify32Boxed, Notify64, Notify64Boxed, NotifyState,
    NotifyStateU16, NotifyStateU32, NotifyStateU64,
};
pub use self::observable_lock::ObservableLock;
pub use self::park_strategy::ParkStrategy;
pub use self::semaphore::{
    Semaphore, Semaphore16, Semaphore32, Semaphore64, SemaphoreState, SemaphoreStateU16,
    SemaphoreStateU32, SemaphoreStateU64,
};
pub use self::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};
