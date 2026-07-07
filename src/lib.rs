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
    Lock, Lock16, Lock16Boxed, Lock16Inline, Lock32, Lock32Boxed, Lock32Inline, Lock64,
    Lock64Boxed, Lock64Inline, LockState, LockStateU16, LockStateU32, LockStateU64,
};
pub use self::notify::{
    Notify, Notify16, Notify16Boxed, Notify16Inline, Notify32, Notify32Boxed, Notify32Inline,
    Notify64, Notify64Boxed, Notify64Inline, NotifyState, NotifyStateU16, NotifyStateU32,
    NotifyStateU64,
};
pub use self::observable_lock::ObservableLock;
pub use self::park_strategy::ParkStrategy;
pub use self::semaphore::{
    Semaphore, Semaphore16, Semaphore16Boxed, Semaphore16Inline, Semaphore32, Semaphore32Boxed,
    Semaphore32Inline, Semaphore64, Semaphore64Boxed, Semaphore64Inline, SemaphoreState,
    SemaphoreStateU16, SemaphoreStateU32, SemaphoreStateU64,
};
pub use self::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};
