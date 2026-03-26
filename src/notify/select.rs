use core::pin::Pin;
use core::task::{Context, RawWaker, RawWakerVTable, Waker};

use parking_lot_core::{DEFAULT_PARK_TOKEN, DEFAULT_UNPARK_TOKEN};

use super::NotifyListener;

/// Blocks until one of the listeners is notified.
/// Returns the index of the listener that triggered, or None if listeners is empty.
pub fn select_blocking(listeners: &mut [NotifyListener<'_>]) -> Option<usize> {
    if listeners.is_empty() {
        return None;
    }

    for (index, listener) in listeners.iter().enumerate() {
        if listener.is_notified() {
            return Some(index);
        }
    }

    for _ in 0..64 {
        core::hint::spin_loop();
        for (index, listener) in listeners.iter().enumerate() {
            if listener.is_notified() {
                return Some(index);
            }
        }
    }

    // Key is the address of the `listeners` variable (not the slice), which is unique to this
    // scope.
    let key = core::ptr::from_ref(&listeners) as usize;
    let waker = create_waker(key);
    let mut cx = Context::from_waker(&waker);

    loop {
        // This registers our thread-unparking waker in the Notify's WakerQueue.
        for (index, listener) in listeners.iter_mut().enumerate() {
            if Pin::new(listener).poll(&mut cx).is_ready() {
                return Some(index);
            }
        }

        // We use a validation closure to ensure we don't sleep if a notification arrived between
        // the poll and the park call.
        unsafe {
            parking_lot_core::park(
                key,
                || {
                    // Validation: if any listener became ready, don't sleep.
                    !listeners.iter().any(|l| l.is_notified())
                },
                || {},
                |_, _| {},
                DEFAULT_PARK_TOKEN,
                None,
            );
        }
    }
}

fn create_waker(key: usize) -> Waker {
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake, |_| {});

    unsafe fn clone(ptr: *const ()) -> RawWaker {
        RawWaker::new(ptr, &VTABLE)
    }

    unsafe fn wake(ptr: *const ()) {
        unsafe {
            parking_lot_core::unpark_all(ptr as usize, DEFAULT_UNPARK_TOKEN);
        }
    }

    unsafe { Waker::from_raw(RawWaker::new(key as *const (), &VTABLE)) }
}
