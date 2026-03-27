#[cfg(feature = "std")]
pub use self::std_impl::*;

#[cfg(feature = "std")]
pub type DefaultParkStrategy = ParkingLot;

#[cfg(not(feature = "std"))]
pub type DefaultUnparkStrategy = Spin;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FilterOp {
    Unpark,
    Skip,
    Stop,
}

pub trait ParkStrategy {
    fn park<V>(key: usize, validate: V)
    where
        V: FnMut() -> bool;

    #[cfg(feature = "std")]
    fn park_timeout<V>(key: usize, validate: V, timeout: std::time::Instant)
    where
        V: FnMut() -> bool;

    fn unpark_one(key: usize) -> bool;

    fn unpark_filter<F>(key: usize, filter: F) -> usize
    where
        F: FnMut() -> FilterOp;

    fn unpark_all(key: usize) -> usize;
}

#[cfg(feature = "std")]
mod std_impl {
    use crate::park_strategy::{FilterOp, ParkStrategy};

    #[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
    pub struct ParkingLot;

    impl ParkStrategy for ParkingLot {
        #[inline(always)]
        fn park<V>(key: usize, validate: V)
        where
            V: FnMut() -> bool,
        {
            unsafe {
                parking_lot_core::park(
                    key,
                    validate,
                    || {},
                    |_, _| {},
                    parking_lot_core::DEFAULT_PARK_TOKEN,
                    None,
                );
            }
        }

        #[inline(always)]
        fn park_timeout<V>(key: usize, validate: V, timeout: std::time::Instant)
        where
            V: FnMut() -> bool,
        {
            unsafe {
                parking_lot_core::park(
                    key,
                    validate,
                    || {},
                    |_, _| {},
                    parking_lot_core::DEFAULT_PARK_TOKEN,
                    Some(timeout),
                );
            }
        }

        #[inline(always)]
        fn unpark_one(key: usize) -> bool {
            let result = unsafe {
                parking_lot_core::unpark_one(key, |_| parking_lot_core::DEFAULT_UNPARK_TOKEN)
            };
            result.unparked_threads != 0
        }

        #[inline(always)]
        fn unpark_filter<F>(key: usize, mut filter: F) -> usize
        where
            F: FnMut() -> FilterOp,
        {
            let result = unsafe {
                parking_lot_core::unpark_filter(
                    key,
                    |_| match filter() {
                        FilterOp::Unpark => parking_lot_core::FilterOp::Unpark,
                        FilterOp::Skip => parking_lot_core::FilterOp::Skip,
                        FilterOp::Stop => parking_lot_core::FilterOp::Stop,
                    },
                    |_| parking_lot_core::DEFAULT_UNPARK_TOKEN,
                )
            };
            result.unparked_threads
        }

        #[inline(always)]
        fn unpark_all(key: usize) -> usize {
            unsafe { parking_lot_core::unpark_all(key, parking_lot_core::DEFAULT_UNPARK_TOKEN) }
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spin;

impl ParkStrategy for Spin {
    #[inline(always)]
    fn park<V>(_key: usize, mut validate: V)
    where
        V: FnMut() -> bool,
    {
        while validate() {
            core::hint::spin_loop();
        }
    }

    #[cfg_attr(feature = "std", inline(always))]
    #[cfg(feature = "std")]
    fn park_timeout<V>(_key: usize, mut validate: V, timeout: std::time::Instant)
    where
        V: FnMut() -> bool,
    {
        while validate() && std::time::Instant::now() < timeout {
            core::hint::spin_loop();
        }
    }

    #[inline(always)]
    fn unpark_one(_key: usize) -> bool {
        true
    }

    #[inline(always)]
    fn unpark_filter<F>(_key: usize, mut filter: F) -> usize
    where
        F: FnMut() -> FilterOp,
    {
        filter();
        1
    }

    #[inline(always)]
    fn unpark_all(_key: usize) -> usize {
        1
    }
}
