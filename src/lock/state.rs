#![allow(clippy::new_without_default)]

use core::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};

use num_traits::{ConstOne, ConstZero, NumCast};

pub trait LockState: Sized + Copy + Clone + PartialEq + Eq {
    type Atomic: core::fmt::Debug;

    type ReadWakers: Eq + Ord + NumCast + ConstZero + ConstOne;
    type ReadParked: Eq + Ord + NumCast + ConstZero + ConstOne;
    type WriteWakers: Eq + Ord + NumCast + ConstZero + ConstOne;
    type WriteParked: Eq + Ord + NumCast + ConstZero + ConstOne;
    type Writers: Eq + Ord + NumCast + ConstZero + ConstOne;
    type Readers: Eq + Ord + NumCast + ConstZero + ConstOne;

    type BatchSub: Copy + Clone;

    const INITIAL_ATOMIC: Self::Atomic;

    // --- State Constructors ---
    fn empty() -> Self;
    fn with_writer() -> Self;

    // --- State Getters ---
    fn readers(self) -> Self::Readers;
    fn writers(self) -> Self::Writers;
    fn write_parked(self) -> Self::WriteParked;
    fn write_wakers(self) -> Self::WriteWakers;
    fn read_parked(self) -> Self::ReadParked;
    fn read_wakers(self) -> Self::ReadWakers;

    // --- State Conditions ---
    fn has_readers_or_writers(self) -> bool;
    fn has_any_write_state(self) -> bool;
    fn has_write_waiters(self) -> bool;
    fn has_read_waiters(self) -> bool;
    fn has_any_waiters(self) -> bool;

    // --- State Mutations (Pure) ---
    fn add_reader_state(self) -> Self;
    fn sub_reader_state(self) -> Self;
    fn add_writer_state(self) -> Self;
    fn sub_writer_state(self) -> Self;

    // --- Atomic Operations ---
    fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_compare_exchange_weak(
        atomic: &Self::Atomic,
        current: Self,
        new: Self,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self, Self>;

    fn atomic_add_read_waker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_sub_read_waker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_add_read_parker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_sub_read_parker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_add_write_waker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_sub_write_waker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_add_write_parker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_sub_write_parker(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_add_writer(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_sub_writer(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_add_reader(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_sub_reader(atomic: &Self::Atomic, order: Ordering) -> Self;

    // --- Batch Subtraction ---
    fn batch_sub_new() -> Self::BatchSub;
    fn batch_sub_read_waker(batch: Self::BatchSub, n: Self::ReadWakers) -> Self::BatchSub;
    fn batch_sub_write_waker(batch: Self::BatchSub, n: Self::WriteWakers) -> Self::BatchSub;
    fn atomic_fetch_sub_batch(atomic: &Self::Atomic, batch: Self::BatchSub, order: Ordering);
}

#[macro_export]
macro_rules! atomic_lock_state {
    (
        $vis:vis struct $struct_name:ident(
            $atomic_ty:ident($prim_ty:ty) {
                read_wakers: $rw_ty:ty = $rw_bits:expr,
                read_parked: $rp_ty:ty = $rp_bits:expr,
                write_wakers: $ww_ty:ty = $ww_bits:expr,
                write_parked: $wp_ty:ty = $wp_bits:expr,
                writers: $w_ty:ty = $w_bits:expr,
                readers: $r_ty:ty = $r_bits:expr $(,)?
            }
        )
    ) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        $vis struct $struct_name(pub $prim_ty);

        impl $struct_name {
            pub const RW_SHIFT: $prim_ty = 0;
            pub const RP_SHIFT: $prim_ty = Self::RW_SHIFT + $rw_bits;
            pub const WW_SHIFT: $prim_ty = Self::RP_SHIFT + $rp_bits;
            pub const WP_SHIFT: $prim_ty = Self::WW_SHIFT + $ww_bits;
            pub const W_SHIFT: $prim_ty = Self::WP_SHIFT + $wp_bits;
            pub const R_SHIFT: $prim_ty = Self::W_SHIFT + $w_bits;

            const _ASSERT_SIZE: () = assert!(
                ($rw_bits + $rp_bits + $ww_bits + $wp_bits + $w_bits + $r_bits) <= <$prim_ty>::BITS as $prim_ty,
                "Total bits specified exceed the capacity of the chosen primitive type."
            );

            const fn mask(bits: $prim_ty, shift: $prim_ty) -> $prim_ty {
                if bits == 0 {
                    0
                } else if bits == <$prim_ty>::BITS as $prim_ty {
                    !0
                } else {
                    ((1 << bits) - 1) << shift
                }
            }

            pub const RW_MASK: $prim_ty = Self::mask($rw_bits, Self::RW_SHIFT);
            pub const RP_MASK: $prim_ty = Self::mask($rp_bits, Self::RP_SHIFT);
            pub const WW_MASK: $prim_ty = Self::mask($ww_bits, Self::WW_SHIFT);
            pub const WP_MASK: $prim_ty = Self::mask($wp_bits, Self::WP_SHIFT);
            pub const W_MASK: $prim_ty = Self::mask($w_bits, Self::W_SHIFT);
            pub const R_MASK: $prim_ty = Self::mask($r_bits, Self::R_SHIFT);

            pub const READERS_AND_WRITERS_MASK: $prim_ty = Self::R_MASK | Self::W_MASK;
            pub const ANY_WRITE_STATE_MASK: $prim_ty = Self::W_MASK | Self::WP_MASK | Self::WW_MASK;
            pub const WRITE_WAITERS_MASK: $prim_ty = Self::WP_MASK | Self::WW_MASK;
            pub const READ_WAITERS_MASK: $prim_ty = Self::RP_MASK | Self::RW_MASK;
            pub const ANY_WAITERS_MASK: $prim_ty = Self::WRITE_WAITERS_MASK | Self::READ_WAITERS_MASK;
        }

        impl LockState for $struct_name {
            type Atomic = $atomic_ty;
            type ReadWakers = $rw_ty;
            type ReadParked = $rp_ty;
            type WriteWakers = $ww_ty;
            type WriteParked = $wp_ty;
            type Writers = $w_ty;
            type Readers = $r_ty;
            type BatchSub = $prim_ty;

            #[allow(clippy::declare_interior_mutable_const)]
            const INITIAL_ATOMIC: Self::Atomic = <$atomic_ty>::new(0);

            #[inline(always)] fn empty() -> Self { Self(0) }
            #[inline(always)] fn with_writer() -> Self { Self(1 << Self::W_SHIFT) }

            #[inline(always)] fn readers(self) -> Self::Readers { ((self.0 & Self::R_MASK) >> Self::R_SHIFT) as Self::Readers }
            #[inline(always)] fn writers(self) -> Self::Writers { ((self.0 & Self::W_MASK) >> Self::W_SHIFT) as Self::Writers }
            #[inline(always)] fn write_parked(self) -> Self::WriteParked { ((self.0 & Self::WP_MASK) >> Self::WP_SHIFT) as Self::WriteParked }
            #[inline(always)] fn write_wakers(self) -> Self::WriteWakers { ((self.0 & Self::WW_MASK) >> Self::WW_SHIFT) as Self::WriteWakers }
            #[inline(always)] fn read_parked(self) -> Self::ReadParked { ((self.0 & Self::RP_MASK) >> Self::RP_SHIFT) as Self::ReadParked }
            #[inline(always)] fn read_wakers(self) -> Self::ReadWakers { ((self.0 & Self::RW_MASK) >> Self::RW_SHIFT) as Self::ReadWakers }

            #[inline(always)] fn has_readers_or_writers(self) -> bool { (self.0 & Self::READERS_AND_WRITERS_MASK) != 0 }
            #[inline(always)] fn has_any_write_state(self) -> bool { (self.0 & Self::ANY_WRITE_STATE_MASK) != 0 }
            #[inline(always)] fn has_write_waiters(self) -> bool { (self.0 & Self::WRITE_WAITERS_MASK) != 0 }
            #[inline(always)] fn has_read_waiters(self) -> bool { (self.0 & Self::READ_WAITERS_MASK) != 0 }
            #[inline(always)] fn has_any_waiters(self) -> bool { (self.0 & Self::ANY_WAITERS_MASK) != 0 }

            #[inline(always)] fn add_reader_state(self) -> Self { Self(self.0 + (1 << Self::R_SHIFT)) }
            #[inline(always)] fn sub_reader_state(self) -> Self { Self(self.0 - (1 << Self::R_SHIFT)) }
            #[inline(always)] fn add_writer_state(self) -> Self { Self(self.0 + (1 << Self::W_SHIFT)) }
            #[inline(always)] fn sub_writer_state(self) -> Self { Self(self.0 - (1 << Self::W_SHIFT)) }

            #[inline(always)] fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.load(order)) }
            #[inline(always)] fn atomic_compare_exchange_weak(
                atomic: &Self::Atomic,
                current: Self,
                new: Self,
                success: Ordering,
                failure: Ordering,
            ) -> Result<Self, Self> {
                atomic.compare_exchange_weak(current.0, new.0, success, failure)
                    .map(Self)
                    .map_err(Self)
            }

            #[inline(always)] fn atomic_add_read_waker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_add(1 << Self::RW_SHIFT, order)) }
            #[inline(always)] fn atomic_sub_read_waker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_sub(1 << Self::RW_SHIFT, order)) }
            #[inline(always)] fn atomic_add_read_parker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_add(1 << Self::RP_SHIFT, order)) }
            #[inline(always)] fn atomic_sub_read_parker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_sub(1 << Self::RP_SHIFT, order)) }
            #[inline(always)] fn atomic_add_write_waker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_add(1 << Self::WW_SHIFT, order)) }
            #[inline(always)] fn atomic_sub_write_waker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_sub(1 << Self::WW_SHIFT, order)) }
            #[inline(always)] fn atomic_add_write_parker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_add(1 << Self::WP_SHIFT, order)) }
            #[inline(always)] fn atomic_sub_write_parker(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_sub(1 << Self::WP_SHIFT, order)) }
            #[inline(always)] fn atomic_add_writer(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_add(1 << Self::W_SHIFT, order)) }
            #[inline(always)] fn atomic_sub_writer(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_sub(1 << Self::W_SHIFT, order)) }
            #[inline(always)] fn atomic_add_reader(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_add(1 << Self::R_SHIFT, order)) }
            #[inline(always)] fn atomic_sub_reader(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_sub(1 << Self::R_SHIFT, order)) }

            #[inline(always)] fn batch_sub_new() -> Self::BatchSub { 0 }
            #[inline(always)] fn batch_sub_read_waker(batch: Self::BatchSub, n: Self::ReadWakers) -> Self::BatchSub { batch + ((n as $prim_ty) << Self::RW_SHIFT) }
            #[inline(always)] fn batch_sub_write_waker(batch: Self::BatchSub, n: Self::WriteWakers) -> Self::BatchSub { batch + ((n as $prim_ty) << Self::WW_SHIFT) }
            #[inline(always)] fn atomic_fetch_sub_batch(atomic: &Self::Atomic, batch: Self::BatchSub, order: Ordering) {
                if batch != 0 {
                    atomic.fetch_sub(batch, order);
                }
            }
        }
    };
}

#[macro_export]
macro_rules! split_atomic_lock_state {
    (
        $vis:vis struct $struct_name:ident,
        $atomic_vis:vis struct $atomic_struct_name:ident {
            core: $core_atomic_ty:ident($core_prim_ty:ty) {
                writers: $w_ty:ty = $w_bits:expr,
                readers: $r_ty:ty = $r_bits:expr $(,)?
            },
            waiters: $wait_atomic_ty:ident($wait_prim_ty:ty) {
                read_wakers: $rw_ty:ty = $rw_bits:expr,
                read_parked: $rp_ty:ty = $rp_bits:expr,
                write_wakers: $ww_ty:ty = $ww_bits:expr,
                write_parked: $wp_ty:ty = $wp_bits:expr $(,)?
            }
        }
    ) => {
        #[derive(Debug)]
        $atomic_vis struct $atomic_struct_name {
            pub core: $core_atomic_ty,
            pub waiters: $wait_atomic_ty,
        }

        impl $atomic_struct_name {
            pub const fn new() -> Self {
                Self {
                    core: <$core_atomic_ty>::new(0),
                    waiters: <$wait_atomic_ty>::new(0),
                }
            }
        }

        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        $vis struct $struct_name {
            pub core: $core_prim_ty,
            pub waiters: $wait_prim_ty,
        }

        impl $struct_name {
            // --- Core Shifts & Masks ---
            pub const W_SHIFT: $core_prim_ty = 0;
            pub const R_SHIFT: $core_prim_ty = Self::W_SHIFT + $w_bits;

            const _ASSERT_CORE_SIZE: () = assert!(
                ($w_bits + $r_bits) <= <$core_prim_ty>::BITS as $core_prim_ty,
                "Total bits specified exceed the capacity of the core primitive type."
            );

            const fn mask_core(bits: $core_prim_ty, shift: $core_prim_ty) -> $core_prim_ty {
                if bits == 0 { 0 }
                else if bits == <$core_prim_ty>::BITS as $core_prim_ty { !0 }
                else { ((1 << bits) - 1) << shift }
            }

            pub const W_MASK: $core_prim_ty = Self::mask_core($w_bits, Self::W_SHIFT);
            pub const R_MASK: $core_prim_ty = Self::mask_core($r_bits, Self::R_SHIFT);
            pub const READERS_AND_WRITERS_MASK: $core_prim_ty = Self::R_MASK | Self::W_MASK;

            // --- Waiter Shifts & Masks ---
            pub const RW_SHIFT: $wait_prim_ty = 0;
            pub const RP_SHIFT: $wait_prim_ty = Self::RW_SHIFT + $rw_bits;
            pub const WW_SHIFT: $wait_prim_ty = Self::RP_SHIFT + $rp_bits;
            pub const WP_SHIFT: $wait_prim_ty = Self::WW_SHIFT + $ww_bits;

            const _ASSERT_WAIT_SIZE: () = assert!(
                ($rw_bits + $rp_bits + $ww_bits + $wp_bits) <= <$wait_prim_ty>::BITS as $wait_prim_ty,
                "Total bits specified exceed the capacity of the wait primitive type."
            );

            const fn mask_wait(bits: $wait_prim_ty, shift: $wait_prim_ty) -> $wait_prim_ty {
                if bits == 0 { 0 }
                else if bits == <$wait_prim_ty>::BITS as $wait_prim_ty { !0 }
                else { ((1 << bits) - 1) << shift }
            }

            pub const RW_MASK: $wait_prim_ty = Self::mask_wait($rw_bits, Self::RW_SHIFT);
            pub const RP_MASK: $wait_prim_ty = Self::mask_wait($rp_bits, Self::RP_SHIFT);
            pub const WW_MASK: $wait_prim_ty = Self::mask_wait($ww_bits, Self::WW_SHIFT);
            pub const WP_MASK: $wait_prim_ty = Self::mask_wait($wp_bits, Self::WP_SHIFT);

            pub const ANY_WRITE_WAITER_MASK: $wait_prim_ty = Self::WP_MASK | Self::WW_MASK;
            pub const ANY_READ_WAITER_MASK: $wait_prim_ty = Self::RP_MASK | Self::RW_MASK;
            pub const ANY_WAITERS_MASK: $wait_prim_ty = Self::ANY_WRITE_WAITER_MASK | Self::ANY_READ_WAITER_MASK;
        }

        impl LockState for $struct_name {
            type Atomic = $atomic_struct_name;
            type ReadWakers = $rw_ty;
            type ReadParked = $rp_ty;
            type WriteWakers = $ww_ty;
            type WriteParked = $wp_ty;
            type Writers = $w_ty;
            type Readers = $r_ty;
            type BatchSub = $wait_prim_ty;

            #[allow(clippy::declare_interior_mutable_const)]
            const INITIAL_ATOMIC: Self::Atomic = $atomic_struct_name::new();

            #[inline(always)] fn empty() -> Self { Self { core: 0, waiters: 0 } }
            #[inline(always)] fn with_writer() -> Self { Self { core: 1 << Self::W_SHIFT, waiters: 0 } }

            #[inline(always)] fn readers(self) -> Self::Readers { ((self.core & Self::R_MASK) >> Self::R_SHIFT) as Self::Readers }
            #[inline(always)] fn writers(self) -> Self::Writers { ((self.core & Self::W_MASK) >> Self::W_SHIFT) as Self::Writers }
            #[inline(always)] fn write_parked(self) -> Self::WriteParked { ((self.waiters & Self::WP_MASK) >> Self::WP_SHIFT) as Self::WriteParked }
            #[inline(always)] fn write_wakers(self) -> Self::WriteWakers { ((self.waiters & Self::WW_MASK) >> Self::WW_SHIFT) as Self::WriteWakers }
            #[inline(always)] fn read_parked(self) -> Self::ReadParked { ((self.waiters & Self::RP_MASK) >> Self::RP_SHIFT) as Self::ReadParked }
            #[inline(always)] fn read_wakers(self) -> Self::ReadWakers { ((self.waiters & Self::RW_MASK) >> Self::RW_SHIFT) as Self::ReadWakers }

            #[inline(always)] fn has_readers_or_writers(self) -> bool { (self.core & Self::READERS_AND_WRITERS_MASK) != 0 }
            #[inline(always)] fn has_any_write_state(self) -> bool {
                (self.core & Self::W_MASK) != 0 || (self.waiters & Self::ANY_WRITE_WAITER_MASK) != 0
            }
            #[inline(always)] fn has_write_waiters(self) -> bool { (self.waiters & Self::ANY_WRITE_WAITER_MASK) != 0 }
            #[inline(always)] fn has_read_waiters(self) -> bool { (self.waiters & Self::ANY_READ_WAITER_MASK) != 0 }
            #[inline(always)] fn has_any_waiters(self) -> bool { (self.waiters & Self::ANY_WAITERS_MASK) != 0 }

            #[inline(always)] fn add_reader_state(self) -> Self { Self { core: self.core + (1 << Self::R_SHIFT), waiters: self.waiters } }
            #[inline(always)] fn sub_reader_state(self) -> Self { Self { core: self.core - (1 << Self::R_SHIFT), waiters: self.waiters } }
            #[inline(always)] fn add_writer_state(self) -> Self { Self { core: self.core + (1 << Self::W_SHIFT), waiters: self.waiters } }
            #[inline(always)] fn sub_writer_state(self) -> Self { Self { core: self.core - (1 << Self::W_SHIFT), waiters: self.waiters } }

            #[inline(always)] fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self {
                    core: atomic.core.load(order),
                    waiters: atomic.waiters.load(order),
                }
            }

            #[inline(always)] fn atomic_compare_exchange_weak(
                atomic: &Self::Atomic,
                current: Self,
                new: Self,
                success: Ordering,
                failure: Ordering,
            ) -> Result<Self, Self> {
                // By design, CAS only targets the core readers/writers lock state. Waiters are
                // completely ignored in the CAS comparison to prevent false-sharing failures.
                match atomic.core.compare_exchange_weak(current.core, new.core, success, failure) {
                    Ok(_) => Ok(new),
                    Err(actual_core) => Err(Self {
                        core: actual_core,
                        waiters: atomic.waiters.load(Ordering::Relaxed),
                    }),
                }
            }

            #[inline(always)] fn atomic_add_read_waker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_add(1 << Self::RW_SHIFT, order) }
            }
            #[inline(always)] fn atomic_sub_read_waker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_sub(1 << Self::RW_SHIFT, order) }
            }
            #[inline(always)] fn atomic_add_read_parker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_add(1 << Self::RP_SHIFT, order) }
            }
            #[inline(always)] fn atomic_sub_read_parker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_sub(1 << Self::RP_SHIFT, order) }
            }
            #[inline(always)] fn atomic_add_write_waker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_add(1 << Self::WW_SHIFT, order) }
            }
            #[inline(always)] fn atomic_sub_write_waker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_sub(1 << Self::WW_SHIFT, order) }
            }
            #[inline(always)] fn atomic_add_write_parker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_add(1 << Self::WP_SHIFT, order) }
            }
            #[inline(always)] fn atomic_sub_write_parker(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.load(Ordering::Relaxed), waiters: atomic.waiters.fetch_sub(1 << Self::WP_SHIFT, order) }
            }

            #[inline(always)] fn atomic_add_writer(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.fetch_add(1 << Self::W_SHIFT, order), waiters: atomic.waiters.load(Ordering::Relaxed) }
            }

            #[inline(always)] fn atomic_sub_writer(atomic: &Self::Atomic, order: Ordering) -> Self {
                let old_core = atomic.core.fetch_sub(1 << Self::W_SHIFT, order);
                // Required to prevent a releasing thread from missing a waiter that just parked
                core::sync::atomic::fence(Ordering::SeqCst);
                Self { core: old_core, waiters: atomic.waiters.load(Ordering::Relaxed) }
            }

            #[inline(always)] fn atomic_add_reader(atomic: &Self::Atomic, order: Ordering) -> Self {
                Self { core: atomic.core.fetch_add(1 << Self::R_SHIFT, order), waiters: atomic.waiters.load(Ordering::Relaxed) }
            }

            #[inline(always)] fn atomic_sub_reader(atomic: &Self::Atomic, order: Ordering) -> Self {
                let old_core = atomic.core.fetch_sub(1 << Self::R_SHIFT, order);
                // Required to prevent a releasing thread from missing a waiter that just parked
                core::sync::atomic::fence(Ordering::SeqCst);
                Self { core: old_core, waiters: atomic.waiters.load(Ordering::Relaxed) }
            }

            #[inline(always)] fn batch_sub_new() -> Self::BatchSub { 0 }
            #[inline(always)] fn batch_sub_read_waker(batch: Self::BatchSub, n: Self::ReadWakers) -> Self::BatchSub {
                batch + ((n as $wait_prim_ty) << Self::RW_SHIFT)
            }
            #[inline(always)] fn batch_sub_write_waker(batch: Self::BatchSub, n: Self::WriteWakers) -> Self::BatchSub {
                batch + ((n as $wait_prim_ty) << Self::WW_SHIFT)
            }
            #[inline(always)] fn atomic_fetch_sub_batch(atomic: &Self::Atomic, batch: Self::BatchSub, order: Ordering) {
                if batch != 0 {
                    atomic.waiters.fetch_sub(batch, order);
                }
            }
        }
    };
}

atomic_lock_state!(pub struct LockStateU64(
    AtomicU64(u64) {
        read_wakers: u16 = 16,
        read_parked: u8 = 9,
        write_wakers: u8 = 9,
        write_parked: u8 = 9,
        writers: u8 = 1,
        readers: u16 = 20,
    }
));

atomic_lock_state!(pub struct LockStateU32(
    AtomicU32(u32) {
        read_wakers: u8 = 8,
        read_parked: u8 = 5,
        write_wakers: u8 = 5,
        write_parked: u8 = 5,
        writers: u8 = 1,
        readers: u8 = 8,
    }
));

atomic_lock_state!(pub struct LockStateU16(
    AtomicU16(u16) {
        read_wakers: u8 = 4,
        read_parked: u8 = 2,
        write_wakers: u8 = 2,
        write_parked: u8 = 2,
        writers: u8 = 1,
        readers: u8 = 5,
    }
));

split_atomic_lock_state!(
    pub struct SplitLockState32,
    pub struct SplitLockAtomics32 {
        core: AtomicU32(u32) {
            writers: u16 = 1,
            readers: u16 = 31,
        },
        waiters: AtomicU32(u32) {
            read_wakers: u8 = 8,
            read_parked: u8 = 8,
            write_wakers: u8 = 8,
            write_parked: u8 = 8,
        }
    }
);
