use core::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};

use num_traits::{ConstOne, ConstZero, NumCast};

pub trait NotifyState: Sized + Copy + Clone + PartialEq + Eq {
    type Atomic: core::fmt::Debug;

    type Wakers: Eq + Ord + NumCast + ConstZero + ConstOne;
    type Parked: Eq + Ord + NumCast + ConstZero + ConstOne;
    type Epoch: core::fmt::Debug + Eq + Ord + NumCast;

    const INITIAL_ATOMIC: Self::Atomic;

    // --- State Getters ---
    fn epoch(self) -> Self::Epoch;
    fn wakers(self) -> Self::Wakers;
    fn parked(self) -> Self::Parked;
    fn has_listeners(self) -> bool;

    // --- Atomic Operations ---
    fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self;
    fn atomic_inc_epoch(atomic: &Self::Atomic, order: Ordering) -> Self;

    fn atomic_add_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering);
    fn atomic_sub_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering);
    fn atomic_add_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering);
    fn atomic_sub_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering);
}

#[macro_export]
macro_rules! atomic_notify_state {
    (
        $vis:vis struct $struct_name:ident(
            $atomic_ty:ident($prim_ty:ty) {
                wakers: $w_ty:ty = $w_bits:expr,
                parked: $p_ty:ty = $p_bits:expr,
                epoch: $e_ty:ty = $e_bits:expr $(,)?
            }
        )
    ) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        $vis struct $struct_name(pub $prim_ty);

        impl $struct_name {
            pub const WAKER_SHIFT: $prim_ty = 0;
            pub const PARKER_SHIFT: $prim_ty = Self::WAKER_SHIFT + $w_bits;
            pub const EPOCH_SHIFT: $prim_ty = Self::PARKER_SHIFT + $p_bits;

            const _ASSERT_SIZE: () = assert!(
                ($w_bits + $p_bits + $e_bits) <= <$prim_ty>::BITS as $prim_ty,
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

            pub const WAKERS_MASK: $prim_ty = Self::mask($w_bits, Self::WAKER_SHIFT);
            pub const PARKED_MASK: $prim_ty = Self::mask($p_bits, Self::PARKER_SHIFT);
            pub const EPOCH_MASK: $prim_ty = Self::mask($e_bits, Self::EPOCH_SHIFT);
            pub const LISTENERS_MASK: $prim_ty = Self::WAKERS_MASK | Self::PARKED_MASK;
        }

        impl NotifyState for $struct_name {
            type Atomic = $atomic_ty;
            type Wakers = $w_ty;
            type Parked = $p_ty;
            type Epoch = $e_ty;

            #[allow(clippy::declare_interior_mutable_const)]
            const INITIAL_ATOMIC: Self::Atomic = <$atomic_ty>::new(0);

            #[inline(always)] fn epoch(self) -> Self::Epoch { ((self.0 & Self::EPOCH_MASK) >> Self::EPOCH_SHIFT) as Self::Epoch }
            #[inline(always)] fn wakers(self) -> Self::Wakers { ((self.0 & Self::WAKERS_MASK) >> Self::WAKER_SHIFT) as Self::Wakers }
            #[inline(always)] fn parked(self) -> Self::Parked { ((self.0 & Self::PARKED_MASK) >> Self::PARKER_SHIFT) as Self::Parked }
            #[inline(always)] fn has_listeners(self) -> bool { (self.0 & Self::LISTENERS_MASK) != 0 }

            #[inline(always)] fn atomic_load(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.load(order)) }
            #[inline(always)] fn atomic_inc_epoch(atomic: &Self::Atomic, order: Ordering) -> Self { Self(atomic.fetch_add(1 << Self::EPOCH_SHIFT, order)) }

            #[inline(always)] fn atomic_add_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering) { atomic.fetch_add((n as $prim_ty) << Self::PARKER_SHIFT, order); }
            #[inline(always)] fn atomic_sub_parkers(atomic: &Self::Atomic, n: Self::Parked, order: Ordering) { atomic.fetch_sub((n as $prim_ty) << Self::PARKER_SHIFT, order); }
            #[inline(always)] fn atomic_add_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering) { atomic.fetch_add((n as $prim_ty) << Self::WAKER_SHIFT, order); }
            #[inline(always)] fn atomic_sub_wakers(atomic: &Self::Atomic, n: Self::Wakers, order: Ordering) { atomic.fetch_sub((n as $prim_ty) << Self::WAKER_SHIFT, order); }
        }
    };
}

atomic_notify_state!(pub struct NotifyStateU64(
    AtomicU64(u64) {
        wakers: u16 = 16,
        parked: u16 = 16,
        epoch: u32 = 32,
    }
));

atomic_notify_state!(pub struct NotifyStateU32(
    AtomicU32(u32) {
        wakers: u8 = 8,
        parked: u8 = 8,
        epoch: u16 = 16,
    }
));

atomic_notify_state!(pub struct NotifyStateU16(
    AtomicU16(u16) {
        wakers: u8 = 4,
        parked: u8 = 4,
        epoch: u16 = 8,
    }
));
