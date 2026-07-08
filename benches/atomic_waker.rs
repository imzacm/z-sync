use std::hint::black_box;

use atomic_waker::AtomicWaker as CrateAtomicWaker;
use criterion::{Criterion, criterion_group, criterion_main};
use futures::task::{AtomicWaker as FuturesAtomicWaker, noop_waker};
use z_sync::AtomicWaker as ZAtomicWaker;
use z_sync::waker_storage::{BoxedWakers, InlineWakers, WakerStorage};

// The single-waiter queue: one array slot, no spill. This is the fairest apples-to-apples for what
// AtomicWaker replaces (one registered waker at a time).
const CAP: usize = 1;

// ============================================================================
// 1. Register a waker, then wake it — the full single-consumer cycle a one-shot channel performs
//    (receiver registers in poll, sender delivers).
// ============================================================================
fn bench_register_wake(c: &mut Criterion) {
    let mut group = c.benchmark_group("register_wake");
    let waker = noop_waker();

    group.bench_function("z_sync::AtomicWaker", |b| {
        let cell = ZAtomicWaker::new();
        b.iter(|| {
            cell.register(black_box(&waker));
            cell.wake();
        });
    });
    group.bench_function("futures::AtomicWaker", |b| {
        let cell = FuturesAtomicWaker::new();
        b.iter(|| {
            cell.register(black_box(&waker));
            cell.wake();
        });
    });
    group.bench_function("atomic-waker crate", |b| {
        let cell = CrateAtomicWaker::new();
        b.iter(|| {
            cell.register(black_box(&waker));
            cell.wake();
        });
    });

    group.bench_function("InlineWakers queue", |b| {
        let storage = <InlineWakers<CAP> as WakerStorage<CAP>>::INIT;
        b.iter(|| {
            black_box(storage.queue().lock().push(black_box(waker.clone())));
            if let Some(w) = storage.queue().lock().pop_and_take() {
                w.wake();
            }
        });
    });

    group.bench_function("BoxedWakers queue", |b| {
        let storage = <BoxedWakers<CAP> as WakerStorage<CAP>>::INIT;
        // Force the lazy allocation so we measure steady state, not the one-off first-use alloc.
        let _ = storage.queue();
        b.iter(|| {
            black_box(storage.queue().lock().push(black_box(waker.clone())));
            if let Some(w) = storage.queue().lock().pop_and_take() {
                w.wake();
            }
        });
    });

    group.finish();
}

// ============================================================================
// 2. Wake with nothing registered — the common "sender fires before the receiver ever waits" path.
// ============================================================================
fn bench_wake_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("wake_empty");

    group.bench_function("z_sync::AtomicWaker", |b| {
        let cell = ZAtomicWaker::new();
        b.iter(|| cell.wake());
    });
    group.bench_function("futures::AtomicWaker", |b| {
        let cell = FuturesAtomicWaker::new();
        b.iter(|| cell.wake());
    });
    group.bench_function("atomic-waker crate", |b| {
        let cell = CrateAtomicWaker::new();
        b.iter(|| cell.wake());
    });

    group.bench_function("InlineWakers queue", |b| {
        let storage = <InlineWakers<CAP> as WakerStorage<CAP>>::INIT;
        b.iter(|| black_box(storage.queue().lock().pop_and_take()));
    });

    group.bench_function("BoxedWakers queue", |b| {
        let storage = <BoxedWakers<CAP> as WakerStorage<CAP>>::INIT;
        let _ = storage.queue();
        b.iter(|| black_box(storage.queue().lock().pop_and_take()));
    });

    group.finish();
}

criterion_group!(benches, bench_register_wake, bench_wake_empty);
criterion_main!(benches);
