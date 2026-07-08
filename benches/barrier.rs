use std::hint::black_box;
use std::sync::{Arc, Barrier as StdBarrier};

use async_std::sync::Barrier as AsBarrier;
use criterion::{Criterion, criterion_group, criterion_main};
use tokio::sync::Barrier as TkBarrier;
use z_sync::{
    Barrier16 as ZBarrier16, Barrier32 as ZBarrier32, Barrier32Boxed as ZBarrier32Boxed,
    Barrier64 as ZBarrier64, Barrier64Inline as ZBarrier64Inline,
};

const WORKERS: usize = 8;
const ROUNDS: usize = 200;

// ============================================================================
// 1. Uncontended (n = 1): the arrive fast path — CAS the count to full, then a notify with no
//    registered waiters. Returns immediately, no parking.
// ============================================================================
fn bench_uncontended(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_Uncontended");

    // Blocking
    group.bench_function("z_sync::Barrier16", |b| {
        let barrier = ZBarrier16::new(1);
        b.iter(|| black_box(barrier.wait()));
    });
    group.bench_function("z_sync::Barrier32", |b| {
        let barrier = ZBarrier32::new(1);
        b.iter(|| black_box(barrier.wait()));
    });
    group.bench_function("z_sync::Barrier64", |b| {
        let barrier = ZBarrier64::new(1);
        b.iter(|| black_box(barrier.wait()));
    });
    group.bench_function("std::sync::Barrier", |b| {
        let barrier = StdBarrier::new(1);
        b.iter(|| black_box(barrier.wait()));
    });

    // Async
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("z_sync::Barrier32 (Async)", |b| {
        let barrier = ZBarrier32::new(1);
        b.to_async(&rt).iter(|| async { black_box(barrier.wait_async().await) });
    });
    group.bench_function("z_sync::Barrier64 (Async)", |b| {
        let barrier = ZBarrier64::new(1);
        b.to_async(&rt).iter(|| async { black_box(barrier.wait_async().await) });
    });
    group.bench_function("tokio::sync::Barrier", |b| {
        let barrier = TkBarrier::new(1);
        b.to_async(&rt).iter(|| async { black_box(barrier.wait().await) });
    });
    group.bench_function("async-std::Barrier", |b| {
        let barrier = AsBarrier::new(1);
        b.to_async(&rt).iter(|| async { black_box(barrier.wait().await) });
    });

    group.finish();
}

// ============================================================================
// 2. Contended rendezvous: WORKERS parties each hit the barrier ROUNDS times, so the whole group
//    synchronises ROUNDS times. std::sync::Barrier is blocking-only; tokio::sync::Barrier is
//    async-only.
// ============================================================================
fn bench_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Contended");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Blocking
    group.bench_function("z_sync::Barrier32 (Blocking)", |b| {
        b.iter(|| {
            let barrier = Arc::new(ZBarrier32::new(WORKERS));
            std::thread::scope(|scope| {
                for _ in 0..WORKERS {
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        for _ in 0..ROUNDS {
                            black_box(barrier.wait());
                        }
                    });
                }
            });
        });
    });
    group.bench_function("z_sync::Barrier64 (Blocking)", |b| {
        b.iter(|| {
            let barrier = Arc::new(ZBarrier64::new(WORKERS));
            std::thread::scope(|scope| {
                for _ in 0..WORKERS {
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        for _ in 0..ROUNDS {
                            black_box(barrier.wait());
                        }
                    });
                }
            });
        });
    });
    group.bench_function("std::sync::Barrier (Blocking)", |b| {
        b.iter(|| {
            let barrier = Arc::new(StdBarrier::new(WORKERS));
            std::thread::scope(|scope| {
                for _ in 0..WORKERS {
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        for _ in 0..ROUNDS {
                            black_box(barrier.wait());
                        }
                    });
                }
            });
        });
    });

    // Async
    group.bench_function("z_sync::Barrier32 (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let barrier = Arc::new(ZBarrier32::new(WORKERS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let barrier = Arc::clone(&barrier);
                handles.push(tokio::spawn(async move {
                    for _ in 0..ROUNDS {
                        black_box(barrier.wait_async().await);
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
    group.bench_function("z_sync::Barrier32Boxed (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let barrier = Arc::new(ZBarrier32Boxed::new(WORKERS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let barrier = Arc::clone(&barrier);
                handles.push(tokio::spawn(async move {
                    for _ in 0..ROUNDS {
                        black_box(barrier.wait_async().await);
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
    group.bench_function("z_sync::Barrier64Inline (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let barrier = Arc::new(ZBarrier64Inline::new(WORKERS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let barrier = Arc::clone(&barrier);
                handles.push(tokio::spawn(async move {
                    for _ in 0..ROUNDS {
                        black_box(barrier.wait_async().await);
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
    group.bench_function("tokio::sync::Barrier (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let barrier = Arc::new(TkBarrier::new(WORKERS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let barrier = Arc::clone(&barrier);
                handles.push(tokio::spawn(async move {
                    for _ in 0..ROUNDS {
                        black_box(barrier.wait().await);
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
    group.bench_function("async-std::Barrier (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let barrier = Arc::new(AsBarrier::new(WORKERS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let barrier = Arc::clone(&barrier);
                handles.push(tokio::spawn(async move {
                    for _ in 0..ROUNDS {
                        black_box(barrier.wait().await);
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_uncontended, bench_contended);
criterion_main!(benches);
