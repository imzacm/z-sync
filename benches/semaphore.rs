use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use tokio::sync::Semaphore as TkSemaphore;
use z_sync::{Semaphore16 as ZSem16, Semaphore32 as ZSem32, Semaphore64 as ZSem64};

const WORKERS: usize = 8;
const OPS_PER_WORKER: usize = 1000;
const PERMITS: usize = 4;

// ============================================================================
// 1. Uncontended Workload (Fast Path: acquire -> drop)
// ============================================================================
fn bench_uncontended(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_Uncontended");

    // Blocking
    group.bench_function("z_sync::Semaphore16", |b| {
        let s = ZSem16::new(1);
        b.iter(|| black_box(s.acquire()));
    });
    group.bench_function("z_sync::Semaphore32", |b| {
        let s = ZSem32::new(1);
        b.iter(|| black_box(s.acquire()));
    });
    group.bench_function("z_sync::Semaphore64", |b| {
        let s = ZSem64::new(1);
        b.iter(|| black_box(s.acquire()));
    });

    // Async
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("z_sync::Semaphore16 (Async)", |b| {
        let s = ZSem16::new(1);
        b.to_async(&rt).iter(|| async { black_box(s.acquire_async().await) });
    });
    group.bench_function("z_sync::Semaphore32 (Async)", |b| {
        let s = ZSem32::new(1);
        b.to_async(&rt).iter(|| async { black_box(s.acquire_async().await) });
    });
    group.bench_function("z_sync::Semaphore64 (Async)", |b| {
        let s = ZSem64::new(1);
        b.to_async(&rt).iter(|| async { black_box(s.acquire_async().await) });
    });
    group.bench_function("tokio::sync::Semaphore", |b| {
        let s = TkSemaphore::new(1);
        b.to_async(&rt).iter(|| async { black_box(s.acquire().await.unwrap()) });
    });

    group.finish();
}

// ============================================================================
// 2. Uncontended try_acquire (Non-blocking Fast Path)
// ============================================================================
fn bench_try_acquire(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Try_Acquire");

    group.bench_function("z_sync::Semaphore32", |b| {
        let s = ZSem32::new(1);
        b.iter(|| black_box(s.try_acquire()));
    });
    group.bench_function("z_sync::Semaphore64", |b| {
        let s = ZSem64::new(1);
        b.iter(|| black_box(s.try_acquire()));
    });
    group.bench_function("tokio::sync::Semaphore", |b| {
        let s = TkSemaphore::new(1);
        b.iter(|| black_box(s.try_acquire().ok()));
    });

    group.finish();
}

// ============================================================================
// 3. Contended Workload (8 workers, bounded permits)
// ============================================================================
fn bench_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_Contended");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Blocking
    group.bench_function("z_sync::Semaphore32 (Blocking)", |b| {
        b.iter(|| {
            let s = Arc::new(ZSem32::new(PERMITS));
            std::thread::scope(|scope| {
                for _ in 0..WORKERS {
                    let s = Arc::clone(&s);
                    scope.spawn(move || {
                        for _ in 0..OPS_PER_WORKER {
                            black_box(s.acquire());
                        }
                    });
                }
            });
        });
    });
    group.bench_function("z_sync::Semaphore64 (Blocking)", |b| {
        b.iter(|| {
            let s = Arc::new(ZSem64::new(PERMITS));
            std::thread::scope(|scope| {
                for _ in 0..WORKERS {
                    let s = Arc::clone(&s);
                    scope.spawn(move || {
                        for _ in 0..OPS_PER_WORKER {
                            black_box(s.acquire());
                        }
                    });
                }
            });
        });
    });

    // Async
    group.bench_function("z_sync::Semaphore32 (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let s = Arc::new(ZSem32::new(PERMITS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let s = Arc::clone(&s);
                handles.push(tokio::spawn(async move {
                    for _ in 0..OPS_PER_WORKER {
                        black_box(s.acquire_async().await);
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
    group.bench_function("z_sync::Semaphore64 (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let s = Arc::new(ZSem64::new(PERMITS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let s = Arc::clone(&s);
                handles.push(tokio::spawn(async move {
                    for _ in 0..OPS_PER_WORKER {
                        black_box(s.acquire_async().await);
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
    group.bench_function("tokio::sync::Semaphore", |b| {
        b.to_async(&rt).iter(|| async {
            let s = Arc::new(TkSemaphore::new(PERMITS));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let s = Arc::clone(&s);
                handles.push(tokio::spawn(async move {
                    for _ in 0..OPS_PER_WORKER {
                        let _ = black_box(s.acquire().await.unwrap());
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

criterion_group!(benches, bench_uncontended, bench_try_acquire, bench_contended);
criterion_main!(benches);
