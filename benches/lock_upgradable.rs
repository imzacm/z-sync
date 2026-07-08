use std::hint::black_box;
use std::sync::Arc;

use async_std::sync::{RwLock as AsRwLock, RwLockUpgradableReadGuard as AsUpgradable};
use criterion::{Criterion, criterion_group, criterion_main};
use parking_lot::{RwLock as PlRwLock, RwLockUpgradableReadGuard as PlUpgradable};
use z_sync::lock::{ReadGuard as ZReadGuard, WriteGuard as ZWriteGuard};
use z_sync::{Lock32 as ZLock32, Lock64 as ZLock64};

const WORKERS: usize = 8;
const OPS_PER_WORKER: usize = 500;

struct Payload(u64);

// ============================================================================
// 1. Uncontended acquire/release of an upgradable read lock (fast path).
// ============================================================================
fn bench_uncontended(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_Uncontended_Acquire");

    group.bench_function("z_sync::Lock64", |b| {
        let lock = ZLock64::new(Payload(0));
        b.iter(|| black_box(lock.upgradable_read().0));
    });
    group.bench_function("parking_lot", |b| {
        let lock = PlRwLock::new(Payload(0));
        b.iter(|| black_box(lock.upgradable_read().0));
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("z_sync::Lock64 (Async)", |b| {
        let lock = ZLock64::new(Payload(0));
        b.to_async(&rt)
            .iter(|| async { black_box(lock.upgradable_read_async().await.0) });
    });
    group.bench_function("async-std (Async)", |b| {
        let lock = AsRwLock::new(Payload(0));
        b.to_async(&rt).iter(|| async { black_box(lock.upgradable_read().await.0) });
    });

    group.finish();
}

// ============================================================================
// 2. Uncontended read-modify-write via upgrade: acquire upgradable, read, upgrade to write, mutate.
//    The pattern upgradable reads exist for.
// ============================================================================
fn bench_upgrade(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Upgrade_RMW");

    group.bench_function("z_sync::Lock64", |b| {
        let lock = ZLock64::new(Payload(0));
        b.iter(|| {
            let up = lock.upgradable_read();
            let mut w = up.upgrade();
            w.0 += 1;
            black_box(w.0)
        });
    });
    group.bench_function("parking_lot", |b| {
        let lock = PlRwLock::new(Payload(0));
        b.iter(|| {
            let up = lock.upgradable_read();
            let mut w = PlUpgradable::upgrade(up);
            w.0 += 1;
            black_box(w.0)
        });
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("z_sync::Lock64 (Async)", |b| {
        let lock = ZLock64::new(Payload(0));
        b.to_async(&rt).iter(|| async {
            let up = lock.upgradable_read_async().await;
            let mut w = up.upgrade_async().await;
            w.0 += 1;
            black_box(w.0)
        });
    });
    group.bench_function("async-std (Async)", |b| {
        let lock = AsRwLock::new(Payload(0));
        b.to_async(&rt).iter(|| async {
            let up = lock.upgradable_read().await;
            let mut w = AsUpgradable::upgrade(up).await;
            w.0 += 1;
            black_box(w.0)
        });
    });

    group.finish();
}

// ============================================================================
// 3. Contended read-modify-write: WORKERS threads/tasks each acquire upgradable, read, upgrade, and
//    increment. Exercises the upgrader mutual-exclusion + reader-drain path under contention.
// ============================================================================
fn bench_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_Contended_Upgrade");

    group.bench_function("z_sync::Lock64 (Blocking)", |b| {
        b.iter(|| {
            let lock = Arc::new(ZLock64::new(Payload(0)));
            std::thread::scope(|s| {
                for _ in 0..WORKERS {
                    let lock = Arc::clone(&lock);
                    s.spawn(move || {
                        for _ in 0..OPS_PER_WORKER {
                            let up = lock.upgradable_read();
                            let seen = up.0;
                            let mut w = up.upgrade();
                            w.0 = seen + 1;
                        }
                    });
                }
            });
        });
    });
    group.bench_function("parking_lot (Blocking)", |b| {
        b.iter(|| {
            let lock = Arc::new(PlRwLock::new(Payload(0)));
            std::thread::scope(|s| {
                for _ in 0..WORKERS {
                    let lock = Arc::clone(&lock);
                    s.spawn(move || {
                        for _ in 0..OPS_PER_WORKER {
                            let up = lock.upgradable_read();
                            let seen = up.0;
                            let mut w = PlUpgradable::upgrade(up);
                            w.0 = seen + 1;
                        }
                    });
                }
            });
        });
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("z_sync::Lock64 (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let lock = Arc::new(ZLock64::new(Payload(0)));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let lock = Arc::clone(&lock);
                handles.push(tokio::spawn(async move {
                    for _ in 0..OPS_PER_WORKER {
                        let up = lock.upgradable_read_async().await;
                        let seen = up.0;
                        let mut w = up.upgrade_async().await;
                        w.0 = seen + 1;
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
    group.bench_function("async-std (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let lock = Arc::new(AsRwLock::new(Payload(0)));
            let mut handles = Vec::with_capacity(WORKERS);
            for _ in 0..WORKERS {
                let lock = Arc::clone(&lock);
                handles.push(tokio::spawn(async move {
                    for _ in 0..OPS_PER_WORKER {
                        let up = lock.upgradable_read().await;
                        let seen = up.0;
                        let mut w = AsUpgradable::upgrade(up).await;
                        w.0 = seen + 1;
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

// ============================================================================
// 4. Mapped guards: project to a field, then read/write through the mapping.
// ============================================================================
fn bench_mapped(c: &mut Criterion) {
    let mut group = c.benchmark_group("4_Mapped_Guards");

    group.bench_function("z_sync::Lock64 read-map", |b| {
        let lock = ZLock64::new((1u64, 2u64));
        b.iter(|| {
            let g = ZReadGuard::map(lock.read(), |t| &t.1);
            black_box(*g)
        });
    });
    group.bench_function("parking_lot read-map", |b| {
        let lock = PlRwLock::new((1u64, 2u64));
        b.iter(|| {
            let g = parking_lot::RwLockReadGuard::map(lock.read(), |t| &t.1);
            black_box(*g)
        });
    });
    group.bench_function("z_sync::Lock64 write-map", |b| {
        let lock = ZLock64::new((1u64, 2u64));
        b.iter(|| {
            let mut g = ZWriteGuard::map(lock.write(), |t| &mut t.1);
            *g += 1;
            black_box(*g)
        });
    });
    group.bench_function("parking_lot write-map", |b| {
        let lock = PlRwLock::new((1u64, 2u64));
        b.iter(|| {
            let mut g = parking_lot::RwLockWriteGuard::map(lock.write(), |t| &mut t.1);
            *g += 1;
            black_box(*g)
        });
    });

    // Silence unused-import warnings when only some widths are exercised.
    let _ = ZLock32::new(Payload(0));

    group.finish();
}

criterion_group!(benches, bench_uncontended, bench_upgrade, bench_contended, bench_mapped);
criterion_main!(benches);
