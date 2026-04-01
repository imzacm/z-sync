use std::hint::black_box;
use std::sync::{Arc, Mutex as StdMutex, RwLock as StdRwLock};

use criterion::{Criterion, criterion_group, criterion_main};
use parking_lot::{Mutex as PlMutex, RwLock as PlRwLock};
use tokio::sync::{Mutex as TkMutex, RwLock as TkRwLock};
use z_sync::{Lock16 as ZLock16, Lock32 as ZLock32, Lock64 as ZLock64};

const WORKERS: usize = 8;
const OPS_PER_WORKER: usize = 1000;

struct Payload(u64);

// ============================================================================
// 1. Uncontended Workload (Fast Path)
// ============================================================================
fn bench_uncontended(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_Uncontended");

    // Mutexes
    group.bench_function("std::Mutex", |b| {
        let m = StdMutex::new(Payload(0));
        b.iter(|| black_box(m.lock().unwrap().0 += 1));
    });
    group.bench_function("parking_lot::Mutex", |b| {
        let m = PlMutex::new(Payload(0));
        b.iter(|| black_box(m.lock().0 += 1));
    });

    // RwLocks (Write)
    group.bench_function("std::RwLock", |b| {
        let m = StdRwLock::new(Payload(0));
        b.iter(|| black_box(m.write().unwrap().0 += 1));
    });
    group.bench_function("parking_lot::RwLock", |b| {
        let m = PlRwLock::new(Payload(0));
        b.iter(|| black_box(m.write().0 += 1));
    });
    group.bench_function("z_sync::Lock16", |b| {
        let m = ZLock16::new(Payload(0));
        b.iter(|| black_box(m.write().0 += 1));
    });
    group.bench_function("z_sync::Lock32", |b| {
        let m = ZLock32::new(Payload(0));
        b.iter(|| black_box(m.write().0 += 1));
    });
    group.bench_function("z_sync::Lock64", |b| {
        let m = ZLock64::new(Payload(0));
        b.iter(|| black_box(m.write().0 += 1));
    });

    // Async (Uncontended)
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("tokio::Mutex", |b| {
        let m = TkMutex::new(Payload(0));
        b.to_async(&rt).iter(|| async { black_box(m.lock().await.0 += 1) });
    });
    group.bench_function("tokio::RwLock", |b| {
        let m = TkRwLock::new(Payload(0));
        b.to_async(&rt).iter(|| async { black_box(m.write().await.0 += 1) });
    });
    group.bench_function("z_sync::Lock16 (Async)", |b| {
        let m = ZLock16::new(Payload(0));
        b.to_async(&rt).iter(|| async { black_box(m.write_async().await.0 += 1) });
    });
    group.bench_function("z_sync::Lock32 (Async)", |b| {
        let m = ZLock32::new(Payload(0));
        b.to_async(&rt).iter(|| async { black_box(m.write_async().await.0 += 1) });
    });
    group.bench_function("z_sync::Lock64 (Async)", |b| {
        let m = ZLock64::new(Payload(0));
        b.to_async(&rt).iter(|| async { black_box(m.write_async().await.0 += 1) });
    });

    group.finish();
}

// ============================================================================
// Helper Macros for Contended Workloads
// ============================================================================

macro_rules! bench_blocking {
    ($group:expr, $name:expr, $lock_init:expr, |$i:ident, $m:ident| $worker_logic:expr) => {
        $group.bench_function($name, |b| {
            b.iter(|| {
                let lock = Arc::new($lock_init);
                std::thread::scope(|s| {
                    for _ in 0..WORKERS {
                        let $m = Arc::clone(&lock);
                        s.spawn(move || {
                            for $i in 0..OPS_PER_WORKER {
                                $worker_logic;
                            }
                        });
                    }
                });
            });
        });
    };
}

macro_rules! bench_async {
    ($group:expr, $name:expr, $rt:expr, $lock_init:expr, |$i:ident, $m:ident| $worker_logic:expr) => {
        $group.bench_function($name, |b| {
            b.to_async($rt).iter(|| async {
                let lock = Arc::new($lock_init);
                let mut handles = Vec::with_capacity(WORKERS);
                for _ in 0..WORKERS {
                    let $m = Arc::clone(&lock);
                    handles.push(tokio::spawn(async move {
                        for $i in 0..OPS_PER_WORKER {
                            $worker_logic;
                        }
                    }));
                }
                for h in handles {
                    h.await.unwrap();
                }
            });
        });
    };
}

// ============================================================================
// 2. Read-Only Workload
// ============================================================================
fn bench_read_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Read_Only");
    let rt = tokio::runtime::Runtime::new().unwrap();

    bench_blocking!(group, "std::Mutex", StdMutex::new(Payload(0)), |_i, m| black_box(
        m.lock().unwrap().0
    ));
    bench_blocking!(group, "std::RwLock", StdRwLock::new(Payload(0)), |_i, m| black_box(
        m.read().unwrap().0
    ));
    bench_blocking!(group, "parking_lot::Mutex", PlMutex::new(Payload(0)), |_i, m| black_box(
        m.lock().0
    ));
    bench_blocking!(group, "parking_lot::RwLock", PlRwLock::new(Payload(0)), |_i, m| black_box(
        m.read().0
    ));
    bench_blocking!(group, "z_sync::Lock16 (Blocking)", ZLock16::new(Payload(0)), |_i, m| {
        black_box(m.read().0)
    });
    bench_blocking!(group, "z_sync::Lock32 (Blocking)", ZLock32::new(Payload(0)), |_i, m| {
        black_box(m.read().0)
    });
    bench_blocking!(group, "z_sync::Lock64 (Blocking)", ZLock64::new(Payload(0)), |_i, m| {
        black_box(m.read().0)
    });

    bench_async!(group, "tokio::Mutex", &rt, TkMutex::new(Payload(0)), |_i, m| black_box(
        m.lock().await.0
    ));
    bench_async!(group, "tokio::RwLock", &rt, TkRwLock::new(Payload(0)), |_i, m| black_box(
        m.read().await.0
    ));
    bench_async!(
        group,
        "z_sync::Lock16 (Async)",
        &rt,
        ZLock16::new(Payload(0)),
        |_i, m| black_box(m.read_async().await.0)
    );
    bench_async!(
        group,
        "z_sync::Lock32 (Async)",
        &rt,
        ZLock32::new(Payload(0)),
        |_i, m| black_box(m.read_async().await.0)
    );
    bench_async!(
        group,
        "z_sync::Lock64 (Async)",
        &rt,
        ZLock64::new(Payload(0)),
        |_i, m| black_box(m.read_async().await.0)
    );

    group.finish();
}

// ============================================================================
// 3. Write-Only Workload
// ============================================================================
fn bench_write_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_Write_Only");
    let rt = tokio::runtime::Runtime::new().unwrap();

    bench_blocking!(group, "std::Mutex", StdMutex::new(Payload(0)), |_i, m| black_box(
        m.lock().unwrap().0 += 1
    ));
    bench_blocking!(group, "std::RwLock", StdRwLock::new(Payload(0)), |_i, m| black_box(
        m.write().unwrap().0 += 1
    ));
    bench_blocking!(group, "parking_lot::Mutex", PlMutex::new(Payload(0)), |_i, m| black_box(
        m.lock().0 += 1
    ));
    bench_blocking!(group, "parking_lot::RwLock", PlRwLock::new(Payload(0)), |_i, m| black_box(
        m.write().0 += 1
    ));
    bench_blocking!(group, "z_sync::Lock32 (Blocking)", ZLock32::new(Payload(0)), |_i, m| {
        black_box(m.write().0 += 1)
    });
    bench_blocking!(group, "z_sync::Lock64 (Blocking)", ZLock64::new(Payload(0)), |_i, m| {
        black_box(m.write().0 += 1)
    });

    bench_async!(group, "tokio::Mutex", &rt, TkMutex::new(Payload(0)), |_i, m| black_box(
        m.lock().await.0 += 1
    ));
    bench_async!(group, "tokio::RwLock", &rt, TkRwLock::new(Payload(0)), |_i, m| black_box(
        m.write().await.0 += 1
    ));
    bench_async!(
        group,
        "z_sync::Lock32 (Async)",
        &rt,
        ZLock32::new(Payload(0)),
        |_i, m| black_box(m.write_async().await.0 += 1)
    );
    bench_async!(
        group,
        "z_sync::Lock64 (Async)",
        &rt,
        ZLock64::new(Payload(0)),
        |_i, m| black_box(m.write_async().await.0 += 1)
    );

    group.finish();
}

// ============================================================================
// 4. Read-Heavy Workload (90% Read, 10% Write)
// ============================================================================
fn bench_read_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("4_Read_Heavy");
    let rt = tokio::runtime::Runtime::new().unwrap();

    bench_blocking!(group, "std::Mutex", StdMutex::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.lock().unwrap().0 += 1);
        } else {
            black_box(m.lock().unwrap().0);
        }
    });
    bench_blocking!(group, "std::RwLock", StdRwLock::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.write().unwrap().0 += 1);
        } else {
            black_box(m.read().unwrap().0);
        }
    });
    bench_blocking!(group, "parking_lot::Mutex", PlMutex::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.lock().0 += 1);
        } else {
            black_box(m.lock().0);
        }
    });
    bench_blocking!(group, "parking_lot::RwLock", PlRwLock::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.write().0 += 1);
        } else {
            black_box(m.read().0);
        }
    });
    bench_blocking!(group, "z_sync::Lock32 (Blocking)", ZLock32::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.write().0 += 1);
        } else {
            black_box(m.read().0);
        }
    });
    bench_blocking!(group, "z_sync::Lock64 (Blocking)", ZLock64::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.write().0 += 1);
        } else {
            black_box(m.read().0);
        }
    });

    bench_async!(group, "tokio::Mutex", &rt, TkMutex::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.lock().await.0 += 1);
        } else {
            black_box(m.lock().await.0);
        }
    });
    bench_async!(group, "tokio::RwLock", &rt, TkRwLock::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.write().await.0 += 1);
        } else {
            black_box(m.read().await.0);
        }
    });
    bench_async!(group, "z_sync::Lock32 (Async)", &rt, ZLock32::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.write_async().await.0 += 1);
        } else {
            black_box(m.read_async().await.0);
        }
    });
    bench_async!(group, "z_sync::Lock64 (Async)", &rt, ZLock64::new(Payload(0)), |i, m| {
        if i % 10 == 0 {
            black_box(m.write_async().await.0 += 1);
        } else {
            black_box(m.read_async().await.0);
        }
    });

    group.finish();
}

// ============================================================================
// 5. Write-Heavy Workload (10% Read, 90% Write)
// ============================================================================
fn bench_write_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("5_Write_Heavy");
    let rt = tokio::runtime::Runtime::new().unwrap();

    bench_blocking!(group, "std::Mutex", StdMutex::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.lock().unwrap().0 += 1);
        } else {
            black_box(m.lock().unwrap().0);
        }
    });
    bench_blocking!(group, "std::RwLock", StdRwLock::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.write().unwrap().0 += 1);
        } else {
            black_box(m.read().unwrap().0);
        }
    });
    bench_blocking!(group, "parking_lot::Mutex", PlMutex::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.lock().0 += 1);
        } else {
            black_box(m.lock().0);
        }
    });
    bench_blocking!(group, "parking_lot::RwLock", PlRwLock::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.write().0 += 1);
        } else {
            black_box(m.read().0);
        }
    });
    bench_blocking!(group, "z_sync::Lock32 (Blocking)", ZLock32::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.write().0 += 1);
        } else {
            black_box(m.read().0);
        }
    });
    bench_blocking!(group, "z_sync::Lock64 (Blocking)", ZLock64::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.write().0 += 1);
        } else {
            black_box(m.read().0);
        }
    });

    bench_async!(group, "tokio::Mutex", &rt, TkMutex::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.lock().await.0 += 1);
        } else {
            black_box(m.lock().await.0);
        }
    });
    bench_async!(group, "tokio::RwLock", &rt, TkRwLock::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.write().await.0 += 1);
        } else {
            black_box(m.read().await.0);
        }
    });
    bench_async!(group, "z_sync::Lock32 (Async)", &rt, ZLock32::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.write_async().await.0 += 1);
        } else {
            black_box(m.read_async().await.0);
        }
    });
    bench_async!(group, "z_sync::Lock64 (Async)", &rt, ZLock64::new(Payload(0)), |i, m| {
        if i % 10 != 0 {
            black_box(m.write_async().await.0 += 1);
        } else {
            black_box(m.read_async().await.0);
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_uncontended,
    bench_read_only,
    bench_write_only,
    bench_read_heavy,
    bench_write_heavy
);
criterion_main!(benches);
