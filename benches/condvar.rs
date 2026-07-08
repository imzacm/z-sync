use std::sync::{Arc, Condvar as StdCondvar, Mutex as StdMutex};

use async_std::sync::{Condvar as AsCondvar, Mutex as AsMutex};
use criterion::{Criterion, criterion_group, criterion_main};
use parking_lot::{Condvar as PlCondvar, Mutex as PlMutex};
use tokio::sync::{Mutex as TkMutex, Notify as TkNotify};
use z_sync::{Condvar32 as ZCondvar32, Condvar64 as ZCondvar64, Lock64 as ZLock};

// Total one-way handoffs in the ping-pong benchmark. Each handoff is a wait/notify pair plus a
// context switch, so this dominates wall-clock; keep it modest.
const HANDOFFS: usize = 1000;

// ============================================================================
// 1. Uncontended: `notify` with no registered waiters — just the epoch bump (z_sync) or the
//    internal fast path (std / parking_lot). This is the cost paid on every signal in the common
//    "signal while holding the lock, nobody parked yet" case.
// ============================================================================
fn bench_uncontended(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_NotifyNoWaiters");

    group.bench_function("z_sync::Condvar32", |b| {
        let cvar = ZCondvar32::new();
        b.iter(|| cvar.notify_one());
    });
    group.bench_function("z_sync::Condvar64", |b| {
        let cvar = ZCondvar64::new();
        b.iter(|| cvar.notify_one());
    });
    group.bench_function("std::sync::Condvar", |b| {
        let cvar = StdCondvar::new();
        b.iter(|| cvar.notify_one());
    });
    group.bench_function("parking_lot::Condvar", |b| {
        let cvar = PlCondvar::new();
        b.iter(|| cvar.notify_one());
    });

    group.finish();
}

// ============================================================================
// 2. Handoff (ping-pong): two parties bounce a turn counter back and forth `HANDOFFS` times through
//    a single mutex + condvar. Each party waits for its parity, bumps the counter, and signals the
//    other — the canonical condvar stress test. std and parking_lot are blocking-only; the async
//    row exercises z_sync's async wait path (no comparable async condvar exists in the ecosystem).
// ============================================================================
fn bench_handoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Handoff");

    group.bench_function("z_sync::Condvar64 (Blocking)", |b| {
        b.iter(|| {
            let pair = Arc::new((ZLock::new(0usize), ZCondvar64::new()));
            std::thread::scope(|scope| {
                for parity in 0..2 {
                    let pair = Arc::clone(&pair);
                    scope.spawn(move || {
                        let (lock, cvar) = &*pair;
                        let mut turn = lock.write();
                        loop {
                            if *turn >= HANDOFFS {
                                cvar.notify_one();
                                break;
                            }
                            if *turn % 2 == parity {
                                *turn += 1;
                                cvar.notify_one();
                            } else {
                                turn = cvar.wait(turn);
                            }
                        }
                    });
                }
            });
        });
    });

    group.bench_function("std::sync::Condvar (Blocking)", |b| {
        b.iter(|| {
            let pair = Arc::new((StdMutex::new(0usize), StdCondvar::new()));
            std::thread::scope(|scope| {
                for parity in 0..2 {
                    let pair = Arc::clone(&pair);
                    scope.spawn(move || {
                        let (lock, cvar) = &*pair;
                        let mut turn = lock.lock().unwrap();
                        loop {
                            if *turn >= HANDOFFS {
                                cvar.notify_one();
                                break;
                            }
                            if *turn % 2 == parity {
                                *turn += 1;
                                cvar.notify_one();
                            } else {
                                turn = cvar.wait(turn).unwrap();
                            }
                        }
                    });
                }
            });
        });
    });

    group.bench_function("parking_lot::Condvar (Blocking)", |b| {
        b.iter(|| {
            let pair = Arc::new((PlMutex::new(0usize), PlCondvar::new()));
            std::thread::scope(|scope| {
                for parity in 0..2 {
                    let pair = Arc::clone(&pair);
                    scope.spawn(move || {
                        let (lock, cvar) = &*pair;
                        let mut turn = lock.lock();
                        loop {
                            if *turn >= HANDOFFS {
                                cvar.notify_one();
                                break;
                            }
                            if *turn % 2 == parity {
                                *turn += 1;
                                cvar.notify_one();
                            } else {
                                cvar.wait(&mut turn);
                            }
                        }
                    });
                }
            });
        });
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("z_sync::Condvar64 (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let pair = Arc::new((ZLock::new(0usize), ZCondvar64::new()));
            let mut handles = Vec::with_capacity(2);
            for parity in 0..2 {
                let pair = Arc::clone(&pair);
                handles.push(tokio::spawn(async move {
                    let (lock, cvar) = &*pair;
                    let mut turn = lock.write_async().await;
                    loop {
                        if *turn >= HANDOFFS {
                            cvar.notify_one();
                            break;
                        }
                        if *turn % 2 == parity {
                            *turn += 1;
                            cvar.notify_one();
                        } else {
                            turn = cvar.wait_async(turn).await;
                        }
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });

    // tokio has no `Condvar`; the idiomatic async equivalent is `Notify` + `Mutex`, registering
    // interest (`notified().enable()`) *before* releasing the lock — the same discipline `Condvar`
    // encapsulates. This is what `z_sync::Condvar` most directly replaces in async code.
    group.bench_function("tokio Notify+Mutex (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let pair = Arc::new((TkMutex::new(0usize), TkNotify::new()));
            let mut handles = Vec::with_capacity(2);
            for parity in 0..2 {
                let pair = Arc::clone(&pair);
                handles.push(tokio::spawn(async move {
                    let (mutex, notify) = &*pair;
                    loop {
                        let mut turn = mutex.lock().await;
                        if *turn >= HANDOFFS {
                            notify.notify_waiters();
                            break;
                        }
                        if *turn % 2 == parity {
                            *turn += 1;
                            drop(turn);
                            notify.notify_one();
                        } else {
                            let notified = notify.notified();
                            tokio::pin!(notified);
                            // Enroll before releasing the lock so the wake can't be lost.
                            notified.as_mut().enable();
                            drop(turn);
                            notified.await;
                        }
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });

    // A genuine async `Condvar` API (async-std), for an apples-to-apples comparison.
    group.bench_function("async-std Condvar (Async)", |b| {
        b.to_async(&rt).iter(|| async {
            let pair = Arc::new((AsMutex::new(0usize), AsCondvar::new()));
            let mut handles = Vec::with_capacity(2);
            for parity in 0..2 {
                let pair = Arc::clone(&pair);
                handles.push(tokio::spawn(async move {
                    let (mutex, cvar) = &*pair;
                    let mut turn = mutex.lock().await;
                    loop {
                        if *turn >= HANDOFFS {
                            cvar.notify_all();
                            break;
                        }
                        if *turn % 2 == parity {
                            *turn += 1;
                            cvar.notify_one();
                        } else {
                            turn = cvar.wait(turn).await;
                        }
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

criterion_group!(benches, bench_uncontended, bench_handoff);
criterion_main!(benches);
