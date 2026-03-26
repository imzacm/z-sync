use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use criterion::{Criterion, criterion_group, criterion_main};
use event_listener::{Event, Listener};
use tokio::runtime::Runtime;
use tokio::sync::Barrier as AsyncBarrier;
use z_sync::notify::Notify;

/// 1. Uncontended notify (no listeners registered)
fn bench_uncontended(c: &mut Criterion) {
    let mut group = c.benchmark_group("uncontended_notify");

    group.bench_function("z_queue::Notify", |b| {
        let notify = Notify::new();
        b.iter(|| {
            notify.notify(1);
        })
    });

    group.bench_function("event_listener::Event", |b| {
        let event = Event::new();
        b.iter(|| {
            event.notify(1);
        })
    });

    group.bench_function("tokio::sync::Notify", |b| {
        let notify = tokio::sync::Notify::new();
        b.iter(|| {
            notify.notify_one();
        })
    });

    group.finish();
}

/// 2. Blocking fast-path (Create -> Notify -> Wait on same thread)
/// (tokio::sync::Notify omitted as it lacks a blocking API)
fn bench_blocking_fast_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocking_fast_path");

    group.bench_function("z_queue::Notify", |b| {
        let notify = Notify::new();
        b.iter(|| {
            let listener = notify.listener();
            notify.notify(1);
            listener.wait();
        })
    });

    group.bench_function("event_listener::Event", |b| {
        let event = Event::new();
        b.iter(|| {
            let listener = event.listen();
            event.notify(1);
            listener.wait();
        })
    });

    group.finish();
}

/// 3. Async fast-path (Create -> Notify -> Await on same task)
fn bench_async_fast_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_fast_path");
    let rt = Runtime::new().unwrap();

    group.bench_function("z_queue::Notify", |b| {
        let notify = Notify::new();
        b.to_async(&rt).iter(|| async {
            let listener = notify.listener();
            notify.notify(1);
            listener.await;
        })
    });

    group.bench_function("event_listener::Event", |b| {
        let event = Event::new();
        b.to_async(&rt).iter(|| async {
            let listener = event.listen();
            event.notify(1);
            listener.await;
        })
    });

    group.bench_function("tokio::sync::Notify", |b| {
        let notify = tokio::sync::Notify::new();
        b.to_async(&rt).iter(|| async {
            let listener = notify.notified();
            notify.notify_one();
            listener.await;
        })
    });

    group.finish();
}

/// 4. Async Ping-Pong (Cross-task notification)
fn bench_async_ping_pong(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_ping_pong_100_iters");
    let rt = Runtime::new().unwrap();
    const ITERS: usize = 100;

    group.bench_function("z_queue::Notify", |b| {
        b.to_async(&rt).iter(|| async {
            let n1 = Arc::new(Notify::new());
            let n2 = Arc::new(Notify::new());
            let turn = Arc::new(AtomicUsize::new(0));

            let n1_clone = n1.clone();
            let n2_clone = n2.clone();
            let turn_clone = turn.clone();

            let task = tokio::spawn(async move {
                for i in 0..ITERS {
                    let expected_turn = i * 2 + 1;
                    loop {
                        if turn_clone.load(Ordering::Acquire) == expected_turn {
                            break;
                        }
                        let l = n2_clone.listener();
                        if turn_clone.load(Ordering::Acquire) == expected_turn {
                            break;
                        }
                        l.await;
                    }

                    turn_clone.store(i * 2 + 2, Ordering::Release);
                    n1_clone.notify(1);
                }
            });

            for i in 0..ITERS {
                turn.store(i * 2 + 1, Ordering::Release);
                n2.notify(1);

                let expected_turn = i * 2 + 2;
                loop {
                    if turn.load(Ordering::Acquire) == expected_turn {
                        break;
                    }
                    let l = n1.listener();
                    if turn.load(Ordering::Acquire) == expected_turn {
                        break;
                    }
                    l.await;
                }
            }

            task.await.unwrap();
        });
    });

    group.bench_function("event_listener::Event", |b| {
        // ... (Keep existing event_listener implementation intact)
        b.to_async(&rt).iter(|| async {
            let e1 = Arc::new(Event::new());
            let e2 = Arc::new(Event::new());
            let turn = Arc::new(AtomicUsize::new(0));

            let e1_clone = e1.clone();
            let e2_clone = e2.clone();
            let turn_clone = turn.clone();

            let task = tokio::spawn(async move {
                for i in 0..ITERS {
                    let expected_turn = i * 2 + 1;
                    loop {
                        if turn_clone.load(Ordering::Acquire) == expected_turn {
                            break;
                        }
                        let l = e2_clone.listen();
                        if turn_clone.load(Ordering::Acquire) == expected_turn {
                            break;
                        }
                        l.await;
                    }

                    turn_clone.store(i * 2 + 2, Ordering::Release);
                    e1_clone.notify(1);
                }
            });

            for i in 0..ITERS {
                turn.store(i * 2 + 1, Ordering::Release);
                e2.notify(1);

                let expected_turn = i * 2 + 2;
                loop {
                    if turn.load(Ordering::Acquire) == expected_turn {
                        break;
                    }
                    let l = e1.listen();
                    if turn.load(Ordering::Acquire) == expected_turn {
                        break;
                    }
                    l.await;
                }
            }

            task.await.unwrap();
        });
    });

    group.bench_function("tokio::sync::Notify", |b| {
        b.to_async(&rt).iter(|| async {
            let t1 = Arc::new(tokio::sync::Notify::new());
            let t2 = Arc::new(tokio::sync::Notify::new());
            let turn = Arc::new(AtomicUsize::new(0));

            let t1_clone = t1.clone();
            let t2_clone = t2.clone();
            let turn_clone = turn.clone();

            let task = tokio::spawn(async move {
                for i in 0..ITERS {
                    let expected_turn = i * 2 + 1;
                    loop {
                        if turn_clone.load(Ordering::Acquire) == expected_turn {
                            break;
                        }
                        let l = t2_clone.notified();
                        if turn_clone.load(Ordering::Acquire) == expected_turn {
                            break;
                        }
                        l.await;
                    }

                    turn_clone.store(i * 2 + 2, Ordering::Release);
                    t1_clone.notify_one();
                }
            });

            for i in 0..ITERS {
                turn.store(i * 2 + 1, Ordering::Release);
                t2.notify_one();

                let expected_turn = i * 2 + 2;
                loop {
                    if turn.load(Ordering::Acquire) == expected_turn {
                        break;
                    }
                    let l = t1.notified();
                    if turn.load(Ordering::Acquire) == expected_turn {
                        break;
                    }
                    l.await;
                }
            }

            task.await.unwrap();
        });
    });

    group.finish();
}

/// 5. Blocking Thundering Herd (Wake many threads at once)
/// (tokio::sync::Notify omitted as it lacks a blocking API)
fn bench_blocking_thundering_herd(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocking_thundering_herd_10_threads");
    const THREADS: usize = 10;

    group.bench_function("z_queue::Notify", |b| {
        b.iter(|| {
            let notify = Notify::new();
            let ready = Barrier::new(THREADS + 1);

            std::thread::scope(|s| {
                for _ in 0..THREADS {
                    s.spawn(|| {
                        let listener = notify.listener();
                        ready.wait();
                        listener.wait();
                    });
                }
                ready.wait(); // Wait for all to register listeners
                notify.notify(usize::MAX); // Wake all
            });
        })
    });

    group.bench_function("event_listener::Event", |b| {
        b.iter(|| {
            let event = Event::new();
            let ready = Barrier::new(THREADS + 1);

            std::thread::scope(|s| {
                for _ in 0..THREADS {
                    s.spawn(|| {
                        let listener = event.listen();
                        ready.wait();
                        listener.wait();
                    });
                }
                ready.wait();
                event.notify(usize::MAX);
            });
        })
    });

    group.finish();
}

/// 6. Blocking MPSC (Many notifiers, one waiter)
/// (tokio::sync::Notify omitted as it lacks a blocking API)
fn bench_blocking_mpsc(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocking_mpsc_10_producers");
    const PRODUCERS: usize = 10;
    const ITERS_PER_PROD: usize = 100;

    group.bench_function("z_queue::Notify", |b| {
        // ... (Keep existing z_queue implementation intact)
        b.iter(|| {
            let notify = Notify::new();
            let counter = AtomicUsize::new(0);

            std::thread::scope(|s| {
                // Consumer
                s.spawn(|| {
                    let mut total = 0;
                    let target = PRODUCERS * ITERS_PER_PROD;
                    while total < target {
                        let l = notify.listener();
                        let current = counter.load(Ordering::Acquire);
                        if current > total {
                            total = current;
                        } else {
                            l.wait();
                            total = counter.load(Ordering::Acquire);
                        }
                    }
                });

                // Producers
                for _ in 0..PRODUCERS {
                    s.spawn(|| {
                        for _ in 0..ITERS_PER_PROD {
                            counter.fetch_add(1, Ordering::Release);
                            notify.notify(1);
                        }
                    });
                }
            });
        })
    });

    group.bench_function("event_listener::Event", |b| {
        // ... (Keep existing event_listener implementation intact)
        b.iter(|| {
            let event = Event::new();
            let counter = AtomicUsize::new(0);

            std::thread::scope(|s| {
                // Consumer
                s.spawn(|| {
                    let mut total = 0;
                    let target = PRODUCERS * ITERS_PER_PROD;
                    while total < target {
                        let l = event.listen();
                        let current = counter.load(Ordering::Acquire);
                        if current > total {
                            total = current;
                        } else {
                            l.wait();
                            total = counter.load(Ordering::Acquire);
                        }
                    }
                });

                // Producers
                for _ in 0..PRODUCERS {
                    s.spawn(|| {
                        for _ in 0..ITERS_PER_PROD {
                            counter.fetch_add(1, Ordering::Release);
                            event.notify(1);
                        }
                    });
                }
            });
        })
    });

    group.finish();
}

/// 7. Async Thundering Herd (Wake many tasks at once)
fn bench_async_thundering_herd(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_thundering_herd_100_tasks");
    let rt = Runtime::new().unwrap();
    const TASKS: usize = 100;

    group.bench_function("z_queue::Notify", |b| {
        b.to_async(&rt).iter(|| async {
            let notify = Arc::new(Notify::new());
            let barrier = Arc::new(AsyncBarrier::new(TASKS + 1));
            let mut handles = Vec::with_capacity(TASKS);

            for _ in 0..TASKS {
                let notify_clone = notify.clone();
                let barrier_clone = barrier.clone();
                handles.push(tokio::spawn(async move {
                    let listener = notify_clone.listener();
                    barrier_clone.wait().await;
                    listener.await;
                }));
            }

            barrier.wait().await; // Ensure all tasks are awaiting
            notify.notify(usize::MAX);

            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.bench_function("event_listener::Event", |b| {
        b.to_async(&rt).iter(|| async {
            let event = Arc::new(Event::new());
            let barrier = Arc::new(AsyncBarrier::new(TASKS + 1));
            let mut handles = Vec::with_capacity(TASKS);

            for _ in 0..TASKS {
                let event_clone = event.clone();
                let barrier_clone = barrier.clone();
                handles.push(tokio::spawn(async move {
                    let listener = event_clone.listen();
                    barrier_clone.wait().await;
                    listener.await;
                }));
            }

            barrier.wait().await;
            event.notify(usize::MAX);

            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.bench_function("tokio::sync::Notify", |b| {
        b.to_async(&rt).iter(|| async {
            let notify = Arc::new(tokio::sync::Notify::new());
            let barrier = Arc::new(AsyncBarrier::new(TASKS + 1));
            let mut handles = Vec::with_capacity(TASKS);

            for _ in 0..TASKS {
                let notify_clone = notify.clone();
                let barrier_clone = barrier.clone();
                handles.push(tokio::spawn(async move {
                    let listener = notify_clone.notified();
                    barrier_clone.wait().await;
                    listener.await;
                }));
            }

            barrier.wait().await;
            // notify_waiters is the semantic equivalent of notify(usize::MAX)
            notify.notify_waiters();

            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.finish();
}

/// 8. Async Chain (Domino effect: Task 0 -> Task 1 -> Task 2...)
fn bench_async_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_chain_50_tasks");
    let rt = Runtime::new().unwrap();
    const TASKS: usize = 50;

    group.bench_function("z_queue::Notify", |b| {
        b.to_async(&rt).iter(|| async {
            let mut notifies = Vec::with_capacity(TASKS + 1);
            for _ in 0..=TASKS {
                notifies.push(Arc::new(Notify::new()));
            }

            let barrier = Arc::new(AsyncBarrier::new(TASKS + 1));
            let mut handles = Vec::with_capacity(TASKS);

            for i in 0..TASKS {
                let wait_notify = notifies[i].clone();
                let wake_notify = notifies[i + 1].clone();
                let b = barrier.clone();

                handles.push(tokio::spawn(async move {
                    let listener = wait_notify.listener();
                    b.wait().await; // Synchronize setup
                    listener.await;
                    wake_notify.notify(1);
                }));
            }

            barrier.wait().await;
            notifies[0].notify(1); // Knock over the first domino

            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.bench_function("event_listener::Event", |b| {
        b.to_async(&rt).iter(|| async {
            let mut events = Vec::with_capacity(TASKS + 1);
            for _ in 0..=TASKS {
                events.push(Arc::new(Event::new()));
            }

            let barrier = Arc::new(AsyncBarrier::new(TASKS + 1));
            let mut handles = Vec::with_capacity(TASKS);

            for i in 0..TASKS {
                let wait_event = events[i].clone();
                let wake_event = events[i + 1].clone();
                let b = barrier.clone();

                handles.push(tokio::spawn(async move {
                    let listener = wait_event.listen();
                    b.wait().await;
                    listener.await;
                    wake_event.notify(1);
                }));
            }

            barrier.wait().await;
            events[0].notify(1);

            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.bench_function("tokio::sync::Notify", |b| {
        b.to_async(&rt).iter(|| async {
            let mut notifies = Vec::with_capacity(TASKS + 1);
            for _ in 0..=TASKS {
                notifies.push(Arc::new(tokio::sync::Notify::new()));
            }

            let barrier = Arc::new(AsyncBarrier::new(TASKS + 1));
            let mut handles = Vec::with_capacity(TASKS);

            for i in 0..TASKS {
                let wait_notify = notifies[i].clone();
                let wake_notify = notifies[i + 1].clone();
                let b = barrier.clone();

                handles.push(tokio::spawn(async move {
                    let listener = wait_notify.notified();
                    b.wait().await;
                    listener.await;
                    wake_notify.notify_one();
                }));
            }

            barrier.wait().await;
            notifies[0].notify_one();

            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_uncontended,
    bench_blocking_fast_path,
    bench_async_fast_path,
    bench_async_ping_pong,
    bench_blocking_thundering_herd,
    bench_blocking_mpsc,
    bench_async_thundering_herd,
    bench_async_chain
);
criterion_main!(benches);
