use std::hint::black_box;
use std::rc::Rc;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use z_sync::channels::broadcast::Broadcast;

const MESSAGES: u64 = 100;
const CAPACITY: usize = 256;
const RECEIVERS: usize = 8;

/// Cost of cloning a receiver handle (the fan-out primitive).
fn bench_receiver_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_receiver_clone");

    let channel = Arc::new(Broadcast::<u64>::new(CAPACITY));
    let (_tx, rx) = channel.arc_split();
    group.bench_function("z_sync (Arc)", |b| b.iter(|| black_box(rx.clone())));

    let channel = Rc::new(Broadcast::<u64>::new(CAPACITY));
    let (_tx, rx) = channel.rc_split();
    group.bench_function("z_sync (Rc)", |b| b.iter(|| black_box(rx.clone())));

    let (_tx, rx) = tokio::sync::broadcast::channel::<u64>(CAPACITY);
    group.bench_function("tokio", |b| b.iter(|| black_box(rx.resubscribe())));

    let (_tx, rx) = async_broadcast::broadcast::<u64>(CAPACITY);
    group.bench_function("async-broadcast", |b| b.iter(|| black_box(rx.clone())));

    group.finish();
}

/// Clone-and-spawn fan-out: one sender, `RECEIVERS` cloned receivers each spawned onto the runtime,
/// each draining `MESSAGES`. Uses z-sync's owned `Arc` handles (`arc_split` + clone).
fn bench_fanout(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_fanout_8rx");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("z_sync (Arc)", |b| {
        b.to_async(&rt).iter(|| async {
            let channel = Arc::new(Broadcast::<u64>::new(CAPACITY));
            let (tx, rx0) = channel.arc_split();
            let mut handles = Vec::with_capacity(RECEIVERS);
            for _ in 0..RECEIVERS {
                let mut rx = rx0.clone();
                handles.push(tokio::spawn(async move {
                    let mut sum = 0u64;
                    for _ in 0..MESSAGES {
                        sum += rx.recv_async().await.unwrap();
                    }
                    sum
                }));
            }
            drop(rx0);
            for i in 0..MESSAGES {
                tx.send(i).unwrap();
            }
            for h in handles {
                black_box(h.await.unwrap());
            }
        });
    });

    group.bench_function("tokio", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, rx0) = tokio::sync::broadcast::channel::<u64>(CAPACITY);
            let mut handles = Vec::with_capacity(RECEIVERS);
            for _ in 0..RECEIVERS {
                let mut rx = rx0.resubscribe();
                handles.push(tokio::spawn(async move {
                    let mut sum = 0u64;
                    for _ in 0..MESSAGES {
                        sum += rx.recv().await.unwrap();
                    }
                    sum
                }));
            }
            drop(rx0);
            for i in 0..MESSAGES {
                tx.send(i).unwrap();
            }
            for h in handles {
                black_box(h.await.unwrap());
            }
        });
    });

    group.bench_function("async-broadcast", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, rx0) = async_broadcast::broadcast::<u64>(CAPACITY);
            let mut handles = Vec::with_capacity(RECEIVERS);
            for _ in 0..RECEIVERS {
                let mut rx = rx0.clone();
                handles.push(tokio::spawn(async move {
                    let mut sum = 0u64;
                    for _ in 0..MESSAGES {
                        sum += rx.recv().await.unwrap();
                    }
                    sum
                }));
            }
            drop(rx0);
            for i in 0..MESSAGES {
                tx.broadcast(i).await.unwrap();
            }
            for h in handles {
                black_box(h.await.unwrap());
            }
        });
    });

    group.finish();
}

/// Send `MESSAGES` values, then drain them all through a single receiver.
fn bench_send_drain(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_send_drain_1rx");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("z_sync", |b| {
        b.to_async(&rt).iter(|| async {
            let mut chan = Broadcast::new(CAPACITY);
            let (tx, mut rx) = chan.split();
            for i in 0..MESSAGES {
                tx.send(i).unwrap();
            }
            let mut sum = 0u64;
            for _ in 0..MESSAGES {
                sum += rx.recv_async().await.unwrap();
            }
            black_box(sum)
        });
    });

    group.bench_function("tokio", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, mut rx) = tokio::sync::broadcast::channel(CAPACITY);
            for i in 0..MESSAGES {
                tx.send(i).unwrap();
            }
            let mut sum = 0u64;
            for _ in 0..MESSAGES {
                sum += rx.recv().await.unwrap();
            }
            black_box(sum)
        });
    });

    group.bench_function("async-broadcast", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, mut rx) = async_broadcast::broadcast(CAPACITY);
            for i in 0..MESSAGES {
                tx.broadcast(i).await.unwrap();
            }
            let mut sum = 0u64;
            for _ in 0..MESSAGES {
                sum += rx.recv().await.unwrap();
            }
            black_box(sum)
        });
    });

    group.finish();
}

/// Blocking fan-out across threads: one sender, N receivers each reading every message.
fn bench_blocking_fanout(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_blocking_fanout_4rx");
    const RECEIVERS: usize = 4;

    group.bench_function("z_sync", |b| {
        b.iter(|| {
            let mut chan = Broadcast::<u64>::new(CAPACITY);
            let (tx, rx0) = chan.split();
            std::thread::scope(|s| {
                for _ in 0..RECEIVERS {
                    let mut rx = rx0.clone();
                    s.spawn(move || {
                        let mut sum = 0u64;
                        for _ in 0..MESSAGES {
                            loop {
                                match rx.recv() {
                                    Ok(v) => {
                                        sum += v;
                                        break;
                                    }
                                    Err(z_sync::channels::broadcast::RecvError::Lagged(_)) => {
                                        continue;
                                    }
                                    Err(z_sync::channels::broadcast::RecvError::Closed) => {
                                        return sum;
                                    }
                                }
                            }
                        }
                        sum
                    });
                }
                drop(rx0);
                for i in 0..MESSAGES {
                    tx.send(i).unwrap();
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_send_drain,
    bench_blocking_fanout,
    bench_receiver_clone,
    bench_fanout
);
criterion_main!(benches);
