use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use z_sync::channels::broadcast::Broadcast;

const MESSAGES: u64 = 100;
const CAPACITY: usize = 256;

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
                                    Err(z_sync::channels::broadcast::RecvError::Lagged(_)) => continue,
                                    Err(z_sync::channels::broadcast::RecvError::Closed) => return sum,
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

criterion_group!(benches, bench_send_drain, bench_blocking_fanout);
criterion_main!(benches);
