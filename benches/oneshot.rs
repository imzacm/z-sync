use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use z_sync::oneshot::OneShot;

/// Round-trip: create a channel, send a value, receive it. Measures per-use overhead
/// (construction + send + receive) on the resolved fast path.
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("oneshot_roundtrip");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("z_sync", |b| {
        b.to_async(&rt).iter(|| async {
            let mut chan = OneShot::new();
            let (tx, rx) = chan.split();
            tx.send(black_box(1u64)).unwrap();
            black_box(rx.recv_async().await.unwrap())
        });
    });

    group.bench_function("tokio", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, rx) = tokio::sync::oneshot::channel();
            tx.send(black_box(1u64)).unwrap();
            black_box(rx.await.unwrap())
        });
    });

    group.bench_function("futures", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, rx) = futures::channel::oneshot::channel();
            tx.send(black_box(1u64)).unwrap();
            black_box(rx.await.unwrap())
        });
    });

    group.finish();
}

/// Blocking round-trip on a single thread (z_sync only; tokio/futures are async-only).
fn bench_blocking(c: &mut Criterion) {
    let mut group = c.benchmark_group("oneshot_blocking_roundtrip");

    group.bench_function("z_sync", |b| {
        b.iter(|| {
            let mut chan = OneShot::new();
            let (tx, rx) = chan.split();
            tx.send(black_box(1u64)).unwrap();
            black_box(rx.recv().unwrap())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_roundtrip, bench_blocking);
criterion_main!(benches);
