use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use z_sync::channels::oneshot::OneShot;

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

/// Owned-handle scenario: spawn a worker task that produces the value, and await it on the
/// receiver. This is the realistic one-shot pattern and needs owned (`Arc`-backed) handles, which
/// every competitor also uses; z-sync builds them from `arc_split`.
fn bench_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("oneshot_spawn_worker");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("z_sync (Arc)", |b| {
        b.to_async(&rt).iter(|| async {
            let channel = Arc::new(OneShot::<u64>::new());
            let (tx, rx) = channel.arc_split();
            tokio::spawn(async move {
                tx.send(black_box(1)).unwrap();
            });
            black_box(rx.recv_async().await.unwrap())
        });
    });

    group.bench_function("tokio", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                tx.send(black_box(1u64)).unwrap();
            });
            black_box(rx.await.unwrap())
        });
    });

    group.bench_function("futures", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, rx) = futures::channel::oneshot::channel();
            tokio::spawn(async move {
                tx.send(black_box(1u64)).unwrap();
            });
            black_box(rx.await.unwrap())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_roundtrip, bench_blocking, bench_spawn);
criterion_main!(benches);
