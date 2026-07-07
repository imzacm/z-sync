use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use z_sync::watch::Watch;

/// Round-trip: send a new value and observe the change from a receiver.
fn bench_send_changed(c: &mut Criterion) {
    let mut group = c.benchmark_group("watch_send_changed");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("z_sync", |b| {
        b.to_async(&rt).iter(|| async {
            let mut chan = Watch::new(0u64);
            let (tx, mut rx) = chan.split();
            tx.send(black_box(1)).unwrap();
            rx.changed_async().await.unwrap();
            black_box(*rx.borrow())
        });
    });

    group.bench_function("tokio", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, mut rx) = tokio::sync::watch::channel(0u64);
            tx.send(black_box(1)).unwrap();
            rx.changed().await.unwrap();
            black_box(*rx.borrow())
        });
    });

    group.finish();
}

/// Uncontended `borrow` of the current value.
fn bench_borrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("watch_borrow");

    group.bench_function("z_sync", |b| {
        let mut chan = Watch::new(0u64);
        let (_tx, rx) = chan.split();
        b.iter(|| black_box(*rx.borrow()));
    });

    group.bench_function("tokio", |b| {
        let (_tx, rx) = tokio::sync::watch::channel(0u64);
        b.iter(|| black_box(*rx.borrow()));
    });

    group.finish();
}

criterion_group!(benches, bench_send_changed, bench_borrow);
criterion_main!(benches);
