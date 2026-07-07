use std::hint::black_box;
use std::rc::Rc;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use z_sync::channels::watch::Watch;

const RECEIVERS: usize = 8;

/// Cost of cloning a receiver handle (the fan-out primitive): pointer refcount + internal counter.
fn bench_receiver_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("watch_receiver_clone");

    let channel = Arc::new(Watch::new(0u64));
    let (_tx, rx) = channel.arc_split();
    group.bench_function("z_sync (Arc)", |b| b.iter(|| black_box(rx.clone())));

    let channel = Rc::new(Watch::new(0u64));
    let (_tx, rx) = channel.rc_split();
    group.bench_function("z_sync (Rc)", |b| b.iter(|| black_box(rx.clone())));

    let (_tx, rx) = tokio::sync::watch::channel(0u64);
    group.bench_function("tokio", |b| b.iter(|| black_box(rx.clone())));

    group.finish();
}

/// Clone-and-spawn fan-out: one sender, `RECEIVERS` cloned receivers each spawned onto the runtime,
/// all observing a single change. Uses z-sync's owned `Arc` handles (`arc_split` + clone).
fn bench_fanout(c: &mut Criterion) {
    let mut group = c.benchmark_group("watch_fanout_8rx");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("z_sync (Arc)", |b| {
        b.to_async(&rt).iter(|| async {
            let channel = Arc::new(Watch::new(0u64));
            let (tx, rx0) = channel.arc_split();
            let mut handles = Vec::with_capacity(RECEIVERS);
            for _ in 0..RECEIVERS {
                let mut rx = rx0.clone();
                handles.push(tokio::spawn(async move {
                    rx.changed_async().await.unwrap();
                    *rx.borrow()
                }));
            }
            drop(rx0);
            tx.send(1).unwrap();
            for h in handles {
                black_box(h.await.unwrap());
            }
        });
    });

    group.bench_function("tokio", |b| {
        b.to_async(&rt).iter(|| async {
            let (tx, rx0) = tokio::sync::watch::channel(0u64);
            let mut handles = Vec::with_capacity(RECEIVERS);
            for _ in 0..RECEIVERS {
                let mut rx = rx0.clone();
                handles.push(tokio::spawn(async move {
                    rx.changed().await.unwrap();
                    *rx.borrow()
                }));
            }
            drop(rx0);
            tx.send(1).unwrap();
            for h in handles {
                black_box(h.await.unwrap());
            }
        });
    });

    group.finish();
}

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

criterion_group!(benches, bench_send_changed, bench_borrow, bench_receiver_clone, bench_fanout);
criterion_main!(benches);
