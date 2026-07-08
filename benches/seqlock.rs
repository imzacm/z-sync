use std::hint::black_box;
use std::sync::{Arc, RwLock as StdRwLock};

use arc_swap::ArcSwap;
use criterion::{Criterion, criterion_group, criterion_main};
use crossbeam_utils::atomic::AtomicCell;
use parking_lot::RwLock as PlRwLock;
use seqlock::SeqLock as RefSeqLock;
use z_sync::{Lock64 as ZLock, SeqLock as ZSeqLock};

// A small `Copy` payload too large for a single native atomic — the case SeqLock targets (a config
// snapshot / a set of counters that must be read consistently).
// Fields are only written and the whole struct is `black_box`ed on read, never inspected per-field.
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct Snapshot {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

impl Snapshot {
    #[inline]
    fn of(n: u64) -> Self {
        Self { a: n, b: n, c: n, d: n }
    }
}

const THREADS: usize = 8;
const READ_OPS: usize = 2000;
const MIX_OPS: usize = 1000;

// Spawns THREADS workers that each run `$body` `$ops` times; `$i` is the op index, `$l` the shared
// (Arc) handle. Used for both the read-only and read-heavy contended groups.
macro_rules! bench_contended {
    ($group:expr, $name:expr, $ops:expr, $init:expr, |$i:ident, $l:ident| $body:expr) => {
        $group.bench_function($name, |b| {
            b.iter(|| {
                let lock = Arc::new($init);
                std::thread::scope(|s| {
                    for _ in 0..THREADS {
                        let $l = Arc::clone(&lock);
                        s.spawn(move || {
                            for $i in 0..$ops {
                                $body;
                            }
                        });
                    }
                });
            });
        });
    };
}

// ============================================================================
// 1. Uncontended read: copy the snapshot out on a single thread.
// ============================================================================
fn bench_uncontended_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_Uncontended_Read");
    let v = Snapshot::of(7);

    group.bench_function("z_sync::SeqLock", |b| {
        let l = ZSeqLock::new(v);
        b.iter(|| black_box(l.read()));
    });
    group.bench_function("seqlock crate", |b| {
        let l = RefSeqLock::new(v);
        b.iter(|| black_box(l.read()));
    });
    group.bench_function("crossbeam::AtomicCell", |b| {
        let l = AtomicCell::new(v);
        b.iter(|| black_box(l.load()));
    });
    group.bench_function("arc_swap::ArcSwap", |b| {
        let l = ArcSwap::from_pointee(v);
        b.iter(|| black_box(**l.load()));
    });
    group.bench_function("parking_lot::RwLock", |b| {
        let l = PlRwLock::new(v);
        b.iter(|| black_box(*l.read()));
    });
    group.bench_function("std::RwLock", |b| {
        let l = StdRwLock::new(v);
        b.iter(|| black_box(*l.read().unwrap()));
    });
    group.bench_function("z_sync::Lock::read_copy", |b| {
        let l = ZLock::new(v);
        b.iter(|| black_box(l.read_copy()));
    });

    group.finish();
}

// ============================================================================
// 2. Uncontended write: publish a new snapshot on a single thread.
// ============================================================================
fn bench_uncontended_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Uncontended_Write");
    let v = Snapshot::of(9);

    group.bench_function("z_sync::SeqLock", |b| {
        let l = ZSeqLock::new(v);
        b.iter(|| l.set(black_box(v)));
    });
    group.bench_function("seqlock crate", |b| {
        let l = RefSeqLock::new(v);
        b.iter(|| *l.lock_write() = black_box(v));
    });
    group.bench_function("crossbeam::AtomicCell", |b| {
        let l = AtomicCell::new(v);
        b.iter(|| l.store(black_box(v)));
    });
    group.bench_function("arc_swap::ArcSwap", |b| {
        let l = ArcSwap::from_pointee(v);
        b.iter(|| l.store(Arc::new(black_box(v))));
    });
    group.bench_function("parking_lot::RwLock", |b| {
        let l = PlRwLock::new(v);
        b.iter(|| *l.write() = black_box(v));
    });
    group.bench_function("std::RwLock", |b| {
        let l = StdRwLock::new(v);
        b.iter(|| *l.write().unwrap() = black_box(v));
    });
    group.bench_function("z_sync::Lock", |b| {
        let l = ZLock::new(v);
        b.iter(|| *l.write() = black_box(v));
    });

    group.finish();
}

// ============================================================================
// 3. Read-only under contention: THREADS readers, no writer. SeqLock readers only *load* the
//    sequence word (no RMW), so they should scale without cache-line ping-pong; RwLock readers RMW
//    a shared reader count.
// ============================================================================
fn bench_readonly_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_ReadOnly_Contended");
    let v = Snapshot::of(3);

    bench_contended!(group, "z_sync::SeqLock", READ_OPS, ZSeqLock::new(v), |_i, l| black_box(
        l.read()
    ));
    bench_contended!(group, "seqlock crate", READ_OPS, RefSeqLock::new(v), |_i, l| black_box(
        l.read()
    ));
    bench_contended!(group, "crossbeam::AtomicCell", READ_OPS, AtomicCell::new(v), |_i, l| {
        black_box(l.load())
    });
    bench_contended!(group, "arc_swap::ArcSwap", READ_OPS, ArcSwap::from_pointee(v), |_i, l| {
        black_box(**l.load())
    });
    bench_contended!(group, "parking_lot::RwLock", READ_OPS, PlRwLock::new(v), |_i, l| black_box(
        *l.read()
    ));
    bench_contended!(group, "std::RwLock", READ_OPS, StdRwLock::new(v), |_i, l| black_box(
        *l.read().unwrap()
    ));
    bench_contended!(group, "z_sync::Lock::read_copy", READ_OPS, ZLock::new(v), |_i, l| {
        black_box(l.read_copy())
    });

    group.finish();
}

// ============================================================================
// 4. Read-heavy under contention: THREADS workers, 90% read / 10% write (the realistic read-mostly
//    mix). SeqLock readers never block the writer and vice versa.
// ============================================================================
fn bench_readheavy_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("4_ReadHeavy_Contended");
    let v = Snapshot::of(1);

    bench_contended!(group, "z_sync::SeqLock", MIX_OPS, ZSeqLock::new(v), |i, l| {
        if i % 10 == 0 {
            l.set(Snapshot::of(i as u64));
        } else {
            black_box(l.read());
        }
    });
    bench_contended!(group, "seqlock crate", MIX_OPS, RefSeqLock::new(v), |i, l| {
        if i % 10 == 0 {
            *l.lock_write() = Snapshot::of(i as u64);
        } else {
            black_box(l.read());
        }
    });
    bench_contended!(group, "crossbeam::AtomicCell", MIX_OPS, AtomicCell::new(v), |i, l| {
        if i % 10 == 0 {
            l.store(Snapshot::of(i as u64));
        } else {
            black_box(l.load());
        }
    });
    bench_contended!(group, "arc_swap::ArcSwap", MIX_OPS, ArcSwap::from_pointee(v), |i, l| {
        if i % 10 == 0 {
            l.store(Arc::new(Snapshot::of(i as u64)));
        } else {
            black_box(**l.load());
        }
    });
    bench_contended!(group, "parking_lot::RwLock", MIX_OPS, PlRwLock::new(v), |i, l| {
        if i % 10 == 0 {
            *l.write() = Snapshot::of(i as u64);
        } else {
            black_box(*l.read());
        }
    });
    bench_contended!(group, "std::RwLock", MIX_OPS, StdRwLock::new(v), |i, l| {
        if i % 10 == 0 {
            *l.write().unwrap() = Snapshot::of(i as u64);
        } else {
            black_box(*l.read().unwrap());
        }
    });
    bench_contended!(group, "z_sync::Lock (read_copy)", MIX_OPS, ZLock::new(v), |i, l| {
        if i % 10 == 0 {
            *l.write() = Snapshot::of(i as u64);
        } else {
            black_box(l.read_copy());
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_uncontended_read,
    bench_uncontended_write,
    bench_readonly_contended,
    bench_readheavy_contended
);
criterion_main!(benches);
