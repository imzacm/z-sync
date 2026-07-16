use std::hint::black_box;
use std::sync::{Arc, Mutex as StdMutex, RwLock as StdRwLock};

use arc_swap::ArcSwap;
use criterion::{Criterion, criterion_group, criterion_main};
use parking_lot::{Mutex as PlMutex, RwLock as PlRwLock};
use z_sync::{AtomicArc, Lock64 as ZLock};

// The payload `AtomicArc` targets: a snapshot published whole and read by many, too big to be a
// native atomic and shared rather than copied. Only ever written and `black_box`ed, never inspected
// per-field.
#[allow(dead_code)]
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

    #[inline]
    fn arc(n: u64) -> Arc<Self> {
        Arc::new(Self::of(n))
    }
}

const THREADS: usize = 8;
const READ_OPS: usize = 2000;
const MIX_OPS: usize = 1000;

// Spawns THREADS workers that each run `$body` `$ops` times; `$i` is the op index, `$c` the shared
// (Arc) handle. Used for all three contended groups.
macro_rules! bench_contended {
    ($group:expr, $name:expr, $ops:expr, $init:expr, |$i:ident, $c:ident| $body:expr) => {
        $group.bench_function($name, |b| {
            b.iter(|| {
                let cell = Arc::new($init);
                std::thread::scope(|s| {
                    for _ in 0..THREADS {
                        let $c = Arc::clone(&cell);
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
// 1. Uncontended load: take an owned snapshot handle on a single thread.
//
// Every contender returns a full `Arc<Snapshot>`, so all pay the strong-count bump; what differs is
// what they pay around it. `arc_swap::ArcSwap::load` is also listed because its borrowed `Guard` is
// the idiomatic fast path — it is *not* like-for-like with the rest, it skips the bump.
// ============================================================================
fn bench_uncontended_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_Uncontended_Load");

    group.bench_function("z_sync::AtomicArc", |b| {
        let cell = AtomicArc::new(Snapshot::arc(7));
        b.iter(|| black_box(cell.load()));
    });
    group.bench_function("arc_swap::ArcSwap (load_full)", |b| {
        let cell = ArcSwap::from(Snapshot::arc(7));
        b.iter(|| black_box(cell.load_full()));
    });
    group.bench_function("arc_swap::ArcSwap (load, borrowed)", |b| {
        let cell = ArcSwap::from(Snapshot::arc(7));
        b.iter(|| black_box(cell.load()));
    });
    group.bench_function("parking_lot::RwLock<Arc>", |b| {
        let cell = PlRwLock::new(Snapshot::arc(7));
        b.iter(|| black_box(Arc::clone(&cell.read())));
    });
    group.bench_function("parking_lot::Mutex<Arc>", |b| {
        let cell = PlMutex::new(Snapshot::arc(7));
        b.iter(|| black_box(Arc::clone(&cell.lock())));
    });
    group.bench_function("std::RwLock<Arc>", |b| {
        let cell = StdRwLock::new(Snapshot::arc(7));
        b.iter(|| black_box(Arc::clone(&cell.read().unwrap())));
    });
    group.bench_function("std::Mutex<Arc>", |b| {
        let cell = StdMutex::new(Snapshot::arc(7));
        b.iter(|| black_box(Arc::clone(&cell.lock().unwrap())));
    });
    group.bench_function("z_sync::Lock<Arc>", |b| {
        let cell = ZLock::new(Snapshot::arc(7));
        b.iter(|| black_box(Arc::clone(&cell.read())));
    });

    group.finish();
}

// ============================================================================
// 2. Uncontended store: publish a new snapshot on a single thread. `AtomicArc::store` waits out
//    in-flight readers before reclaiming the old value; with no readers that check is one load.
// ============================================================================
fn bench_uncontended_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Uncontended_Store");

    group.bench_function("z_sync::AtomicArc", |b| {
        let cell = AtomicArc::new(Snapshot::arc(0));
        b.iter(|| cell.store(Snapshot::arc(black_box(9))));
    });
    group.bench_function("arc_swap::ArcSwap", |b| {
        let cell = ArcSwap::from(Snapshot::arc(0));
        b.iter(|| cell.store(Snapshot::arc(black_box(9))));
    });
    group.bench_function("parking_lot::RwLock<Arc>", |b| {
        let cell = PlRwLock::new(Snapshot::arc(0));
        b.iter(|| *cell.write() = Snapshot::arc(black_box(9)));
    });
    group.bench_function("parking_lot::Mutex<Arc>", |b| {
        let cell = PlMutex::new(Snapshot::arc(0));
        b.iter(|| *cell.lock() = Snapshot::arc(black_box(9)));
    });
    group.bench_function("std::RwLock<Arc>", |b| {
        let cell = StdRwLock::new(Snapshot::arc(0));
        b.iter(|| *cell.write().unwrap() = Snapshot::arc(black_box(9)));
    });
    group.bench_function("std::Mutex<Arc>", |b| {
        let cell = StdMutex::new(Snapshot::arc(0));
        b.iter(|| *cell.lock().unwrap() = Snapshot::arc(black_box(9)));
    });
    group.bench_function("z_sync::Lock<Arc>", |b| {
        let cell = ZLock::new(Snapshot::arc(0));
        b.iter(|| *cell.write() = Snapshot::arc(black_box(9)));
    });

    group.finish();
}

// ============================================================================
// 3. Read-only under contention: THREADS loaders, no writer. Nothing here is blocked by anything
//    else, so this isolates the cache traffic of the read path itself: `AtomicArc` readers RMW a
//    shared `reading` counter, `ArcSwap` readers use per-thread debt slots, lock readers RMW a
//    shared reader count.
// ============================================================================
fn bench_readonly_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_ReadOnly_Contended");

    bench_contended!(
        group,
        "z_sync::AtomicArc",
        READ_OPS,
        AtomicArc::new(Snapshot::arc(3)),
        |_i, c| { black_box(c.load()) }
    );
    bench_contended!(
        group,
        "arc_swap::ArcSwap (load_full)",
        READ_OPS,
        ArcSwap::from(Snapshot::arc(3)),
        |_i, c| { black_box(c.load_full()) }
    );
    bench_contended!(
        group,
        "arc_swap::ArcSwap (load, borrowed)",
        READ_OPS,
        ArcSwap::from(Snapshot::arc(3)),
        |_i, c| { black_box(c.load()) }
    );
    bench_contended!(
        group,
        "parking_lot::RwLock<Arc>",
        READ_OPS,
        PlRwLock::new(Snapshot::arc(3)),
        |_i, c| { black_box(Arc::clone(&c.read())) }
    );
    bench_contended!(
        group,
        "parking_lot::Mutex<Arc>",
        READ_OPS,
        PlMutex::new(Snapshot::arc(3)),
        |_i, c| { black_box(Arc::clone(&c.lock())) }
    );
    bench_contended!(
        group,
        "std::RwLock<Arc>",
        READ_OPS,
        StdRwLock::new(Snapshot::arc(3)),
        |_i, c| { black_box(Arc::clone(&c.read().unwrap())) }
    );
    bench_contended!(
        group,
        "z_sync::Lock<Arc>",
        READ_OPS,
        ZLock::new(Snapshot::arc(3)),
        |_i, c| { black_box(Arc::clone(&c.read())) }
    );

    group.finish();
}

// ============================================================================
// 4. Read-heavy under contention: THREADS workers, 90% load / 10% store — the read-mostly mix
//    `AtomicArc` exists for. Readers never block, and a writer waits only for the O(1) windows in
//    which readers are cloning, so neither side can starve the other.
// ============================================================================
fn bench_readheavy_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("4_ReadHeavy_Contended");

    bench_contended!(
        group,
        "z_sync::AtomicArc",
        MIX_OPS,
        AtomicArc::new(Snapshot::arc(1)),
        |i, c| {
            if i % 10 == 0 {
                c.store(Snapshot::arc(i as u64));
            } else {
                black_box(c.load());
            }
        }
    );
    bench_contended!(
        group,
        "arc_swap::ArcSwap (load_full)",
        MIX_OPS,
        ArcSwap::from(Snapshot::arc(1)),
        |i, c| {
            if i % 10 == 0 {
                c.store(Snapshot::arc(i as u64));
            } else {
                black_box(c.load_full());
            }
        }
    );
    bench_contended!(
        group,
        "arc_swap::ArcSwap (load, borrowed)",
        MIX_OPS,
        ArcSwap::from(Snapshot::arc(1)),
        |i, c| {
            if i % 10 == 0 {
                c.store(Snapshot::arc(i as u64));
            } else {
                black_box(c.load());
            }
        }
    );
    bench_contended!(
        group,
        "parking_lot::RwLock<Arc>",
        MIX_OPS,
        PlRwLock::new(Snapshot::arc(1)),
        |i, c| {
            if i % 10 == 0 {
                *c.write() = Snapshot::arc(i as u64);
            } else {
                black_box(Arc::clone(&c.read()));
            }
        }
    );
    bench_contended!(
        group,
        "parking_lot::Mutex<Arc>",
        MIX_OPS,
        PlMutex::new(Snapshot::arc(1)),
        |i, c| {
            if i % 10 == 0 {
                *c.lock() = Snapshot::arc(i as u64);
            } else {
                black_box(Arc::clone(&c.lock()));
            }
        }
    );
    bench_contended!(
        group,
        "std::RwLock<Arc>",
        MIX_OPS,
        StdRwLock::new(Snapshot::arc(1)),
        |i, c| {
            if i % 10 == 0 {
                *c.write().unwrap() = Snapshot::arc(i as u64);
            } else {
                black_box(Arc::clone(&c.read().unwrap()));
            }
        }
    );
    bench_contended!(group, "z_sync::Lock<Arc>", MIX_OPS, ZLock::new(Snapshot::arc(1)), |i, c| {
        if i % 10 == 0 {
            *c.write() = Snapshot::arc(i as u64);
        } else {
            black_box(Arc::clone(&c.read()));
        }
    });

    group.finish();
}

// ============================================================================
// 5. Write-heavy under contention: THREADS workers, 50% load / 50% store. This is the store side's
//    worst case for `AtomicArc`: every store spins until no reader is mid-clone, and with eight
//    threads loading at full rate that window is rarely empty. Included precisely because it is
//    where the design is expected to look worst.
// ============================================================================
fn bench_writeheavy_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("5_WriteHeavy_Contended");

    bench_contended!(
        group,
        "z_sync::AtomicArc",
        MIX_OPS,
        AtomicArc::new(Snapshot::arc(1)),
        |i, c| {
            if i % 2 == 0 {
                c.store(Snapshot::arc(i as u64));
            } else {
                black_box(c.load());
            }
        }
    );
    bench_contended!(
        group,
        "arc_swap::ArcSwap (load_full)",
        MIX_OPS,
        ArcSwap::from(Snapshot::arc(1)),
        |i, c| {
            if i % 2 == 0 {
                c.store(Snapshot::arc(i as u64));
            } else {
                black_box(c.load_full());
            }
        }
    );
    bench_contended!(
        group,
        "parking_lot::RwLock<Arc>",
        MIX_OPS,
        PlRwLock::new(Snapshot::arc(1)),
        |i, c| {
            if i % 2 == 0 {
                *c.write() = Snapshot::arc(i as u64);
            } else {
                black_box(Arc::clone(&c.read()));
            }
        }
    );
    bench_contended!(
        group,
        "parking_lot::Mutex<Arc>",
        MIX_OPS,
        PlMutex::new(Snapshot::arc(1)),
        |i, c| {
            if i % 2 == 0 {
                *c.lock() = Snapshot::arc(i as u64);
            } else {
                black_box(Arc::clone(&c.lock()));
            }
        }
    );
    bench_contended!(
        group,
        "std::RwLock<Arc>",
        MIX_OPS,
        StdRwLock::new(Snapshot::arc(1)),
        |i, c| {
            if i % 2 == 0 {
                *c.write().unwrap() = Snapshot::arc(i as u64);
            } else {
                black_box(Arc::clone(&c.read().unwrap()));
            }
        }
    );
    bench_contended!(group, "z_sync::Lock<Arc>", MIX_OPS, ZLock::new(Snapshot::arc(1)), |i, c| {
        if i % 2 == 0 {
            *c.write() = Snapshot::arc(i as u64);
        } else {
            black_box(Arc::clone(&c.read()));
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_uncontended_load,
    bench_uncontended_store,
    bench_readonly_contended,
    bench_readheavy_contended,
    bench_writeheavy_contended
);
criterion_main!(benches);
