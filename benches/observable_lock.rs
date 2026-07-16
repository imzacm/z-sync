use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar as StdCondvar, Mutex as StdMutex};

use arc_swap::ArcSwap;
use criterion::{Bencher, Criterion, criterion_group, criterion_main};
use event_listener::{Event, Listener};
use parking_lot::{Condvar as PlCondvar, Mutex as PlMutex};
use z_sync::ObservableLock;
use z_sync::channels::watch::Watch;

// A config-snapshot payload: mutated in place under the write lock, then published whole to
// readers. Only ever written and `black_box`ed, never inspected per-field.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct Snapshot {
    generation: u64,
    a: u64,
    b: u64,
    c: u64,
}

impl Snapshot {
    #[inline]
    fn of(n: u64) -> Self {
        Self { generation: n, a: n, b: n, c: n }
    }
}

const THREADS: usize = 8;
const OBSERVERS: usize = 8;
const MIX_OPS: usize = 1000;

// `ObservableLock` bundles three things — a lock to mutate under, a published snapshot readers can
// take without blocking, and a notification on change. No single alternative does all three, so the
// comparisons below are against the combinations people reach for instead:
//
//   - `tokio::sync::watch`  — the closest ready-made equivalent (send / borrow / changed).
//   - `ArcSwap` + `event_listener::Event` — the same design hand-assembled, minus the write lock.
//   - `Mutex` + `Condvar` (parking_lot and std) — the classic shape, where readers block on writers.
//
// A `Mutex`, not an `RwLock`, for the condvar pairs: `Condvar::wait` needs a mutex guard, which is
// exactly the constraint that makes the classic shape block readers in the first place.

/// Round bookkeeping shared by the fan-out contenders in group 4.
///
/// Observers loop over rounds, each waiting for generation `r` to be published and acknowledging
/// it; the writer publishes round `r` and waits for all OBSERVERS acknowledgements before timing
/// the next. Acknowledgements are counted cumulatively rather than reset per round, so a fast
/// observer racing ahead into round `r + 1` cannot be miscounted against round `r`.
struct FanoutState {
    /// Acknowledgements across all rounds; round `r` is complete at `r * OBSERVERS`.
    acks: AtomicUsize,
    shutdown: AtomicBool,
}

impl FanoutState {
    fn new() -> Self {
        Self { acks: AtomicUsize::new(0), shutdown: AtomicBool::new(false) }
    }

    /// The observer side: block in `await_generation(r)` for each round `r` in turn, acknowledging
    /// each, until the writer shuts the round loop down.
    fn observe_rounds(&self, mut await_generation: impl FnMut(u64)) {
        let mut round = 0u64;
        loop {
            round += 1;
            await_generation(round);
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }
            self.acks.fetch_add(1, Ordering::Release);
        }
    }

    /// The writer side: one timed round is `publish(r)` plus the wait for every observer to wake
    /// and acknowledge. Releases the observers once criterion is done.
    fn drive(&self, b: &mut Bencher<'_>, mut publish: impl FnMut(u64)) {
        let mut round = 0u64;
        b.iter(|| {
            round += 1;
            publish(round);
            let complete = round as usize * OBSERVERS;
            while self.acks.load(Ordering::Acquire) < complete {
                std::hint::spin_loop();
            }
        });
        // Every observer is waiting on `generation >= r` for some round `r`, so a final generation
        // no round can reach satisfies all of them at once and lets them see the shutdown flag.
        self.shutdown.store(true, Ordering::Release);
        publish(u64::MAX);
    }
}

// Spawns THREADS workers that each run `$body` `$ops` times; `$i` is the op index, `$l` the shared
// (Arc) handle.
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
// 1. Uncontended read of the latest value on a single thread.
//
// `latest_value` returns a shared-pointer clone of the last published snapshot without touching the
// lock; `read` is listed beside it as the same type's blocking path, for the cost of the guard.
// ============================================================================
fn bench_uncontended_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_Uncontended_Read");

    group.bench_function("z_sync::ObservableLock::latest_value", |b| {
        let lock = ObservableLock::<Snapshot>::new(Snapshot::of(7));
        b.iter(|| black_box(lock.latest_value()));
    });
    group.bench_function("z_sync::ObservableLock::read", |b| {
        let lock = ObservableLock::<Snapshot>::new(Snapshot::of(7));
        b.iter(|| black_box(lock.read().generation));
    });
    group.bench_function("z_sync::Watch", |b| {
        let channel = Arc::new(Watch::new(Snapshot::of(7)));
        let (_tx, rx) = channel.arc_split();
        b.iter(|| black_box(rx.borrow().generation));
    });
    group.bench_function("tokio::sync::watch", |b| {
        let (_tx, rx) = tokio::sync::watch::channel(Snapshot::of(7));
        b.iter(|| black_box(rx.borrow().generation));
    });
    group.bench_function("ArcSwap + event_listener", |b| {
        let value = ArcSwap::from_pointee(Snapshot::of(7));
        b.iter(|| black_box(value.load_full()));
    });
    group.bench_function("parking_lot::Mutex + Condvar", |b| {
        let value = PlMutex::new(Snapshot::of(7));
        b.iter(|| black_box(value.lock().generation));
    });
    group.bench_function("std::Mutex + Condvar", |b| {
        let value = StdMutex::new(Snapshot::of(7));
        b.iter(|| black_box(value.lock().unwrap().generation));
    });

    group.finish();
}

// ============================================================================
// 2. Uncontended write with nobody observing: what a publish costs when it wakes no one.
//
// `ObservableLock`'s guard drop does the most work of any contender here — it clones the value out,
// allocates an `Arc` for it, publishes that, and then notifies. That is the price of `latest_value`
// being lock-free in group 1, and it is charged on every write whether or not anyone reads.
// ============================================================================
fn bench_uncontended_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_Uncontended_Write");

    group.bench_function("z_sync::ObservableLock", |b| {
        let lock = ObservableLock::<Snapshot>::new(Snapshot::of(0));
        b.iter(|| *lock.write() = Snapshot::of(black_box(9)));
    });
    group.bench_function("z_sync::Watch", |b| {
        let channel = Arc::new(Watch::new(Snapshot::of(0)));
        let (tx, _rx) = channel.arc_split();
        b.iter(|| tx.send(Snapshot::of(black_box(9))).unwrap());
    });
    group.bench_function("tokio::sync::watch", |b| {
        let (tx, _rx) = tokio::sync::watch::channel(Snapshot::of(0));
        b.iter(|| tx.send(Snapshot::of(black_box(9))).unwrap());
    });
    group.bench_function("ArcSwap + event_listener", |b| {
        let value = ArcSwap::from_pointee(Snapshot::of(0));
        let event = Event::new();
        b.iter(|| {
            value.store(Arc::new(Snapshot::of(black_box(9))));
            event.notify(usize::MAX);
        });
    });
    group.bench_function("parking_lot::Mutex + Condvar", |b| {
        let value = PlMutex::new(Snapshot::of(0));
        let cv = PlCondvar::new();
        b.iter(|| {
            *value.lock() = Snapshot::of(black_box(9));
            cv.notify_all();
        });
    });
    group.bench_function("std::Mutex + Condvar", |b| {
        let value = StdMutex::new(Snapshot::of(0));
        let cv = StdCondvar::new();
        b.iter(|| {
            *value.lock().unwrap() = Snapshot::of(black_box(9));
            cv.notify_all();
        });
    });

    group.finish();
}

// ============================================================================
// 3. Read-heavy under contention: THREADS workers, 90% read of the latest value / 10% write. The
//    read-mostly config-snapshot workload the type is for, and the one where taking the snapshot
//    off the lock should pay: `latest_value` readers never queue behind a writer.
// ============================================================================
fn bench_readheavy_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_ReadHeavy_Contended");

    bench_contended!(
        group,
        "z_sync::ObservableLock",
        MIX_OPS,
        ObservableLock::<Snapshot>::new(Snapshot::of(1)),
        |i, l| {
            if i % 10 == 0 {
                *l.write() = Snapshot::of(i as u64);
            } else {
                black_box(l.latest_value());
            }
        }
    );
    bench_contended!(
        group,
        "z_sync::ObservableLock (read guard)",
        MIX_OPS,
        ObservableLock::<Snapshot>::new(Snapshot::of(1)),
        |i, l| {
            if i % 10 == 0 {
                *l.write() = Snapshot::of(i as u64);
            } else {
                black_box(l.read().generation);
            }
        }
    );
    bench_contended!(
        group,
        "tokio::sync::watch",
        MIX_OPS,
        tokio::sync::watch::channel(Snapshot::of(1)),
        |i, l| {
            if i % 10 == 0 {
                l.0.send(Snapshot::of(i as u64)).unwrap();
            } else {
                black_box(l.1.borrow().generation);
            }
        }
    );
    bench_contended!(
        group,
        "ArcSwap + event_listener",
        MIX_OPS,
        (ArcSwap::from_pointee(Snapshot::of(1)), Event::new()),
        |i, l| {
            if i % 10 == 0 {
                l.0.store(Arc::new(Snapshot::of(i as u64)));
                l.1.notify(usize::MAX);
            } else {
                black_box(l.0.load_full());
            }
        }
    );
    bench_contended!(
        group,
        "parking_lot::Mutex + Condvar",
        MIX_OPS,
        (PlMutex::new(Snapshot::of(1)), PlCondvar::new()),
        |i, l| {
            if i % 10 == 0 {
                *l.0.lock() = Snapshot::of(i as u64);
                l.1.notify_all();
            } else {
                black_box(l.0.lock().generation);
            }
        }
    );
    bench_contended!(
        group,
        "std::Mutex + Condvar",
        MIX_OPS,
        (StdMutex::new(Snapshot::of(1)), StdCondvar::new()),
        |i, l| {
            if i % 10 == 0 {
                *l.0.lock().unwrap() = Snapshot::of(i as u64);
                l.1.notify_all();
            } else {
                black_box(l.0.lock().unwrap().generation);
            }
        }
    );

    group.finish();
}

// ============================================================================
// 4. Change fan-out: OBSERVERS threads parked on a change, one writer publishes, all wake and read
//    the new value. One iteration is one such round.
//
//    The observer threads are spawned once per contender and live across all iterations — spawning
//    them per round costs ~35µs a thread and would bury the wake-up being measured. Each round the
//    writer publishes generation `r` and then waits for all OBSERVERS to acknowledge, so a round
//    times a full publish-to-every-observer-awake cycle.
//
//    Observers wait on the *predicate* `generation >= r`, never on a bare edge: an observer that is
//    still between rounds when the writer publishes sees the new generation and proceeds instead of
//    parking forever. Shutdown publishes `u64::MAX`, which satisfies every observer's predicate and
//    releases them all.
//
//    `tokio::sync::watch` is absent: its `changed` is async-only, and driving a runtime per round
//    would measure the runtime, not the fan-out.
// ============================================================================
fn bench_fanout_wake(c: &mut Criterion) {
    let mut group = c.benchmark_group("4_Fanout_Wake_8");

    group.bench_function("z_sync::ObservableLock", |b| {
        let lock = ObservableLock::<Snapshot>::new(Snapshot::of(0));
        let state = FanoutState::new();
        std::thread::scope(|s| {
            for _ in 0..OBSERVERS {
                s.spawn(|| {
                    state.observe_rounds(|target| {
                        // Register before testing the predicate: a publish landing in between
                        // notifies the listener rather than being missed.
                        loop {
                            let listener = lock.observe();
                            if lock.latest_value().generation >= target {
                                break;
                            }
                            listener.wait();
                        }
                        black_box(lock.latest_value().generation);
                    });
                });
            }
            state.drive(b, |generation| *lock.write() = Snapshot::of(generation));
        });
    });

    group.bench_function("z_sync::Watch", |b| {
        let channel = Arc::new(Watch::new(Snapshot::of(0)));
        let (tx, rx0) = channel.arc_split();
        let state = FanoutState::new();
        std::thread::scope(|s| {
            for _ in 0..OBSERVERS {
                let mut rx = rx0.clone();
                let state = &state;
                s.spawn(move || {
                    // `changed` tracks the version it last saw, so it needs no target of its own.
                    state.observe_rounds(|_target| {
                        rx.changed().unwrap();
                        black_box(rx.borrow().generation);
                    });
                });
            }
            state.drive(b, |generation| tx.send(Snapshot::of(generation)).unwrap());
        });
    });

    group.bench_function("ArcSwap + event_listener", |b| {
        let value = ArcSwap::from_pointee(Snapshot::of(0));
        let event = Event::new();
        let state = FanoutState::new();
        std::thread::scope(|s| {
            for _ in 0..OBSERVERS {
                s.spawn(|| {
                    state.observe_rounds(|target| {
                        loop {
                            let listener = event.listen();
                            if value.load().generation >= target {
                                break;
                            }
                            listener.wait();
                        }
                        black_box(value.load_full().generation);
                    });
                });
            }
            state.drive(b, |generation| {
                value.store(Arc::new(Snapshot::of(generation)));
                event.notify(usize::MAX);
            });
        });
    });

    group.bench_function("parking_lot::Mutex + Condvar", |b| {
        let value = PlMutex::new(Snapshot::of(0));
        let cv = PlCondvar::new();
        let state = FanoutState::new();
        std::thread::scope(|s| {
            for _ in 0..OBSERVERS {
                s.spawn(|| {
                    state.observe_rounds(|target| {
                        let mut guard = value.lock();
                        cv.wait_while(&mut guard, |v| v.generation < target);
                        black_box(guard.generation);
                    });
                });
            }
            state.drive(b, |generation| {
                *value.lock() = Snapshot::of(generation);
                cv.notify_all();
            });
        });
    });

    group.bench_function("std::Mutex + Condvar", |b| {
        let value = StdMutex::new(Snapshot::of(0));
        let cv = StdCondvar::new();
        let state = FanoutState::new();
        std::thread::scope(|s| {
            for _ in 0..OBSERVERS {
                s.spawn(|| {
                    state.observe_rounds(|target| {
                        let guard = value.lock().unwrap();
                        let guard = cv.wait_while(guard, |v| v.generation < target).unwrap();
                        black_box(guard.generation);
                    });
                });
            }
            state.drive(b, |generation| {
                *value.lock().unwrap() = Snapshot::of(generation);
                cv.notify_all();
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_uncontended_read,
    bench_uncontended_write,
    bench_readheavy_contended,
    bench_fanout_wake
);
criterion_main!(benches);
