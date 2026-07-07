# z-sync

Extremely optimised synchronisation primitives for Rust that work in **both async and blocking** code, designed to be as
fast or faster than the established alternatives (`std`, `parking_lot`, `tokio`, `event_listener`).

A single `Lock` replaces `Mutex`, `RwLock`, `tokio::Mutex`, and `tokio::RwLock` — the same instance can be locked from a
blocking thread and awaited from an async task, and the two coordinate correctly with each other.

```rust
use z_sync::Lock32;

let lock = Lock32::new(0u64);

// Blocking
* lock.write() += 1;
let v = * lock.read();

// Async — same lock
async fn bump(lock: &Lock32<u64>) {
    *lock.write_async().await += 1;
    let v = *lock.read_async().await;
}
```

## Highlights

- **Unified async + blocking API.** Every general primitive exposes both a blocking method (`read`, `write`, `wait`,
  `notify`) and an async one (`read_async`, `write_async`, `.await`). Blocking waiters and async waiters share the same
  wait state, so mixing them in one program is safe and coordinated.
- **`no_std` by default.** The core is `#![no_std]` (needs `alloc`). Enable the `std` feature for OS-level thread
  parking.
- **Zero fixed overhead when idle.** The async waker queue is heap-allocated lazily — a lock or notify that is only ever
  used from blocking code, or is never contended, never allocates.
- **Adaptive fast paths.** Uncontended operations resolve with a single atomic RMW. Under contention the primitives spin
  with exponential backoff before falling back to parking, and the spin tuning is specialised per architecture (`x86`/
  `x86_64` vs. others).
- **Const constructible.** `Lock::new` and `Notify::new` are `const fn`, so primitives can live in statics without lazy
  initialisation.
- **Pluggable parking strategy.** A `ParkStrategy` type parameter selects how blocked waiters wait — `ParkingLot` (
  default, via `parking_lot_core`) or `Spin` for `no_std` / bare-metal targets.

## Primitives

| Type                                       | Replaces                                       | Async | Blocking |
|--------------------------------------------|------------------------------------------------|:-----:|:--------:|
| [`Lock`](src/lock/mod.rs)                  | `Mutex`, `RwLock`, `tokio::{Mutex, RwLock}`    |   ✓   |    ✓     |
| [`Notify`](src/notify/mod.rs)              | `event_listener::Event`, `tokio::sync::Notify` |   ✓   |    ✓     |
| [`ObservableLock`](src/observable_lock.rs) | watch/observe-on-change patterns               |   ✓   |    ✓     |

### `Lock` — one lock, mutex *and* RwLock semantics

`Lock<T, S, P>` is a readers-writer lock. Use it as a mutex by only ever taking write guards, or as an RwLock by mixing
`read`/`write`.

```rust
use z_sync::Lock32;

let lock = Lock32::new(vec![1, 2, 3]);

// Shared reads
{
let a = lock.read();
let b = lock.read();      // multiple readers coexist
assert_eq !(a.len(), b.len());
}

// Exclusive write
lock.write().push(4);

// Non-blocking attempts
if let Some( mut guard) = lock.try_write() {
guard.push(5);
}

// Guard projection — borrow one field through the guard
let first = z_sync::lock::ReadGuard::map(lock.read(), | v| & v[0]);
assert_eq!(*first, 1);
```

The `16`/`32`/`64` suffix (`Lock16`, `Lock32`, `Lock64`) selects the width of the packed atomic state word. A wider word
supports more simultaneously parked/waking readers and writers before counters saturate; a narrower one keeps the lock
smaller. `Lock32` is a good default. All three share the same API and are aliases over `Lock<T, LockStateU{16,32,64}>`.

### `Notify` — lightweight event signalling

`Notify` is an epoch-based notification primitive: a listener snapshots the current epoch, and completes once `notify`
advances the epoch past that snapshot. This makes the *check → listen → re-check → wait* pattern race-free.

```rust
use std::sync::Arc;
use z_sync::Notify32;

let notify = Arc::new(Notify32::new());

// Blocking waiter
let n = notify.clone();
std::thread::spawn(move | | {
let listener = n.listener();   // snapshot epoch
// ... re-check your condition here ...
listener.wait();               // parks until notified
});

// Async waiter — same Notify
async fn wait(notify: &Notify32) {
    let listener = notify.listener();
    listener.await;
}

notify.notify(1);            // wake one waiter (async wakers first, then parked threads)
notify.notify(usize::MAX);   // wake all
```

Extras:

- `NotifyListener::with_timeout(dur)` → a listener whose blocking `wait()` returns `Err` on timeout (requires `std`).
- `Notify::rc_listener()` → an owned `'static` listener when you hold an `Rc<Notify>`.
- `select_blocking(&mut [listener, …])` → block until *any* of several listeners fires, returning its index.

### `ObservableLock` — a lock that notifies on change

`ObservableLock` wraps a `Lock` and fires a notification every time a write guard is dropped, while caching the latest
value for cheap reads by observers.

```rust
use z_sync::ObservableLock;

let obs: ObservableLock<u32> = ObservableLock::new(0u32);

// Observer registers interest
let listener = obs.observe();

// Writer mutates and drops the guard → notification fires
* obs.write() = 42;

// Observer wakes and reads the latest published value
listener.wait();
assert_eq!(*obs.latest_value(), 42);
```

Any `Lock` can be upgraded in place with `lock.into_observable()`.

## Features

| Feature    | Default | Effect                                                                                                                                                |
|------------|:-------:|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `std`      |    ✓    | Enables `parking_lot_core` thread parking and makes `ParkingLot` the default `ParkStrategy`. Without it the crate is `no_std` and defaults to `Spin`. |
| `thin-vec` |    ✓    | Uses `thin_vec::ThinVec` for the internal waker queue, shrinking the queue header and allowing const initialisation.                                  |

```toml
[dependencies]
z-sync = "0.1"

# no_std, spin-based parking:
z-sync = { version = "0.1", default-features = false }
```

## Design notes

- **Packed atomic state.** Each primitive keeps all of its coordination state — reader/writer counts, parked-thread
  counts, async-waker counts, and (for `Notify`) an epoch — bit-packed into a single atomic word, so a lock/unlock is
  typically one atomic operation.
- **Lazy, allocation-free waiting.** The async waker queue is a generational, intrusive doubly-linked list over a `Vec`,
  allocated only on first async use. Generation tags on each slot prevent ABA bugs when a slot is popped and recycled,
  guaranteeing no lost or duplicate wakeups.
- **Batched wakeups.** Waking many waiters (e.g. a thundering herd or `notify(usize::MAX)`) drains the queue in
  fixed-size batches using uninitialised stack storage, avoiding per-waker heap traffic and large `memset`s.
- **Correct drop behaviour.** Dropping a future or listener before it completes deregisters its waker and restores the
  waiter counts, so cancelled/`select!`-losing futures never leak wait slots (covered by tests).

## Benchmarks

Criterion benchmarks compare `z-sync` against `std`, `parking_lot`, `tokio`, and `event_listener` across uncontended,
read-only, write-only, read-heavy, write-heavy, thundering-herd, MPSC, ping-pong, and chain workloads.

```sh
cargo bench --bench lock
cargo bench --bench notify
```

See [`benches/lock.rs`](benches/lock.rs) and [`benches/notify.rs`](benches/notify.rs). Run them on your target
hardware — relative performance depends heavily on CPU and contention level.

## Status

Early stage (`0.1.x`) — the API is functional and tested but may still change. Correctness across blocking/async mixing,
cancellation, and counter accounting is exercised by the test suite:

```sh
cargo test
```

## License

MIT — see [LICENSE](LICENSE).
