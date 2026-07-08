# TODO — planned structures

Candidate additions, chosen to fit the library's foundations: the epoch-based `Notify`, the lazy
`WakerQueueLock`, the `ParkStrategy` generic, and the packed-state `U16/U32/U64` pattern. Everything
below should present the same unified async + blocking API as the existing primitives.

## Recommended order

Maximum impact for the current design: **`Semaphore` → `oneshot` → `mpsc` → `Once` / `Barrier`**.
The semaphore and channels are where the unified async + blocking model plus the lazy waker queue give
a genuine edge over `tokio` (async-only) and `crossbeam` / `std` (blocking-only) — no other `no_std`
crate offers a fast *both* in one place.

## Tier 1 — natural fits, high demand, reuse existing infra

- [x] **`Semaphore`** — counting semaphore with `acquire` / `try_acquire` / `acquire_async`, permits
  released on guard drop. Highest-value lever: substrate for connection pools, rate limiters, and
  bounded concurrency. Maps onto packed state (permit count) + `Notify`-style waiter wakeup. Ships with
  the `WakerStorage` (`Boxed`/`Inline`) convention.
- **Channels** — the `Notify` epoch + `WakerQueueLock` is the wakeup half. Each is a standalone,
  allocation-free core (no internal `Arc`); the caller owns it and `split`s borrowed sender/receiver
  halves, so users choose how to share (stack + scoped threads, `static`, their own `Arc`).
    - [x] `oneshot` — single value, one sender → one receiver, blocking + async, drop-close semantics.
      Benched vs `tokio` and `futures`.
    - [x] `watch` — latest value + version, one sender → many receivers (`borrow` / `changed`).
      Supersedes the `ObservableLock` pattern. Benched vs `tokio`.
    - [x] `broadcast` — bounded ring buffer, every receiver sees every message, `Lagged` detection,
      per-slot `Lock` for concurrent reads. Benched vs `tokio` and `async-broadcast`.
    - ~~`mpsc` (bounded + unbounded)~~ — **won't do here.** A separate library builds
      `mpsc` / `mpmc` / `spsc` / `spmc` on top of these primitives.
- [x] **`Once` / `OnceCell` / `Lazy`** — one-time init with both `get_or_init` (blocking) and
  `get_or_init_async`, plus fallible `call_once_try` / `get_or_try_init` (`*_async` too) that leave the
  primitive uninitialised on error so a later call retries (`std::sync::OnceLock` semantics, no poison).
  `const fn` constructors; built on `Once` (`Notify`-backed wait). Fills a `no_std` async gap
  `std::sync::OnceLock` can't cover.
- [x] **`Barrier`** — N-party rendezvous (`wait` / `wait_async`); a packed `(generation, count)` word +
  `Notify` broadcast, reusable across rounds with a single leader per round. One type covering both
  `std::sync::Barrier` and `tokio::sync::Barrier`. Blocking and async parties share a round.
- [ ] **`WaitGroup` / `CountdownLatch`** — Go-style "wait until N units complete." Counter down to zero,
  then wake-all. Nearly free given `Notify`.

## Tier 2 — specialised but on-theme

- [x] **`Condvar`** — a real condition-variable API pairing with `Lock` write guards (`wait(guard)` /
  `wait_async(guard)`, plus `wait_while` / `wait_while_async` and `notify_one` / `notify_all`) instead
  of hand-rolled `Notify` + re-check. The listener is registered before the guard is released, so the
  epoch snapshot-then-recheck model rules out lost wakeups. Blocking and async waiters share one
  condvar. Benched vs `std` and `parking_lot`.
- [x] **Upgradable read guard on `Lock`** — `upgradable_read()` / `try_upgradable_read()` /
  `upgradable_read_async()`, `UpgradableReadGuard::upgrade` / `try_upgrade` / `upgrade_async` /
  `downgrade`, plus `WriteGuard::downgrade` and `downgrade_to_upgradable`, and mapping guards
  (`ReadGuard`/`WriteGuard` `map`/`try_map` + `MappedReadGuard`/`MappedWriteGuard`). Implemented with
  a packed 1-bit upgradable flag (one reader bit) — benchmarked to add 0% to the read/write hot
  paths. Benched vs `parking_lot` (blocking) and `async-std`/`async_lock` (async).
- [ ] **`SeqLock`** — lock-free reads for small `Copy`, read-mostly data (config snapshots, counters).
  Complements `Lock` where readers must never block writers.
- [ ] **`AtomicWaker`** — single-waiter waker cell (cf. `futures::task::AtomicWaker`); the degenerate
  1-waiter case of `WakerQueueLock`. A lighter building block for oneshot / single-consumer structures.

## Tier 3 — utilities worth exposing

- [ ] **`Backoff` / `SpinWait`** — promote the adaptive, arch-tuned exponential-backoff spin logic in
  `src/lock/mod.rs` to a public utility (cf. `crossbeam_utils::Backoff`) so users can build their own
  primitives on the same tuning.
- [ ] **`ReentrantLock`** — reentrant mutex for recursive acquisition.
- [ ] **`ShardedLock` / striped locks** — per-shard `Lock` array for extreme read concurrency.
