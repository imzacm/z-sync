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

- [ ] **`Semaphore`** — counting semaphore with `acquire` / `try_acquire` / `acquire_async`, permits
  released on guard drop. Highest-value lever: substrate for connection pools, rate limiters, and
  bounded concurrency. Maps onto packed state (permit count) + `Notify`-style waiter wakeup. A `Mutex`
  can become a trivial `Semaphore<1>` special case if we want to unify.
- [ ] **Channels** — the `Notify` epoch + `WakerQueueLock` is essentially the wakeup half already.
  Suggested order:
    - [ ] `oneshot` — simplest, huge demand (task result / cancellation). Pairs with the drop-cleanup story.
    - [ ] `mpsc` (bounded + unbounded) — bounded reuses `Semaphore` for backpressure.
    - [ ] `broadcast` / `watch` — `watch` generalises `ObservableLock` (versioned value + multi-observer);
      consider refactoring `ObservableLock` onto a shared `watch` core.
- [ ] **`Once` / `OnceCell` / `Lazy`** — one-time init with both `get_or_init` (blocking) and
  `get_or_init_async`. `const fn` constructors + `Notify` epoch make the "wait for the other initialiser"
  path cheap. Fills a `no_std` async gap `std::sync::OnceLock` can't cover.
- [ ] **`Barrier`** — N-party rendezvous (`wait` / `wait_async`); a counter + `Notify` broadcast. One type
  covering both `std::sync::Barrier` and `tokio::sync::Barrier`.
- [ ] **`WaitGroup` / `CountdownLatch`** — Go-style "wait until N units complete." Counter down to zero,
  then wake-all. Nearly free given `Notify`.

## Tier 2 — specialised but on-theme

- [ ] **`Condvar`** — a real condition-variable API pairing with `Lock` guards (`wait(guard)` /
  `wait_async(guard)`) instead of hand-rolled `Notify` + re-check. The listener's snapshot-then-recheck
  epoch model already avoids lost wakeups — the hard part of a correct condvar.
- [ ] **Upgradable read guard on `Lock`** — `upgradable_read()` → `upgrade()` (parking_lot parity).
  Reader/writer counts are already in the packed word; mostly a new guard type + one state transition.
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
