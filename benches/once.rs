use std::hint::black_box;
use std::sync::{LazyLock, Once as StdOnce, OnceLock};

use criterion::{Criterion, criterion_group, criterion_main};
use z_sync::{Lazy, Once, OnceCell};

/// Read an already-initialised cell (the common hot path).
fn bench_get_initialized(c: &mut Criterion) {
    let mut group = c.benchmark_group("oncecell_get_initialized");

    let cell = OnceCell::new();
    cell.set(42u64).unwrap();
    group.bench_function("z_sync", |b| b.iter(|| black_box(*cell.get().unwrap())));

    let cell = OnceLock::new();
    cell.set(42u64).unwrap();
    group.bench_function("std::OnceLock", |b| b.iter(|| black_box(*cell.get().unwrap())));

    let cell = once_cell::sync::OnceCell::new();
    cell.set(42u64).unwrap();
    group.bench_function("once_cell", |b| b.iter(|| black_box(*cell.get().unwrap())));

    group.finish();
}

/// `get_or_init` on an already-initialised cell (init closure not run).
fn bench_get_or_init_hot(c: &mut Criterion) {
    let mut group = c.benchmark_group("oncecell_get_or_init_hot");

    let cell = OnceCell::new();
    cell.set(42u64).unwrap();
    group.bench_function("z_sync", |b| b.iter(|| black_box(*cell.get_or_init(|| 0))));

    let cell = OnceLock::new();
    cell.set(42u64).unwrap();
    group.bench_function("std::OnceLock", |b| b.iter(|| black_box(*cell.get_or_init(|| 0))));

    let cell = once_cell::sync::OnceCell::new();
    cell.set(42u64).unwrap();
    group.bench_function("once_cell", |b| b.iter(|| black_box(*cell.get_or_init(|| 0))));

    group.finish();
}

/// Fresh cell + one `get_or_init` (construction + first, uncontended initialisation).
fn bench_first_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("oncecell_first_init");

    group.bench_function("z_sync", |b| {
        b.iter(|| {
            let cell = OnceCell::new();
            black_box(*cell.get_or_init(|| black_box(42u64)))
        });
    });
    group.bench_function("std::OnceLock", |b| {
        b.iter(|| {
            let cell = OnceLock::new();
            black_box(*cell.get_or_init(|| black_box(42u64)))
        });
    });
    group.bench_function("once_cell", |b| {
        b.iter(|| {
            let cell = once_cell::sync::OnceCell::new();
            black_box(*cell.get_or_init(|| black_box(42u64)))
        });
    });

    group.finish();
}

/// `call_once` on an already-completed `Once` (the hot path).
fn bench_call_once_hot(c: &mut Criterion) {
    let mut group = c.benchmark_group("once_call_once_hot");

    let once = Once::new();
    once.call_once(|| {});
    group.bench_function("z_sync", |b| b.iter(|| once.call_once(|| black_box(()))));

    let once = StdOnce::new();
    once.call_once(|| {});
    group.bench_function("std::Once", |b| b.iter(|| once.call_once(|| black_box(()))));

    group.finish();
}

/// Deref an already-initialised `Lazy`.
fn bench_lazy_deref(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_deref");

    let lazy: Lazy<u64> = Lazy::new(|| 42);
    let _ = *lazy;
    group.bench_function("z_sync", |b| b.iter(|| black_box(*lazy)));

    let lazy: once_cell::sync::Lazy<u64> = once_cell::sync::Lazy::new(|| 42);
    let _ = *lazy;
    group.bench_function("once_cell", |b| b.iter(|| black_box(*lazy)));

    let lazy: LazyLock<u64> = LazyLock::new(|| 42);
    let _ = *lazy;
    group.bench_function("std::LazyLock", |b| b.iter(|| black_box(*lazy)));

    group.finish();
}

/// Async `get_or_init` on an already-initialised cell (init future not run).
fn bench_get_or_init_async_hot(c: &mut Criterion) {
    let mut group = c.benchmark_group("oncecell_get_or_init_async_hot");
    let rt = tokio::runtime::Runtime::new().unwrap();

    let cell = OnceCell::new();
    cell.set(42u64).unwrap();
    group.bench_function("z_sync", |b| {
        b.to_async(&rt)
            .iter(|| async { black_box(*cell.get_or_init_async(|| async { 0 }).await) });
    });

    let cell = tokio::sync::OnceCell::new();
    cell.set(42u64).unwrap();
    group.bench_function("tokio", |b| {
        b.to_async(&rt)
            .iter(|| async { black_box(*cell.get_or_init(|| async { 0u64 }).await) });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_get_initialized,
    bench_get_or_init_hot,
    bench_first_init,
    bench_call_once_hot,
    bench_lazy_deref,
    bench_get_or_init_async_hot
);
criterion_main!(benches);
