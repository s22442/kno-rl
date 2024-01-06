use criterion::{criterion_group, criterion_main, BatchSize, Criterion, PlottingBackend};
use tch::{nn::VarStore, Device};

const MAX_ITERATION_COUNT: usize = 500_000;
const EVAL_INTERVAL: usize = 500;

const TARGET_AVG_RETURN: f64 = 475.0;

fn a3c_cpu(c: &mut Criterion) {
    use utils::{a3c::A3C, env};

    c.bench_function("A3C/CPU", |b| {
        b.iter_batched(
            || {
                let env = env::MyVeryOwnStick::new();

                let vs = VarStore::new(Device::Cpu);

                A3C::builder()
                    .vs(vs)
                    .env(env)
                    .fc_layers(&[100, 100])
                    .gamma(0.99)
                    .learning_rate(1e-3)
                    .sync_interval(100)
                    .num_threads(8)
                    .build()
            },
            |mut a3c| {
                for i in 0..(MAX_ITERATION_COUNT / EVAL_INTERVAL) {
                    a3c.train(EVAL_INTERVAL);

                    let avg_return = a3c.evaluate_avg_return();

                    if avg_return >= TARGET_AVG_RETURN {
                        break;
                    }

                    assert!(
                        (i + 1) * EVAL_INTERVAL != MAX_ITERATION_COUNT,
                        "Failed to reach target average return"
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .plotting_backend(PlottingBackend::None);
    targets = a3c_cpu
}
criterion_main!(benches);
