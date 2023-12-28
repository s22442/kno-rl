use criterion::{criterion_group, criterion_main, BatchSize, Criterion, PlottingBackend};
use tch::{nn::VarStore, Device};

const MAX_ITERATION_COUNT: usize = 100_000;
const EVAL_INTERVAL: usize = 500;

const TARGET_AVG_RETURN: f32 = 475.0;

fn a2c_cpu(c: &mut Criterion) {
    use utils::{a2c, env};

    c.bench_function("A2C/CPU", |b| {
        b.iter_batched(
            || {
                let env = env::MyVeryOwnStick::new();

                let vs = VarStore::new(Device::Cpu);

                a2c::Builder::init(&vs)
                    .set_env(env)
                    .set_fc_layers(&[100, 100])
                    .set_gamma(0.99)
                    .set_learning_rate(1e-3)
                    .set_sync_interval(100)
                    .build()
            },
            |mut a2c| {
                for i in 0..(MAX_ITERATION_COUNT / EVAL_INTERVAL) {
                    a2c.train(EVAL_INTERVAL);

                    let avg_return = a2c.evaluate_avg_return();

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
        .sample_size(20)
        .plotting_backend(PlottingBackend::None);
    targets = a2c_cpu
}
criterion_main!(benches);
