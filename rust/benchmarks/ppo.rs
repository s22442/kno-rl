use criterion::{criterion_group, criterion_main, BatchSize, Criterion, PlottingBackend};
use tch::{nn::VarStore, Device};

const MAX_NUM_EPOCHS: usize = 100;
const EVAL_INTERVAL: usize = 2;

const TARGET_AVG_RETURN: f64 = 475.0;

fn a3c_cpu(c: &mut Criterion) {
    use utils::{env, ppo::PPO};

    c.bench_function("PPO/CPU", |b| {
        b.iter_batched(
            || {
                let env = env::MyVeryOwnStick::new();
                let device = Device::Cpu;

                PPO::builder()
                    .actor_vs(VarStore::new(device))
                    .critic_vs(VarStore::new(device))
                    .env(env)
                    .fc_layers(&[100, 100])
                    .gamma(0.99)
                    .lmbda(0.95)
                    .clip_epsilon(0.2)
                    .kl_epsilon(5e-3)
                    .actor_learning_rate(3e-4)
                    .critic_learning_rate(1e-3)
                    .num_steps_per_epoch(10_000)
                    .batch_size(500)
                    .actor_train_num_batches(50)
                    .critic_train_num_batches(50)
                    .num_threads(8)
                    .build()
            },
            |mut ppo| {
                for i in 0..(MAX_NUM_EPOCHS / EVAL_INTERVAL) {
                    ppo.train(EVAL_INTERVAL);

                    let avg_return = ppo.evaluate_avg_return();

                    if avg_return >= TARGET_AVG_RETURN {
                        break;
                    }

                    assert!(
                        (i + 1) * EVAL_INTERVAL != MAX_NUM_EPOCHS,
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
        .sample_size(200)
        .plotting_backend(PlottingBackend::None);
    targets = a3c_cpu
}
criterion_main!(benches);
