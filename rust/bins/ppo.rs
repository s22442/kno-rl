use tch::{nn::VarStore, Device};
use utils::{env, panic_hook, ppo::PPO};

const NUM_EPOCHS: usize = 20;

const EVAL_INTERVAL: usize = 1;

fn main() {
    panic_hook::init();

    let env = env::MyVeryOwnStick::new();
    let device = Device::Cpu;

    let mut ppo = PPO::builder()
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
        .build();

    for i in 0..(NUM_EPOCHS / EVAL_INTERVAL) {
        ppo.train(EVAL_INTERVAL);
        let avg_return = ppo.evaluate_avg_return();
        println!(
            "Epoch: {}, Average return: {}",
            (i + 1) * EVAL_INTERVAL,
            avg_return
        );
    }
}
