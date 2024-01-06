use tch::{nn::VarStore, Device};
use utils::{a3c::A3C, env, panic_hook};

const ITERATION_COUNT: usize = 10_000;
const EVAL_INTERVAL: usize = 1000;

fn main() {
    panic_hook::init();

    let env = env::MyVeryOwnStick::new();

    let device = Device::Cpu;
    let vs = VarStore::new(device);

    let mut a3c = A3C::builder()
        .vs(vs)
        .env(env)
        .fc_layers(&[100, 100])
        .gamma(0.99)
        .learning_rate(1e-3)
        .sync_interval(100)
        .num_threads(8)
        .build();

    for i in 0..(ITERATION_COUNT / EVAL_INTERVAL) {
        a3c.train(EVAL_INTERVAL);
        let avg_return = a3c.evaluate_avg_return();

        println!(
            "Step: {}, Average return: {}",
            (i + 1) * EVAL_INTERVAL,
            avg_return
        );
    }
}
