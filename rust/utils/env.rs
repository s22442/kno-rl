use tch::Tensor;

pub trait Env: Clone + Send + 'static {
    fn observation_space() -> usize;
    fn action_space() -> usize;
    fn observation(&self) -> Tensor;
    fn episode_ended(&self) -> bool;
    fn reset(&mut self);
    fn step(&mut self, action: u32) -> f64;
}

mod my_very_own_stick {
    pub use super::Env as EnvTrait;
    use rand::Rng;
    use std::f64::consts::PI;
    use tch::Tensor;

    #[derive(Clone, Debug)]
    struct State(f64, f64, f64, f64);

    impl Default for State {
        fn default() -> Self {
            let mut rng = rand::thread_rng();
            Self(
                rng.gen_range(-0.05..0.05),
                rng.gen_range(-0.05..0.05),
                rng.gen_range(-0.05..0.05),
                rng.gen_range(-0.05..0.05),
            )
        }
    }

    #[derive(Clone, Debug)]
    pub struct Env {
        current_step: u16,
        episode_ended: bool,
        state: State,
    }

    const MAX_STEPS: u16 = 500;
    const REWARD_PER_STEP: f64 = 1.0;
    const GRAVITY: f64 = 9.8;
    const CART_MASS: f64 = 1.0;
    const POLE_MASS: f64 = 0.1;
    const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;
    const POLE_LENGTH: f64 = 0.5;
    const POLE_MASS_LENGTH: f64 = POLE_MASS * POLE_LENGTH;
    const FORCE_MAG: f64 = 10.0;
    const TAU: f64 = 0.02;
    const THETA_THRESHOLD_RADIANS: f64 = 12.0 * 2.0 * PI / 360.0;
    const X_THRESHOLD: f64 = 2.4;

    impl Env {
        #[must_use]
        pub fn new() -> Self {
            Self {
                current_step: 0,
                episode_ended: false,
                state: State::default(),
            }
        }
    }

    impl EnvTrait for Env {
        #[must_use]
        fn observation_space() -> usize {
            4
        }

        #[must_use]
        fn action_space() -> usize {
            2
        }

        fn observation(&self) -> Tensor {
            Tensor::from_slice(&[self.state.0, self.state.1, self.state.2, self.state.3])
        }

        #[must_use]
        fn episode_ended(&self) -> bool {
            self.episode_ended
        }

        fn reset(&mut self) {
            self.state = State::default();
            self.current_step = 0;
            self.episode_ended = false;
        }

        fn step(&mut self, action: u32) -> f64 {
            self.current_step += 1;

            if self.current_step >= MAX_STEPS {
                self.episode_ended = true;
                return REWARD_PER_STEP;
            }

            let State(mut x, mut x_dot, mut theta, mut theta_dot) = self.state;

            let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };

            let costheta = theta.cos();
            let sintheta = theta.sin();

            let tmp = (force + POLE_MASS_LENGTH * theta_dot.powi(2) * sintheta) / TOTAL_MASS;
            let thetaacc = (GRAVITY * sintheta - costheta * tmp)
                / (POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * costheta.powi(2) / TOTAL_MASS));
            let xacc = tmp - POLE_MASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

            x += TAU * x_dot;
            x_dot += TAU * xacc;
            theta += TAU * theta_dot;
            theta_dot += TAU * thetaacc;

            self.state = State(x, x_dot, theta, theta_dot);

            let terminated = !(-X_THRESHOLD..=X_THRESHOLD).contains(&x)
                || !(-THETA_THRESHOLD_RADIANS..=THETA_THRESHOLD_RADIANS).contains(&theta);

            if terminated {
                self.episode_ended = true;
            }

            REWARD_PER_STEP
        }
    }

    impl Default for Env {
        fn default() -> Self {
            Self::new()
        }
    }
}
pub use my_very_own_stick::Env as MyVeryOwnStick;
