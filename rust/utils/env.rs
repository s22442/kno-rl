use tch::Tensor;

pub trait Instance: Clone + Send + 'static {
    fn observation_space() -> usize;
    fn action_space() -> usize;
    fn observation(&self) -> Tensor;
    fn episode_ended(&self) -> bool;
    fn reset(&mut self);
    fn step(&mut self, action: usize) -> f32;
}

mod my_very_own_stick {
    pub use super::Instance as InstanceTrait;
    use rand::Rng;
    use std::f32::consts::PI;
    use tch::Tensor;

    #[derive(Clone, Debug)]
    struct State(f32, f32, f32, f32);

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
    pub struct Instance {
        current_step: u16,
        episode_ended: bool,
        state: State,
    }

    const MAX_STEPS: u16 = 500;
    const REWARD_PER_STEP: f32 = 1.0;
    const GRAVITY: f32 = 9.8;
    const CART_MASS: f32 = 1.0;
    const POLE_MASS: f32 = 0.1;
    const TOTAL_MASS: f32 = CART_MASS + POLE_MASS;
    const POLE_LENGTH: f32 = 0.5;
    const POLE_MASS_LENGTH: f32 = POLE_MASS * POLE_LENGTH;
    const FORCE_MAG: f32 = 10.0;
    const TAU: f32 = 0.02;
    const THETA_THRESHOLD_RADIANS: f32 = 12.0 * 2.0 * PI / 360.0;
    const X_THRESHOLD: f32 = 2.4;

    impl Instance {
        #[must_use]
        pub fn new() -> Self {
            Self {
                current_step: 0,
                episode_ended: false,
                state: State::default(),
            }
        }
    }

    impl InstanceTrait for Instance {
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

        fn step(&mut self, action: usize) -> f32 {
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

    impl Default for Instance {
        fn default() -> Self {
            Self::new()
        }
    }
}
pub use my_very_own_stick::Instance as MyVeryOwnStick;
