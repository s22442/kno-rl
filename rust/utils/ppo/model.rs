use tch::{
    nn::{self, LinearConfig, Module},
    Kind, Tensor,
};

use crate::env::Env as EnvTrait;

fn build_seq(
    vs_path: &nn::Path,
    observation_space: usize,
    out_dim: usize,
    fc_layers: &[usize],
) -> nn::Sequential {
    let observation_space = observation_space as i64;
    let out_dim = out_dim as i64;

    let mut seq = nn::seq()
        .add(nn::linear(
            vs_path / "l1",
            observation_space,
            fc_layers[0] as i64,
            LinearConfig::default(),
        ))
        .add_fn(Tensor::relu);

    for (i, node_count) in fc_layers.iter().enumerate() {
        if i == 0 {
            continue;
        }

        seq = seq
            .add(nn::linear(
                vs_path / format!("l{}", i + 2),
                fc_layers[i - 1] as i64,
                *node_count as i64,
                LinearConfig::default(),
            ))
            .add_fn(Tensor::relu);
    }

    let last_layer_node_count = *fc_layers.last().unwrap() as i64;

    seq.add(nn::linear(
        vs_path / "out",
        last_layer_node_count,
        out_dim,
        LinearConfig::default(),
    ))
}

#[derive(Debug)]
pub struct Actor {
    seq: nn::Sequential,
}

impl Module for Actor {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.seq)
    }
}

impl Actor {
    pub fn new(
        vs_path: &nn::Path,
        observation_space: usize,
        action_space: usize,
        fc_layers: &[usize],
    ) -> Self {
        Self {
            seq: build_seq(vs_path, observation_space, action_space, fc_layers),
        }
    }

    #[must_use]
    pub fn chose_action(&self, observation: &Tensor) -> u32 {
        let logits = tch::no_grad(|| self.forward(observation));
        let probs = logits.softmax(-1, Kind::Double);
        let action = probs.argmax(-1, false);
        action.int64_value(&[]) as u32
    }

    #[must_use]
    pub fn evaluate_avg_return<Env>(&self, env: &mut Env) -> f64
    where
        Env: EnvTrait,
    {
        const EVAL_EPISODE_COUNT: u8 = 20;

        let mut total_return = 0.0;
        for _ in 0..EVAL_EPISODE_COUNT {
            env.reset();

            while !env.episode_ended() {
                let observation = env.observation();
                let action = self.chose_action(&observation);
                let reward = env.step(action);
                total_return += reward;
            }
        }

        total_return / EVAL_EPISODE_COUNT as f64
    }
}

#[derive(Debug)]
pub struct Critic {
    seq: nn::Sequential,
}

impl Module for Critic {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.seq)
    }
}

impl Critic {
    pub fn new(vs_path: &nn::Path, observation_space: usize, fc_layers: &[usize]) -> Self {
        Self {
            seq: build_seq(vs_path, observation_space, 1, fc_layers),
        }
    }
}
