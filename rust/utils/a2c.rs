use derive_setters::Setters;
use ndarray::Array1;

use tch::{
    nn::{self, LinearConfig, Module, OptimizerConfig},
    Kind::Float,
    Reduction, Tensor,
};

use crate::env;

#[derive(Debug)]
pub struct ActorCriticModel {
    action_space: i64,
    seq: nn::Sequential,
    actor: nn::Linear,
    critic: nn::Linear,
}

impl Module for ActorCriticModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs.apply(&self.seq);
        let actor = x.apply(&self.actor);
        let critic = x.apply(&self.critic);
        Tensor::cat(&[actor, critic], 0)
    }
}

impl ActorCriticModel {
    pub fn new(
        vs_path: &nn::Path,
        observation_space: usize,
        action_space: usize,
        fc_layers: &[usize],
    ) -> ActorCriticModel {
        let action_space = action_space as i64;

        let mut seq = nn::seq()
            .add(nn::linear(
                vs_path / "l1",
                observation_space as i64,
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

        let actor = nn::linear(
            vs_path / "a1",
            last_layer_node_count,
            action_space,
            LinearConfig::default(),
        );
        let critic = nn::linear(
            vs_path / "c1",
            last_layer_node_count,
            1,
            LinearConfig::default(),
        );

        ActorCriticModel {
            action_space,
            seq,
            actor,
            critic,
        }
    }

    #[must_use]
    pub fn chose_action(&self, observation: &Tensor) -> usize {
        let out = self.forward(observation);
        let probs = out.narrow(0, 0, self.action_space).softmax(-1, Float);
        let action = probs.argmax(-1, false);
        action.int64_value(&[]).try_into().unwrap()
    }

    #[must_use]
    pub fn evaluate_avg_return<Env>(&self, env: &mut Env) -> f32
    where
        Env: env::Instance,
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

        total_return / EVAL_EPISODE_COUNT as f32
    }

    pub fn explore<Env>(&self, env: &mut Env) -> (Tensor, usize, f32)
    where
        Env: env::Instance,
    {
        let observation = env.observation();
        let out = self.forward(&observation);
        let action_probs = out.narrow(0, 0, self.action_space).softmax(-1, Float);
        let action = action_probs
            .multinomial(1, true)
            .int64_value(&[0])
            .try_into()
            .unwrap();
        let reward = env.step(action);

        (out, action, reward)
    }

    #[must_use]
    pub fn criticize(&self, observation: &Tensor) -> f32 {
        let out = self.forward(observation);
        out.narrow(0, self.action_space, 1).double_value(&[0]) as f32
    }
}

#[derive(Debug)]
pub struct Memory {
    action_space: i64,
    gamma: f32,
    outputs: Vec<Tensor>,
    actions: Vec<i64>,
    rewards: Vec<f32>,
}

impl Memory {
    #[must_use]
    pub fn new(action_space: i64, gamma: f32) -> Memory {
        Memory {
            action_space,
            gamma,
            outputs: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }

    pub fn push(&mut self, out: Tensor, action: usize, reward: f32) {
        self.outputs.push(out);
        self.actions.push(action as i64);
        self.rewards.push(reward);
    }

    pub fn clear(&mut self) {
        self.outputs.clear();
        self.actions.clear();
        self.rewards.clear();
    }

    fn compute_expected_returns(&self, remaining_episode_reward: f32) -> Vec<f32> {
        let outputs_len = self.outputs.len();
        let mut returns = Array1::<f32>::zeros(outputs_len);

        let mut discounted_sum = remaining_episode_reward;
        for i in (0..outputs_len).rev() {
            let reward = self.rewards[i];
            discounted_sum = reward + self.gamma * discounted_sum;
            returns[i] = discounted_sum;
        }

        let mean = returns.mean().unwrap();
        let std_dev = returns.std(0.);

        returns = (returns - mean) / (std_dev + f32::EPSILON);

        returns.to_vec()
    }

    pub fn compute_loss(&self, remaining_episode_reward: f32) -> Tensor {
        let returns = self.compute_expected_returns(remaining_episode_reward);
        let returns = Tensor::from_slice(&returns);

        let outputs = Tensor::stack(&self.outputs, 0);
        let actions = Tensor::from_slice(&self.actions);

        let all_action_probs = outputs.narrow(1, 0, self.action_space).softmax(-1, Float);
        let action_probs = all_action_probs
            .gather(1, &actions.unsqueeze(-1), false)
            .squeeze_dim(1);
        let values = outputs.narrow(1, self.action_space, 1).squeeze_dim(1);

        let advantage = &returns - &values;

        let action_log_probs = action_probs.log();
        let actor_loss = -action_log_probs.dot(&advantage);
        let critic_loss = values.huber_loss(&returns, Reduction::Sum, 1.0);

        actor_loss + critic_loss
    }
}

#[must_use]
pub struct A2C<Env>
where
    Env: env::Instance,
{
    model: ActorCriticModel,
    memory: Memory,
    optimizer: nn::Optimizer,
    sync_interval: usize,
    train_env: Env,
    eval_env: Env,
}

impl<Env> A2C<Env>
where
    Env: env::Instance,
{
    pub fn train(&mut self, iteration_count: usize) {
        let mut steps_done = 0;

        while steps_done < iteration_count {
            while !self.train_env.episode_ended() && steps_done < iteration_count {
                let (out, action, reward) = self.model.explore(&mut self.train_env);
                self.memory.push(out, action, reward);

                steps_done += 1;

                if steps_done % self.sync_interval == 0 {
                    break;
                }
            }

            let episode_ended = self.train_env.episode_ended();

            let remaining_episode_reward = if episode_ended {
                0.0
            } else {
                self.model.criticize(&self.train_env.observation())
            };

            let loss = self.memory.compute_loss(remaining_episode_reward);
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
            self.memory.clear();

            if episode_ended {
                self.train_env.reset();
            }
        }
    }

    pub fn evaluate_avg_return(&mut self) -> f32 {
        self.model.evaluate_avg_return(&mut self.eval_env)
    }
}

#[must_use]
#[derive(Setters, Debug)]
#[setters(strip_option, prefix = "set_")]
pub struct Builder<'a, Env>
where
    Env: env::Instance,
{
    #[setters(skip)]
    vs: &'a nn::VarStore,
    gamma: Option<f32>,
    learning_rate: Option<f64>,
    sync_interval: Option<usize>,
    fc_layers: Option<&'a [usize]>,
    env: Option<Env>,
}

impl<'a, Env> Builder<'a, Env>
where
    Env: env::Instance,
{
    pub fn init(vs: &'a nn::VarStore) -> Builder<'a, Env> {
        Self {
            vs,
            gamma: None,
            learning_rate: None,
            fc_layers: None,
            env: None,
            sync_interval: None,
        }
    }

    pub fn build(self) -> A2C<Env> {
        let gamma = self.gamma.unwrap();
        let learning_rate = self.learning_rate.unwrap();
        let sync_interval = self.sync_interval.unwrap();
        let fc_layers = self.fc_layers.unwrap();
        let observation_space = Env::observation_space();
        let action_space = Env::action_space();
        let mut env = self.env.unwrap();
        env.reset();

        let model =
            ActorCriticModel::new(&self.vs.root(), observation_space, action_space, fc_layers);
        let memory = Memory::new(action_space as i64, gamma);
        let optimizer = nn::Adam::default().build(self.vs, learning_rate).unwrap();

        A2C {
            model,
            memory,
            optimizer,
            sync_interval,
            train_env: env.clone(),
            eval_env: env,
        }
    }
}
