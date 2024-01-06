use derive_setters::Setters;

use tch::{
    nn::{self, LinearConfig, Module, OptimizerConfig},
    Device, Kind, Reduction, Tensor,
};

use crate::env::Env as EnvTrait;

#[derive(Debug)]
pub struct ActorCriticModel {
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

#[must_use]
#[derive(Debug)]
pub struct ActorCriticTensor(Tensor);

impl ActorCriticTensor {
    fn action_space(&self) -> i64 {
        self.0.size1().unwrap() - 1
    }

    fn action_probs(&self) -> Tensor {
        self.0
            .narrow(0, 0, self.action_space())
            .softmax(-1, Kind::Double)
    }

    fn action(&self, explore: bool) -> u32 {
        let probs = self.action_probs();

        if explore {
            let action_t = probs.multinomial(1, true);
            action_t.int64_value(&[0]) as u32
        } else {
            let action_t = probs.argmax(-1, false);
            action_t.int64_value(&[]) as u32
        }
    }

    fn value(&self) -> f64 {
        self.0.narrow(0, self.action_space(), 1).double_value(&[0])
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
            vs_path / "al",
            last_layer_node_count,
            action_space,
            LinearConfig::default(),
        );
        let critic = nn::linear(
            vs_path / "cl",
            last_layer_node_count,
            1,
            LinearConfig::default(),
        );

        ActorCriticModel { seq, actor, critic }
    }

    pub fn forward_actor_critic(&self, observation: &Tensor) -> ActorCriticTensor {
        let out = self.forward(observation);
        ActorCriticTensor(out)
    }

    #[must_use]
    pub fn act(&self, observation: &Tensor) -> u32 {
        tch::no_grad(|| self.forward_actor_critic(observation).action(false))
    }

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
                let action = self.act(&observation);
                let reward = env.step(action);
                total_return += reward;
            }
        }

        total_return / EVAL_EPISODE_COUNT as f64
    }

    pub fn explore(&self, observation: &Tensor) -> (ActorCriticTensor, u32) {
        let out = self.forward_actor_critic(observation);
        let action = out.action(true);
        (out, action)
    }

    #[must_use]
    pub fn criticize(&self, observation: &Tensor) -> f64 {
        tch::no_grad(|| self.forward_actor_critic(observation).value())
    }
}

#[derive(Debug)]
struct ActorCriticTensorStack {
    vec: Vec<Tensor>,
    action_space: i64,
}

impl ActorCriticTensorStack {
    fn new(action_space: usize) -> Self {
        Self {
            vec: Vec::new(),
            action_space: action_space as i64,
        }
    }

    fn push(&mut self, tensor: ActorCriticTensor) {
        self.vec.push(tensor.0);
    }

    fn stack_actor_critic(&self) -> (Tensor, Tensor) {
        let stacked_outputs = Tensor::stack(&self.vec, 0);

        let actions_probs = stacked_outputs
            .narrow(1, 0, self.action_space)
            .softmax(-1, Kind::Double);
        let values = stacked_outputs
            .narrow(1, self.action_space, 1)
            .squeeze_dim(1);

        (actions_probs, values)
    }

    fn clear(&mut self) {
        self.vec.clear();
    }
}

#[derive(Debug)]
pub struct Memory {
    device: Device,
    gamma: f64,
    output_stack: ActorCriticTensorStack,
    actions: Vec<i64>,
    rewards: Vec<f64>,
}

impl Memory {
    #[must_use]
    pub fn new(device: Device, action_space: usize, gamma: f64) -> Memory {
        Memory {
            device,
            gamma,
            output_stack: ActorCriticTensorStack::new(action_space),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }

    pub fn push(&mut self, out: ActorCriticTensor, action: u32, reward: f64) {
        self.output_stack.push(out);
        self.actions.push(action as i64);
        self.rewards.push(reward);
    }

    pub fn clear(&mut self) {
        self.output_stack.clear();
        self.actions.clear();
        self.rewards.clear();
    }

    fn compute_expected_returns(&self, remaining_episode_reward: f64) -> Tensor {
        let memory_size = self.rewards.len();
        let mut returns = Tensor::zeros([memory_size as i64], (Kind::Double, self.device));

        let mut discounted_sum = remaining_episode_reward;
        for i in (0..memory_size).rev() {
            let reward = self.rewards[i];
            discounted_sum = reward + self.gamma * discounted_sum;
            _ = returns.get(i as i64).fill_(discounted_sum);
        }

        let mean = returns.mean(Kind::Double);
        let std_dev = returns.std(false);

        returns = (returns - mean) / (std_dev + f64::EPSILON);

        returns
    }

    pub fn compute_loss(&self, remaining_episode_reward: f64) -> Tensor {
        let actions = Tensor::from_slice(&self.actions);
        let (actions_probs, values) = self.output_stack.stack_actor_critic();

        let returns = self.compute_expected_returns(remaining_episode_reward);

        let taken_action_probs = actions_probs
            .gather(1, &actions.unsqueeze(-1), false)
            .squeeze_dim(1);

        let advantage = &returns - &values;

        let taken_action_log_probs = taken_action_probs.log();
        let actor_loss = -taken_action_log_probs.dot(&advantage);
        let critic_loss = values.huber_loss(&returns, Reduction::Sum, 1.0);

        actor_loss + critic_loss
    }
}

#[must_use]
pub struct A2C<Env>
where
    Env: EnvTrait,
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
    Env: EnvTrait,
{
    pub fn train(&mut self, min_iteration_count: usize) {
        let mut steps_done = 0;

        while steps_done < min_iteration_count {
            while !self.train_env.episode_ended() {
                let (out, action) = self.model.explore(&self.train_env.observation());

                let reward = self.train_env.step(action);
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

    pub fn evaluate_avg_return(&mut self) -> f64 {
        self.model.evaluate_avg_return(&mut self.eval_env)
    }

    pub fn builder<'a>() -> Builder<'a, Env> {
        Builder {
            vs: None,
            gamma: None,
            learning_rate: None,
            sync_interval: None,
            fc_layers: None,
            env: None,
        }
    }
}

#[must_use]
#[derive(Setters, Debug)]
#[setters(strip_option)]
pub struct Builder<'a, Env>
where
    Env: EnvTrait,
{
    vs: Option<nn::VarStore>,
    gamma: Option<f64>,
    learning_rate: Option<f64>,
    sync_interval: Option<usize>,
    fc_layers: Option<&'a [usize]>,
    env: Option<Env>,
}

impl<'a, Env> Builder<'a, Env>
where
    Env: EnvTrait,
{
    pub fn build(self) -> A2C<Env> {
        let mut vs = self.vs.unwrap();

        let gamma = self.gamma.unwrap();
        let learning_rate = self.learning_rate.unwrap();
        let sync_interval = self.sync_interval.unwrap();
        let fc_layers = self.fc_layers.unwrap();
        let observation_space = Env::observation_space();
        let action_space = Env::action_space();
        let mut env = self.env.unwrap();
        env.reset();

        let model = ActorCriticModel::new(&vs.root(), observation_space, action_space, fc_layers);
        let memory = Memory::new(vs.device(), action_space, gamma);
        let optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();

        vs.double();

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
