mod clipped_loss;
mod collector;
mod estimator;
mod model;
mod sampler;

use std::{rc::Rc, sync::Arc};

use derive_setters::Setters;

use clipped_loss::ClippedLoss;
use collector::Collector;
use sampler::{ActorSample, CriticSample, Sampler};
use tch::nn::{self, OptimizerConfig};

use crate::env::Env as EnvTrait;

#[must_use]
pub struct PPO<Env>
where
    Env: EnvTrait,
{
    actor: Rc<model::Actor>,
    ga_estimator: estimator::GeneralAdvantage,
    return_estimator: estimator::Return,
    clipped_loss: ClippedLoss,
    collector: Collector,
    actor_optimizer: nn::Optimizer,
    critic_optimizer: nn::Optimizer,
    sampler: Sampler,
    eval_env: Env,
    kl_epsilon: f64,
    actor_train_num_batches: usize,
    critic_train_num_batches: usize,
}

impl<Env> PPO<Env>
where
    Env: EnvTrait,
{
    pub fn train(&mut self, num_epochs: usize) {
        for _ in 0..num_epochs {
            let collector::Payload {
                observations,
                next_observations,
                episodes_not_terminated,
                actions,
                action_probs,
                action_log_probs,
                rewards,
            } = self.collector.collect();

            let advantages = self.ga_estimator.compute_advantages(
                &observations,
                &next_observations,
                &episodes_not_terminated,
                &rewards,
            );

            let returns = self
                .return_estimator
                .compute_returns(&episodes_not_terminated, &rewards);

            self.sampler
                .fill_observations(observations)
                .fill_actions(actions)
                .fill_action_probs(action_probs)
                .fill_action_log_probs(action_log_probs)
                .fill_advantages(advantages)
                .fill_returns(returns);

            for _ in 0..self.actor_train_num_batches {
                let ActorSample {
                    actions,
                    action_probs,
                    action_log_probs,
                    observations,
                    advantages,
                } = self.sampler.actor_sample();

                let actor_loss = self.clipped_loss.compute_actor_loss(
                    &actions,
                    &action_log_probs,
                    &observations,
                    &advantages,
                );

                self.actor_optimizer.zero_grad();
                actor_loss.backward();
                self.actor_optimizer.step();

                let kl = self.clipped_loss.compute_kl(
                    &actions,
                    &action_probs,
                    &action_log_probs,
                    &observations,
                );

                if kl > self.kl_epsilon {
                    break;
                }
            }

            self.collector.sync_collecting_actors();

            for _ in 0..self.critic_train_num_batches {
                let CriticSample {
                    observations,
                    returns,
                } = self.sampler.critic_sample();

                let critic_loss = self
                    .clipped_loss
                    .compute_critic_loss(&observations, returns);

                self.critic_optimizer.zero_grad();
                critic_loss.backward();
                self.critic_optimizer.step();
            }
        }
    }

    pub fn evaluate_avg_return(&mut self) -> f64 {
        self.actor.evaluate_avg_return(&mut self.eval_env)
    }

    pub fn builder<'a>() -> Builder<'a, Env> {
        Builder {
            gamma: None,
            actor_learning_rate: None,
            critic_learning_rate: None,
            fc_layers: None,
            env: None,
            actor_vs: None,
            critic_vs: None,
            num_steps_per_epoch: None,
            batch_size: None,
            lmbda: None,
            clip_epsilon: None,
            kl_epsilon: None,
            actor_train_num_batches: None,
            critic_train_num_batches: None,
            num_threads: None,
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
    actor_vs: Option<nn::VarStore>,
    critic_vs: Option<nn::VarStore>,
    gamma: Option<f64>,
    lmbda: Option<f64>,
    clip_epsilon: Option<f64>,
    kl_epsilon: Option<f64>,
    actor_learning_rate: Option<f64>,
    critic_learning_rate: Option<f64>,
    fc_layers: Option<&'a [usize]>,
    env: Option<Env>,
    num_steps_per_epoch: Option<usize>,
    num_threads: Option<usize>,
    batch_size: Option<usize>,
    actor_train_num_batches: Option<usize>,
    critic_train_num_batches: Option<usize>,
}

impl<Env> Builder<'_, Env>
where
    Env: EnvTrait,
{
    pub fn build(self) -> PPO<Env> {
        let actor_learning_rate = self.actor_learning_rate.unwrap();
        let critic_learning_rate = self.critic_learning_rate.unwrap();
        let fc_layers = self.fc_layers.unwrap();
        let num_steps_per_epoch = self.num_steps_per_epoch.unwrap();
        let observation_space = Env::observation_space();
        let action_space = Env::action_space();
        let mut actor_vs = self.actor_vs.unwrap();
        let mut critic_vs = self.critic_vs.unwrap();
        let mut env = self.env.unwrap();
        env.reset();

        let actor: Rc<model::Actor> = Rc::new(model::Actor::new(
            &actor_vs.root(),
            observation_space,
            action_space,
            fc_layers,
        ));
        let critic = Rc::new(model::Critic::new(
            &critic_vs.root(),
            observation_space,
            fc_layers,
        ));

        let actor_optimizer = nn::Adam::default()
            .build(&actor_vs, actor_learning_rate)
            .unwrap();
        let critic_optimizer = nn::Adam::default()
            .build(&critic_vs, critic_learning_rate)
            .unwrap();

        actor_vs.double();
        critic_vs.double();

        let device = actor_vs.device();
        assert_eq!(device, critic_vs.device());

        let actor_vs = Arc::new(actor_vs);

        let collector = Collector::new(collector::Options {
            global_actor_vs: &actor_vs,
            actor_fc_layers: &fc_layers.to_vec(),
            device,
            num_steps: num_steps_per_epoch,
            num_threads: self.num_threads.unwrap(),
            env: &env,
        });

        let batch_size = self.batch_size.unwrap();

        let sampler = Sampler::new(num_steps_per_epoch, batch_size, device);

        let gamma = self.gamma.unwrap();
        let lmbda = self.lmbda.unwrap();

        let ga_estimator = estimator::GeneralAdvantage::builder()
            .critic(Rc::clone(&critic))
            .num_steps(num_steps_per_epoch)
            .gamma(gamma)
            .lmbda(lmbda)
            .device(device)
            .build();

        let return_estimator = estimator::Return::builder()
            .num_steps(num_steps_per_epoch)
            .gamma(gamma)
            .device(device)
            .build();

        let clipped_loss = ClippedLoss::builder()
            .actor(Rc::clone(&actor))
            .critic(critic)
            .clip_epsilon(self.clip_epsilon.unwrap())
            .batch_size(batch_size)
            .device(device)
            .build();

        PPO {
            actor,
            actor_optimizer,
            critic_optimizer,
            collector,
            sampler,
            ga_estimator,
            return_estimator,
            clipped_loss,
            eval_env: env,
            kl_epsilon: self.kl_epsilon.unwrap(),
            actor_train_num_batches: self.actor_train_num_batches.unwrap(),
            critic_train_num_batches: self.critic_train_num_batches.unwrap(),
        }
    }
}
