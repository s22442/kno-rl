use std::rc::Rc;

use tch::{nn::Module, Device, Kind, Tensor};
use typed_builder::TypedBuilder;

use super::model;

#[derive(TypedBuilder)]
pub struct ClippedLoss {
    actor: Rc<model::Actor>,
    critic: Rc<model::Critic>,
    clip_epsilon: f64,

    #[builder(setter(transform = |batch_size: usize| batch_size as i64))]
    batch_size: i64,

    device: Device,
}

impl ClippedLoss {
    pub fn compute_actor_loss(
        &self,
        actions: &Tensor,
        action_log_probs: &Tensor,
        observations: &Tensor,
        advantages: &Tensor,
    ) -> Tensor {
        let logits = self.actor.forward(observations);
        let new_action_log_probs = logits
            .log_softmax(-1, Kind::Double)
            .gather(1, actions, false)
            .squeeze_dim(1);

        let ratio = (&new_action_log_probs - action_log_probs).exp();

        let min_advantages = Tensor::zeros([self.batch_size], (Kind::Double, self.device));

        for i in 0..self.batch_size {
            let advantage = advantages.get(i);
            let multiplier = if advantage.double_value(&[]) > 0.0 {
                1.0 + self.clip_epsilon
            } else {
                1.0 - self.clip_epsilon
            };
            min_advantages.get(i).copy_(&(advantage * multiplier));
        }

        -(&ratio * advantages)
            .min_other(&min_advantages)
            .mean(Kind::Double)
    }

    pub fn compute_kl(
        &self,
        actions: &Tensor,
        action_probs: &Tensor,
        action_log_probs: &Tensor,
        observations: &Tensor,
    ) -> f64 {
        let logits = tch::no_grad(|| self.actor.forward(observations));
        let new_action_log_probs = logits
            .log_softmax(-1, Kind::Double)
            .gather(1, actions, false)
            .squeeze_dim(1);

        let kl = (action_probs * (action_log_probs - new_action_log_probs)).mean(Kind::Double);

        kl.double_value(&[])
    }

    pub fn compute_critic_loss(&self, observations: &Tensor, returns: Tensor) -> Tensor {
        let values = self.critic.forward(observations).squeeze();
        (returns - values).pow_tensor_scalar(2).mean(Kind::Double)
    }
}
