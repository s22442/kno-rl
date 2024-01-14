use std::{sync::Arc, thread};

use crossbeam::channel::{self, Receiver, Sender};
use tch::{
    nn::{Module, VarStore},
    Device, Kind, Tensor,
};

use crate::env::Env as EnvTrait;

use super::model;

#[must_use]
pub struct Payload {
    pub observations: Tensor,
    pub next_observations: Tensor,
    pub episodes_not_terminated: Tensor,
    pub actions: Tensor,
    pub action_log_probs: Tensor,
    pub rewards: Tensor,
}

impl Payload {
    pub fn new(num_steps: usize, observation_space: usize, device: Device) -> Self {
        let num_steps = num_steps as i64;
        let observation_space = observation_space as i64;

        let options = (Kind::Double, device);

        Self {
            observations: Tensor::zeros([num_steps, observation_space], options),
            next_observations: Tensor::zeros([num_steps, observation_space], options),
            episodes_not_terminated: Tensor::zeros([num_steps], options),
            actions: Tensor::zeros([num_steps, 1], (Kind::Int64, device)),
            action_log_probs: Tensor::zeros([num_steps], options),
            rewards: Tensor::zeros([num_steps], options),
        }
    }
}

enum Broadcast {
    Collect,
    PullGlobalActorVs,
    Terminate,
}

struct CollectingThreadOptions<'a, Env>
where
    Env: EnvTrait,
{
    device: Device,
    global_actor_vs: &'a Arc<VarStore>,
    actor_fc_layers: &'a Vec<usize>,
    env: &'a Env,
    num_steps: usize,
    broadcast_rx: &'a Receiver<Broadcast>,
    collecting_tx: &'a Sender<Payload>,
}

fn spawn_collecting_thread<Env>(
    CollectingThreadOptions {
        broadcast_rx,
        collecting_tx,
        env,
        global_actor_vs,
        actor_fc_layers,
        num_steps,
        device,
    }: CollectingThreadOptions<Env>,
) -> thread::JoinHandle<()>
where
    Env: EnvTrait,
{
    let observation_space = Env::observation_space();
    let action_space = Env::action_space();
    let actor_fc_layers = actor_fc_layers.clone();

    let broadcast_rx = broadcast_rx.clone();
    let collecting_tx = collecting_tx.clone();

    let global_actor_vs = Arc::clone(global_actor_vs);
    let mut local_env = env.clone();

    thread::spawn(move || {
        let mut local_vs = VarStore::new(device);

        let local_actor = model::Actor::new(
            &local_vs.root(),
            observation_space,
            action_space,
            &actor_fc_layers,
        );

        local_vs.double();

        while let Ok(msg) = broadcast_rx.recv() {
            match msg {
                Broadcast::Collect => {
                    let payload = Payload::new(num_steps, observation_space, device);

                    let mut observation = local_env.observation();
                    for i in 0..(num_steps as i64) {
                        payload.observations.get(i).copy_(&observation);

                        let logits = tch::no_grad(|| local_actor.forward(&observation));
                        let probs = logits.softmax(-1, Kind::Double);
                        let action_t = probs.multinomial(1, true);
                        payload.actions.get(i).copy_(&action_t);

                        let action_log_prob = logits
                            .log_softmax(0, Kind::Double)
                            .gather(0, &action_t, false)
                            .squeeze_dim(0);

                        payload.action_log_probs.get(i).copy_(&action_log_prob);

                        let action = action_t.int64_value(&[0]).try_into().unwrap();
                        let reward = local_env.step(action);
                        _ = payload.rewards.get(i).fill_(reward as f64);

                        observation = local_env.observation();
                        payload.next_observations.get(i).copy_(&observation);

                        let episode_ended = local_env.episode_ended();

                        if episode_ended {
                            local_env.reset();
                        } else {
                            _ = payload.episodes_not_terminated.get(i).fill_(1.0);
                        }
                    }

                    collecting_tx.send(payload).unwrap();
                }
                Broadcast::PullGlobalActorVs => {
                    local_vs.copy(&global_actor_vs).unwrap();
                }
                Broadcast::Terminate => {
                    break;
                }
            }
        }
    })
}

#[must_use]
pub struct Collector {
    num_threads: usize,
    thread_handles: Vec<thread::JoinHandle<()>>,
    broadcaster: Sender<Broadcast>,
    collecting_rx: Receiver<Payload>,
    device: Device,
}

impl Drop for Collector {
    fn drop(&mut self) {
        for _ in 0..self.thread_handles.len() {
            self.broadcaster.send(Broadcast::Terminate).unwrap();
        }

        for handle in self.thread_handles.drain(..) {
            handle.join().unwrap();
        }
    }
}

pub struct Options<'a, Env>
where
    Env: EnvTrait,
{
    pub global_actor_vs: &'a Arc<VarStore>,
    pub actor_fc_layers: &'a Vec<usize>,
    pub device: Device,
    pub env: &'a Env,
    pub num_steps: usize,
    pub num_threads: usize,
}

impl Collector {
    pub fn new<Env>(
        Options {
            global_actor_vs,
            actor_fc_layers,
            device,
            num_steps,
            num_threads,
            env,
        }: Options<Env>,
    ) -> Self
    where
        Env: EnvTrait,
    {
        let mut thread_handles = Vec::new();

        let (broadcaster, broadcast_rx) = channel::unbounded();

        let (collecting_tx, collecting_rx) = channel::unbounded();

        assert_eq!(num_steps % num_threads, 0);
        let num_steps_per_thread = num_steps / num_threads;

        for _ in 0..num_threads {
            let handle = spawn_collecting_thread(CollectingThreadOptions {
                device,
                global_actor_vs,
                actor_fc_layers,
                env,
                num_steps: num_steps_per_thread,
                broadcast_rx: &broadcast_rx,
                collecting_tx: &collecting_tx,
            });

            thread_handles.push(handle);
        }

        Self {
            num_threads,
            thread_handles,
            broadcaster,
            collecting_rx,
            device,
        }
    }

    fn empty_tensor(&self) -> Tensor {
        Tensor::zeros([0], (Kind::Double, self.device))
    }

    pub fn collect(&self) -> Payload {
        for _ in 0..self.num_threads {
            self.broadcaster.send(Broadcast::Collect).unwrap();
        }

        let mut observations = self.empty_tensor();
        let mut next_observations = self.empty_tensor();
        let mut episodes_not_terminated = self.empty_tensor();
        let mut actions = Tensor::zeros([0], (Kind::Int64, self.device));
        let mut action_log_probs = self.empty_tensor();
        let mut rewards = self.empty_tensor();

        for _ in 0..self.num_threads {
            let payload = self.collecting_rx.recv().unwrap();

            observations = Tensor::cat(&[observations, payload.observations], 0);
            next_observations = Tensor::cat(&[next_observations, payload.next_observations], 0);
            episodes_not_terminated = Tensor::cat(
                &[episodes_not_terminated, payload.episodes_not_terminated],
                0,
            );
            actions = Tensor::cat(&[actions, payload.actions], 0);
            action_log_probs = Tensor::cat(&[action_log_probs, payload.action_log_probs], 0);
            rewards = Tensor::cat(&[rewards, payload.rewards], 0);
        }

        Payload {
            observations,
            next_observations,
            episodes_not_terminated,
            actions,
            action_log_probs,
            rewards,
        }
    }

    pub fn sync_collecting_actors(&self) {
        for _ in 0..self.num_threads {
            self.broadcaster.send(Broadcast::PullGlobalActorVs).unwrap();
        }
    }
}
