use std::{
    sync::{Arc, Barrier, Mutex},
    thread,
};

use derive_setters::Setters;

use tch::nn::{self, OptimizerConfig, VarStore};

use crossbeam::channel::{self, Receiver, Sender};

use crate::{
    a2c::{ActorCriticModel, Memory},
    env,
};

enum Broadcast {
    Train(usize),
    Terminate,
}

#[must_use]
pub struct A3C<Env>
where
    Env: env::Instance,
{
    model: ActorCriticModel,
    eval_env: Env,
    thread_handles: Vec<thread::JoinHandle<()>>,
    training_barrier: Arc<Barrier>,
    broadcaster: Sender<Broadcast>,
}

impl<Env> Drop for A3C<Env>
where
    Env: env::Instance,
{
    fn drop(&mut self) {
        for _ in 0..self.thread_handles.len() {
            self.broadcaster.send(Broadcast::Terminate).unwrap();
        }

        for handle in self.thread_handles.drain(..) {
            handle.join().unwrap();
        }
    }
}

impl<Env> A3C<Env>
where
    Env: env::Instance,
{
    pub fn train(&mut self, iteration_count: usize) {
        for _ in 0..self.thread_handles.len() {
            self.broadcaster
                .send(Broadcast::Train(iteration_count))
                .unwrap();
        }

        self.training_barrier.wait();
    }

    pub fn evaluate_avg_return(&mut self) -> f32 {
        self.model.evaluate_avg_return(&mut self.eval_env)
    }
}

struct TrainingThreadOptions<'a, Env>
where
    Env: env::Instance,
{
    global_vs: &'a Arc<Mutex<VarStore>>,
    env: &'a Env,
    fc_layers: &'a Vec<usize>,
    learning_rate: f64,
    gamma: f32,
    sync_interval: usize,
    broadcast_rx: &'a Receiver<Broadcast>,
    training_barrier: &'a Arc<Barrier>,
}

fn spawn_training_thread<Env>(
    TrainingThreadOptions {
        global_vs,
        env,
        fc_layers,
        learning_rate,
        gamma,
        sync_interval,
        broadcast_rx,
        training_barrier,
    }: TrainingThreadOptions<Env>,
) -> thread::JoinHandle<()>
where
    Env: env::Instance,
{
    let observation_space = Env::observation_space();
    let action_space = Env::action_space();
    let fc_layers = fc_layers.clone();

    let mut local_env = env.clone();
    let broadcast_rx = broadcast_rx.clone();
    let training_barrier = Arc::clone(training_barrier);
    let global_vs = Arc::clone(global_vs);

    thread::spawn(move || {
        let mut local_vs = VarStore::new(global_vs.lock().unwrap().device());

        let local_model = ActorCriticModel::new(
            &local_vs.root(),
            observation_space,
            action_space,
            &fc_layers,
        );

        let mut local_optimizer = nn::Adam::default().build(&local_vs, learning_rate).unwrap();

        let action_space = action_space as i64;

        let mut local_memory = Memory::new(action_space, gamma);

        while let Ok(msg) = broadcast_rx.recv() {
            match msg {
                Broadcast::Train(iteration_count) => {
                    let mut steps_done = 0;
                    while steps_done < iteration_count {
                        while !local_env.episode_ended() && steps_done < iteration_count {
                            let (out, action, reward) = local_model.explore(&mut local_env);
                            local_memory.push(out, action, reward);

                            steps_done += 1;

                            if steps_done % sync_interval == 0 {
                                break;
                            }
                        }

                        let episode_ended = local_env.episode_ended();

                        let remaining_episode_reward = if episode_ended {
                            0.0
                        } else {
                            local_model.criticize(&local_env.observation())
                        };

                        let loss = local_memory.compute_loss(remaining_episode_reward);
                        local_optimizer.zero_grad();
                        loss.backward();
                        {
                            let mut global_vs = global_vs.lock().unwrap();
                            local_vs.copy(&global_vs).unwrap();
                            local_optimizer.step();
                            global_vs.copy(&local_vs).unwrap();
                        }

                        local_memory.clear();

                        if episode_ended {
                            local_env.reset();
                        }
                    }

                    training_barrier.wait();
                }
                Broadcast::Terminate => {
                    break;
                }
            }
        }
    })
}

#[must_use]
#[derive(Setters, Debug)]
#[setters(strip_option, prefix = "set_")]
pub struct Builder<'a, Env>
where
    Env: env::Instance,
{
    vs: Option<nn::VarStore>,
    gamma: Option<f32>,
    learning_rate: Option<f64>,
    sync_interval: Option<usize>,
    fc_layers: Option<&'a [usize]>,
    env: Option<Env>,
    num_threads: Option<usize>,
}

impl<'a, Env> Builder<'a, Env>
where
    Env: env::Instance,
{
    pub fn init() -> Builder<'a, Env> {
        Self {
            vs: None,
            gamma: None,
            learning_rate: None,
            fc_layers: None,
            env: None,
            sync_interval: None,
            num_threads: None,
        }
    }

    pub fn build(self) -> A3C<Env> {
        let gamma = self.gamma.unwrap();
        let learning_rate = self.learning_rate.unwrap();
        let sync_interval = self.sync_interval.unwrap();
        let fc_layers = self.fc_layers.unwrap().to_vec();
        let num_threads = self.num_threads.unwrap();
        let observation_space = Env::observation_space();
        let action_space = Env::action_space();
        let mut env = self.env.unwrap();
        env.reset();

        let global_vs = self.vs.unwrap();

        let global_model = ActorCriticModel::new(
            &global_vs.root(),
            observation_space,
            action_space,
            &fc_layers,
        );

        let global_vs = Arc::new(Mutex::new(global_vs));

        let mut thread_handles = Vec::new();

        let (broadcaster, broadcast_rx) = channel::unbounded::<Broadcast>();

        let training_barrier = Arc::new(Barrier::new(num_threads + 1));

        for _ in 0..num_threads {
            let handle = spawn_training_thread(TrainingThreadOptions {
                global_vs: &global_vs,
                env: &env,
                fc_layers: &fc_layers,
                learning_rate,
                gamma,
                sync_interval,
                broadcast_rx: &broadcast_rx,
                training_barrier: &training_barrier,
            });

            thread_handles.push(handle);
        }

        A3C {
            model: global_model,
            eval_env: env,
            broadcaster,
            thread_handles,
            training_barrier,
        }
    }
}
