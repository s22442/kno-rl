import copy

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories.time_step import TimeStep


def _get_num_actions_from_env(env: PyEnvironment) -> int:
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    return num_actions


EPS = np.finfo(np.float32).eps.item()


def _calculate_expected_returns(
        rewards: tf.TensorArray,
        gamma: float,
        remaining_episode_reward: float
) -> tf.Tensor:
    rewards = rewards.stack()

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(remaining_episode_reward)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + EPS))

    return returns


class ActorCriticModel(keras.Model):
    def __init__(self, num_actions: int, fc_layer_params: tuple[int]):
        super().__init__()

        self.common_layers = [
            layers.Dense(num_units, activation="relu")
            for num_units in fc_layer_params
        ]

        self.actor_layer = layers.Dense(num_actions)
        self.critic_layer = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        for layer in self.common_layers:
            inputs = layer(inputs)

        return self.actor_layer(inputs), self.critic_layer(inputs)


AVG_RETURN_EVALUATION_NUM_EPISODES = 20


class ActorCriticAgent:
    def __init__(self, num_actions: int, fc_layer_params: tuple[int]):
        self.model = ActorCriticModel(num_actions, fc_layer_params)

    def predict_action(self, state: tf.Tensor) -> int:
        state = np.expand_dims(state, axis=0)
        logits, _ = self.model(state)
        action_probs_t = tf.nn.softmax(logits)
        action_probs = action_probs_t.numpy().flatten()
        action = np.argmax(action_probs)
        return action

    def evaluate_avg_return(self, env: PyEnvironment) -> float:
        total_return = 0.0
        for _ in range(AVG_RETURN_EVALUATION_NUM_EPISODES):
            time_step = env.reset()

            while not time_step.is_last():
                action = self.predict_action(time_step.observation)
                time_step = env.step(action)
                total_return += time_step.reward

        avg_return = total_return / AVG_RETURN_EVALUATION_NUM_EPISODES
        return avg_return


class ActorCriticMemory:
    def reset(self):
        self.action_probs = tf.TensorArray(
            dtype=tf.float32, size=0,
            dynamic_size=True
        )
        self.values = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True)
        self.rewards = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True
        )
        self.size = 0

    def __init__(self, gamma: float):
        self.huber_loss = keras.losses.Huber(
            reduction=keras.losses.Reduction.SUM
        )
        self.gamma = gamma

        self.reset()

    def push(self, action_probs: tf.Tensor, value: tf.Tensor, reward: int):
        self.action_probs = self.action_probs.write(
            self.size,
            action_probs
        )
        self.values = self.values.write(
            self.size,
            value
        )
        self.rewards = self.rewards.write(
            self.size,
            reward
        )
        self.size += 1

    def compute_loss(self, remaining_episode_reward: float) -> tf.Tensor:
        returns = _calculate_expected_returns(
            self.rewards,
            self.gamma,
            remaining_episode_reward
        )

        action_probs = tf.expand_dims(self.action_probs.stack(), 1)
        values = tf.expand_dims(self.values.stack(), 1)

        returns = tf.expand_dims(returns, 1)

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss


class A2C():
    def __init__(
        self,
        *,
        env: PyEnvironment,
        sync_interval: int,
        gamma: float,
        fc_layer_params: tuple[int],
        learning_rate: float,

    ):
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.agent = ActorCriticAgent(
            _get_num_actions_from_env(env),
            fc_layer_params
        )

        self.sync_interval = sync_interval

        env.reset()
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.memory = ActorCriticMemory(gamma)

    def _advance_episode(self) -> TimeStep:
        time_step = self.env.current_time_step()
        num_steps_done = 0

        for _ in range(self.sync_interval):
            state = np.expand_dims(time_step.observation, axis=0)

            logits, value = self.agent.model(state)

            action_probs_t = tf.nn.softmax(logits)
            action_probs_flat = action_probs_t.numpy().flatten()
            action = np.random.choice(
                len(action_probs_flat),
                p=action_probs_flat
            )

            time_step = self.env.step(action)

            reward = np.array(time_step.reward, np.int32)

            self.memory.push(
                action_probs_t[0, action],
                tf.squeeze(value),
                reward
            )

            num_steps_done += 1

            if time_step.is_last():
                break

        return num_steps_done

    def train(self, min_num_iterations: int):
        remaining_steps = min_num_iterations

        while remaining_steps > 0:
            with tf.GradientTape() as tape:
                num_steps = self._advance_episode()
                time_step = self.env.current_time_step()
                if time_step.is_last():
                    remaining_episode_reward = 0.0
                else:
                    _, value = self.agent.model(
                        np.expand_dims(time_step.observation, axis=0)
                    )
                    remaining_episode_reward = value.numpy()[0][0]

                loss = self.memory.compute_loss(remaining_episode_reward)

            remaining_steps -= num_steps

            grads = tape.gradient(
                loss,
                self.agent.model.trainable_variables
            )

            self.optimizer.apply_gradients(
                zip(
                    grads,
                    self.agent.model.trainable_variables
                )
            )

            self.memory.reset()

    def evaluate_avg_return(self) -> float:
        return self.agent.evaluate_avg_return(self.eval_env)
