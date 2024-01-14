from __future__ import absolute_import, division, print_function

import copy
import random
import time
from collections import deque

import keras
import numpy as np
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec


def _dense_layer(num_units: int):
    return keras.layers.Dense(
        num_units,
        activation=keras.activations.relu,
        kernel_initializer=keras.initializers.VarianceScaling(
            scale=2.0,
            mode="fan_in",
            distribution="truncated_normal"
        )
    )


AVG_RETURN_EVALUATION_NUM_EPISODES = 20


class BasicDqn(object):
    def __init__(
            self,
            *,
            env: py_environment.PyEnvironment,
            initial_collect_steps: int,
            collect_steps_per_iteration: int,
            replay_buffer_max_length: int,
            batch_size: int,
            learning_rate: float,
            gamma: float,
            fc_layer_params: tuple[int],
    ):
        self.train_py_env = copy.deepcopy(env)
        self.eval_py_env = env

        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.batch_size = batch_size
        self.gamma = gamma

        action_tensor_spec = tensor_spec.from_spec(env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        q_net_dense_layers = [
            _dense_layer(num_units)
            for num_units in fc_layer_params
        ]

        q_values_layer = keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=keras.initializers.RandomUniform(
                minval=-0.03,
                maxval=0.03
            ),
            bias_initializer=keras.initializers.Constant(-0.2)
        )

        self.model = keras.models.Sequential(
            q_net_dense_layers + [q_values_layer]
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss='mse')

        random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )

        self.experience_replay = deque(maxlen=replay_buffer_max_length)

        self.train_env.reset()

        for _ in range(initial_collect_steps):
            time_step = self.train_env.current_time_step()
            action_step = random_policy.action(time_step)
            next_time_step = self.train_env.step(action_step.action)
            self.experience_replay.append(
                (time_step, action_step.action, next_time_step)
            )

        self.train_env.reset()

    def predict_action(self, state):
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def evaluate_avg_return(self):
        total_return = 0.0
        for _ in range(AVG_RETURN_EVALUATION_NUM_EPISODES):
            time_step = self.eval_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action = self.predict_action(time_step.observation)
                time_step = self.eval_env.step(action)
                episode_return += time_step.reward

            total_return += episode_return

        avg_return = total_return / AVG_RETURN_EVALUATION_NUM_EPISODES
        return avg_return.numpy()[0]

    def train(self, num_iterations: int):
        for _ in range(num_iterations // self.collect_steps_per_iteration):
            for _ in range(self.collect_steps_per_iteration):
                time_step = self.train_env.current_time_step()
                action = self.predict_action(time_step.observation)
                next_time_step = self.train_env.step(action)
                self.experience_replay.append(
                    (time_step, action, next_time_step)
                )

            if len(self.experience_replay) < self.batch_size:
                continue

            batch = random.sample(self.experience_replay, self.batch_size)

            observations = np.array(
                [time_step.observation for time_step, _, _ in batch]
            )

            predicted_q_values_arr = self.model.predict_on_batch(
                observations,
            )

            predicted_next_q_values_arr = self.model.predict_on_batch(
                np.array(
                    [next_time_step.observation for _, _, next_time_step in batch]
                )
            )

            target_q_values_arr = []

            for i, (time_step, action, next_time_step) in enumerate(batch):
                target_q_values = predicted_q_values_arr[i]

                if time_step.is_last():
                    target_q_values[0][action] = time_step.reward
                else:
                    target_q_values[0][action] = (
                        time_step.reward +
                        self.gamma *
                        time_step.discount *
                        np.amax(predicted_next_q_values_arr[i])
                    )

                target_q_values_arr.append(target_q_values)

            self.model.train_on_batch(
                observations,
                np.array(target_q_values_arr),
            )
