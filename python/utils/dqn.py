from __future__ import absolute_import, division, print_function

import copy

import keras
import reverb
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy, random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


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


class Dqn(object):
    def __init__(
            self,
            *,
            env: py_environment.PyEnvironment,
            initial_collect_steps: int,
            collect_steps_per_iteration: int,
            replay_buffer_max_length: int,
            batch_size: int,
            learning_rate: float,
            fc_layer_params: tuple[int],
    ):
        self.train_py_env = copy.deepcopy(env)
        self.eval_py_env = env

        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.initial_collect_steps = initial_collect_steps
        self.replay_buffer_max_length = replay_buffer_max_length
        self.collect_steps_per_iteration = collect_steps_per_iteration

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

        q_net = sequential.Sequential(q_net_dense_layers + [q_values_layer])

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0)
        )

        self.agent.initialize()

        table_name = "uniform_table"
        replay_buffer_signature = tensor_spec.from_spec(
            self.agent.collect_data_spec
        )
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature
        )

        table = reverb.Table(
            table_name,
            max_size=replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature
        )

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server
        )

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2
        )

        random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )

        py_driver.PyDriver(
            env,
            py_tf_eager_policy.PyTFEagerPolicy(random_policy),
            [rb_observer],
            max_steps=initial_collect_steps
        ).run(self.train_py_env.reset())

        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=2
        )

        self.experience_iterator = iter(dataset)

        self.collect_driver = py_driver.PyDriver(
            env,
            py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.collect_policy,
                use_tf_function=True
            ),
            [rb_observer],
            max_steps=collect_steps_per_iteration
        )

        self.agent.train_step_counter.assign(0)
        self.last_time_step = self.train_py_env.reset()

    def evaluate_avg_return(self):
        total_return = 0.0
        for _ in range(AVG_RETURN_EVALUATION_NUM_EPISODES):
            time_step = self.eval_env.reset()

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.eval_env.step(action_step.action)
                total_return += time_step.reward

        avg_return = total_return / AVG_RETURN_EVALUATION_NUM_EPISODES
        return avg_return.numpy()[0]

    def train(self, num_iterations: int):
        for _ in range(num_iterations // self.collect_steps_per_iteration):
            self.last_time_step, _ = self.collect_driver.run(
                self.last_time_step
            )

            experience, _ = next(self.experience_iterator)
            self.agent.train(experience)

    def get_policy(self):
        return self.agent.policy
