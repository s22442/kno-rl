import imageio
import IPython
from tf_agents.environments import tf_py_environment
from utils.basic_dqn import BasicDqn
from utils.my_very_own_stick_env import MyVeryOnwStickEnv
from utils.out import embed_mp4, out_dir, write_graph

dqn = BasicDqn(
    env=MyVeryOnwStickEnv(),
    initial_collect_steps=200,
    collect_steps_per_iteration=10,
    replay_buffer_max_length=100_000,
    batch_size=200,
    learning_rate=1e-3,
    gamma=0.99,
    fc_layer_params=(100, 100)
)

NUM_ITERATIONS = 10_000
EVAL_INTERVAL = 500

avg_returns = [dqn.evaluate_avg_return()]

for i in range(NUM_ITERATIONS // EVAL_INTERVAL):
    dqn.train(EVAL_INTERVAL)
    avg_return = dqn.evaluate_avg_return()
    avg_returns.append(avg_return)
    print(f"Step: {(i + 1) * EVAL_INTERVAL} | Average Return: {avg_return}")


write_graph(
    filename="basic_dqn_graph.png",
    num_iterations=NUM_ITERATIONS,
    eval_interval=EVAL_INTERVAL,
    avg_returns=avg_returns,
    max_return=500
)

VIDEO_EPISODES = 3

filename = out_dir() + "basic_dqn_video.mp4"
env_py = MyVeryOnwStickEnv()
env = tf_py_environment.TFPyEnvironment(env_py)

with imageio.get_writer(filename, fps=60) as video:
    for _ in range(VIDEO_EPISODES):
        time_step = env.reset()
        video.append_data(env_py.render())
        while not time_step.is_last():
            action = dqn.predict_action(time_step.observation)
            time_step = env.step(action)
            video.append_data(env_py.render())

embed_mp4(filename)
