import imageio
import IPython
from utils.a2c import A2C
from utils.my_very_own_stick_env import MyVeryOnwStickEnv
from utils.out import embed_mp4, out_dir, write_graph

a2c = A2C(
    env=MyVeryOnwStickEnv(),
    learning_rate=1e-3,
    sync_interval=100,
    gamma=0.99,
    fc_layer_params=(100, 100)
)

NUM_ITERATIONS = 10000
EVAL_INTERVAL = 2000

avg_returns = [a2c.evaluate_avg_return()]

for i in range(NUM_ITERATIONS // EVAL_INTERVAL):
    a2c.train(EVAL_INTERVAL)
    avg_return = a2c.evaluate_avg_return()
    avg_returns.append(avg_return)
    print(f"Step: {(i + 1) * EVAL_INTERVAL} | Average Return: {avg_return}")


write_graph(
    filename="a2c_graph.png",
    num_iterations=NUM_ITERATIONS,
    eval_interval=EVAL_INTERVAL,
    avg_returns=avg_returns,
    max_return=500
)

VIDEO_EPISODES = 3

filename = out_dir() + "a2c_video.mp4"
env = MyVeryOnwStickEnv()

with imageio.get_writer(filename, fps=60) as video:
    for _ in range(VIDEO_EPISODES):
        time_step = env.reset()
        video.append_data(env.render())
        while not time_step.is_last():
            action = a2c.agent.predict_action(time_step.observation)
            time_step = env.step(action)
            video.append_data(env.render())

embed_mp4(filename)
