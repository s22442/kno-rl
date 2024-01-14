import IPython
from utils.dqn import Dqn
from utils.my_very_own_stick_env import MyVeryOnwStickEnv
from utils.out import write_graph, write_video

dqn = Dqn(
    env=MyVeryOnwStickEnv(),
    initial_collect_steps=200,
    collect_steps_per_iteration=10,
    replay_buffer_max_length=100_000,
    batch_size=200,
    learning_rate=1e-3,
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
    filename="dqn_graph.png",
    num_iterations=NUM_ITERATIONS,
    eval_interval=EVAL_INTERVAL,
    avg_returns=avg_returns,
    max_return=500
)

write_video(
    "dqn_video.mp4",
    env=MyVeryOnwStickEnv(),
    policy=dqn.get_policy()
)
