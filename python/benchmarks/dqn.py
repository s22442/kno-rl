import statistics
import time

from utils.dqn import Dqn
from utils.my_very_own_stick_env import MyVeryOnwStickEnv

SAMPLE_SIZE = 20
MAX_ITERATION_COUNT = 500_000
EVAL_INTERVAL = 500

TARGET_AVG_RETURN = 475.0

durations = []

for _ in range(SAMPLE_SIZE):
    dqn = Dqn(
        env=MyVeryOnwStickEnv(),
        initial_collect_steps=200,
        collect_steps_per_iteration=10,
        replay_buffer_max_length=100_000,
        batch_size=200,
        learning_rate=1e-3,
        fc_layer_params=(100, 100)
    )

    start_time = time.perf_counter()

    for i in range(MAX_ITERATION_COUNT // EVAL_INTERVAL):
        dqn.train(EVAL_INTERVAL)
        avg_return = dqn.evaluate_avg_return()

        if avg_return >= TARGET_AVG_RETURN:
            break

        if (i + 1) * EVAL_INTERVAL == MAX_ITERATION_COUNT:
            raise RuntimeError("Failed to reach target average return")

    durations.append(time.perf_counter() - start_time)

print(f"Median duration: {statistics.median(durations)}s")
