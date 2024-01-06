import statistics
import time

from utils.a2c import A2C
from utils.my_very_own_stick_env import MyVeryOnwStickEnv

SAMPLE_SIZE = 50
MAX_ITERATION_COUNT = 500_000
EVAL_INTERVAL = 500

TARGET_AVG_RETURN = 475.0

durations = []

for _ in range(SAMPLE_SIZE):
    a2c = A2C(
        env=MyVeryOnwStickEnv(),
        learning_rate=1e-3,
        sync_interval=100,
        gamma=0.99,
        fc_layer_params=(100, 100)
    )

    start_time = time.perf_counter()

    for i in range(MAX_ITERATION_COUNT // EVAL_INTERVAL):
        a2c.train(EVAL_INTERVAL)
        avg_return = a2c.evaluate_avg_return()

        if avg_return >= TARGET_AVG_RETURN:
            break

        if (i + 1) * EVAL_INTERVAL == MAX_ITERATION_COUNT:
            raise RuntimeError("Failed to reach target average return")

    durations.append(time.perf_counter() - start_time)

print(f"Median duration: {statistics.median(durations)}s")
