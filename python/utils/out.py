import base64
import os

import imageio
import IPython
import matplotlib.pyplot as plt
from tf_agents.environments import tf_py_environment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.policies import TFPolicy


def out_dir():
    return os.path.join(os.path.dirname(__file__), "../out/")


def write_graph(
    filename: str,
    *,
        num_iterations: int,
        eval_interval: int,
        avg_returns: list,
        max_return: int
):
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, avg_returns)
    plt.ylabel("Average Return")
    plt.xlabel("Step")
    plt.ylim(top=max_return)
    plt.savefig(out_dir() + filename)


def embed_mp4(filename: str):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = """
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4">
        </video>
    """.format(b64.decode())

    return IPython.display.HTML(tag)


VIDEO_EPISODES = 3


def write_video(filename: str, *, env: PyEnvironment, policy: TFPolicy):
    filename = out_dir() + filename
    env_py = env
    env = tf_py_environment.TFPyEnvironment(env_py)
    with imageio.get_writer(filename, fps=60) as video:
        for _ in range(VIDEO_EPISODES):
            time_step = env.reset()
            video.append_data(env_py.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)
                video.append_data(env_py.render())

    embed_mp4(filename)
