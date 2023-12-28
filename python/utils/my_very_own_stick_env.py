# Most of the code comes from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

import math

import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step


class MyVeryOnwStickEnv(PyEnvironment):
    def __init__(self):
        super().__init__()

        self.rng = np.random.RandomState()

        self.reward_per_step = 1.0
        self.current_step = 0
        self.max_steps = 500

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self._action_spec = BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action'
        )
        self._observation_spec = BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation'
        )

        self.state = None
        self.episode_ended = False

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,))
        self.current_step = 0
        self.episode_ended = False
        return time_step.restart(np.array(self.state, dtype=np.float32))

    def _step(self, action: int):
        assert self.state is not None, "Call reset before using step method."

        if self.episode_ended:
            return self.reset()

        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.episode_ended = True
            return time_step.termination(
                np.array(self.state, dtype=np.float32), self.reward_per_step
            )

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        tmp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * tmp) / (
            self.length * (
                4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass
            )
        )
        xacc = tmp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if terminated:
            self.episode_ended = True
            return time_step.termination(
                np.array(self.state, dtype=np.float32), self.reward_per_step
            )

        return time_step.transition(
            np.array(self.state, dtype=np.float32), self.reward_per_step
        )

    def is_solved(self):
        return self.current_step >= (self.max_steps * 0.9)

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise Exception("pygame is not installed")

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface(
                (self.screen_width, self.screen_height)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0
        carty = 100
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
