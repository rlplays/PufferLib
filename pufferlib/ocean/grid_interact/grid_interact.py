'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.grid_interact import binding

class GridInteract(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=1000, height=1000, cell_size=100,
            num_rewards=5, fov=10, render_mode=None, log_interval=128,
            size=11, buf=None, seed=0):
        # One hot encoded observation space ego-centric view of the grid from the agent's perspective.
        # Each agent observes the grid in a square of size fov x fov.
        # Number of cell types is at most cell_types. So the observation space is fov*fov*cell_types.
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(fov*fov*cell_types,), dtype=np.float32)

        # Action space: 5 discrete actions (up, down, left, right, stay).
        self.single_action_space = gymnasium.spaces.MultiDiscrete([5])

        self.render_mode = render_mode
        self.log_interval = log_interval

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(self.observations, self.actions,
                self.rewards, self.terminals, self.truncations,
                seed, width=width, height=height, cell_size=cell_size,
                num_rewards=num_rewards, fov=fov
                )
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 512

    env = GridInteract()
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.nvec, size=(CACHE, 2))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += 1
        i += 1

    print('GridInteract SPS:', int(steps / (time.time() - start)))
