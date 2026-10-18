# minimal gym wrapper
from gymnasium import Env, spaces
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class UnityGymWrapper(Env):
    def __init__(self, base_port=5004):
        self.env = UnityEnvironment(file_name=None, base_port=base_port)
        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        spec = self.env.behavior_specs[self.behavior_name]

        # action + observation space
        if spec.action_spec.is_discrete():
            self.action_space = spaces.Discrete(spec.action_spec.discrete_branches[0])
        else:
            self.action_space = spaces.Box(
                -1.0, 1.0, shape=(spec.action_spec.continuous_size,), dtype=np.float32
            )

        obs_shape = spec.observation_specs[0].shape
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=obs_shape, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        obs = decision_steps.obs[0][0]
        return obs, {}

    def step(self, action):
        action_tuple = ActionTuple()
        if isinstance(self.action_space, spaces.Discrete):
            action_tuple.add_discrete(np.array([[action]]))
        else:
            action_tuple.add_continuous(np.array([action]))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if len(terminal_steps):
            step = terminal_steps
            done = True
        else:
            step = decision_steps
            done = False

        obs = step.obs[0][0]
        reward = step.reward[0]
        return obs, reward, done, False, {}

    def close(self):
        self.env.close()
