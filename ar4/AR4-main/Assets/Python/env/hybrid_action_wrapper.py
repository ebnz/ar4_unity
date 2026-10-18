import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper


class HybridActionWrapper(VecEnvWrapper):
    """
    Wrapper that converts a Dict action space with continuous and discrete components
    into a single Box action space that SB3 can handle.

    The wrapper:
    1. Flattens the Dict action space into a single Box space for SB3
    2. Splits the flattened actions back into continuous and discrete parts for Unity
    """

    def __init__(self, venv):
        super().__init__(venv)

        # Check if we have a Dict action space
        if isinstance(self.venv.action_space, spaces.Dict):
            self.is_hybrid = True

            # Store original spaces
            self.original_action_space = self.venv.action_space

            # Extract continuous and discrete components
            self.has_continuous = "continuous" in self.original_action_space.spaces
            self.has_discrete = "discrete" in self.original_action_space.spaces

            # Calculate total action size
            total_size = 0

            if self.has_continuous:
                self.continuous_space = self.original_action_space["continuous"]
                self.continuous_size = np.prod(self.continuous_space.shape)
                total_size += self.continuous_size
            else:
                self.continuous_size = 0

            if self.has_discrete:
                self.discrete_space = self.original_action_space["discrete"]
                # For discrete actions, we'll use one-hot encoding or direct value
                if isinstance(self.discrete_space, spaces.Discrete):
                    self.discrete_size = 1  # Single discrete action
                    self.discrete_n = self.discrete_space.n
                elif isinstance(self.discrete_space, spaces.MultiDiscrete):
                    self.discrete_size = len(self.discrete_space.nvec)
                    self.discrete_n = self.discrete_space.nvec
                else:
                    raise NotImplementedError(
                        f"Discrete space type {type(self.discrete_space)} not supported"
                    )
                total_size += self.discrete_size
            else:
                self.discrete_size = 0

            # Create flattened Box action space for SB3
            # Use wider bounds for discrete part to allow clear separation
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(int(total_size),), dtype=np.float32
            )
        else:
            # Not a hybrid space, pass through
            self.is_hybrid = False
            self.action_space = self.venv.action_space

    def step_async(self, actions):
        if self.is_hybrid:
            # Convert flattened actions back to Dict format
            dict_actions = []
            for action in actions:
                action_dict = {}

                idx = 0
                # Extract continuous part
                if self.has_continuous:
                    continuous_end = idx + self.continuous_size
                    action_dict["continuous"] = action[idx:continuous_end]
                    idx = continuous_end

                # Extract discrete part
                if self.has_discrete:
                    discrete_end = idx + self.discrete_size
                    discrete_values = action[idx:discrete_end]

                    if isinstance(self.discrete_space, spaces.Discrete):
                        # Convert continuous value to discrete action
                        # Use tanh-like transformation and scale to action range
                        discrete_action = np.clip(discrete_values[0], -1, 1)
                        # Map from [-1, 1] to [0, n-1]
                        discrete_action = int(
                            (discrete_action + 1) * 0.5 * (self.discrete_n - 1) + 0.5
                        )
                        discrete_action = np.clip(
                            discrete_action, 0, self.discrete_n - 1
                        )
                        action_dict["discrete"] = discrete_action
                    else:  # MultiDiscrete
                        # Convert each continuous value to discrete
                        discrete_actions = []
                        for i, val in enumerate(discrete_values):
                            val = np.clip(val, -1, 1)
                            # Map from [-1, 1] to [0, n-1] for this branch
                            discrete_val = int(
                                (val + 1) * 0.5 * (self.discrete_n[i] - 1) + 0.5
                            )
                            discrete_val = np.clip(
                                discrete_val, 0, self.discrete_n[i] - 1
                            )
                            discrete_actions.append(discrete_val)
                        action_dict["discrete"] = np.array(discrete_actions)

                dict_actions.append(action_dict)

            # Pass Dict actions to underlying environment
            self.venv.step_async(dict_actions)
        else:
            # Pass through for non-hybrid spaces
            self.venv.step_async(actions)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()
