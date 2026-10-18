from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)


class UnityVecEnv(VecEnv):
    """
    Vectorized wrapper for Unity ML-Agents environments.

    Key features:
    - Automatically detects and manages multiple agents in Unity
    - Handles individual agent resets (no global reset on episode end)
    - Maps actions only to agents requesting decisions
    - Supports both discrete and continuous action spaces
    """

    def __init__(
        self,
        file_name=None,
        base_port=5004,
        worker_id=0,
        no_graphics=False,
        timeout_wait=60,
        seed=0,
        time_scale=5.0,
    ):
        self.engine_config_channel = EngineConfigurationChannel()
        self.engine_config_channel.set_configuration_parameters(time_scale=time_scale)

        self.unity_env = UnityEnvironment(
            file_name=file_name,
            base_port=base_port,
            worker_id=worker_id,
            no_graphics=no_graphics,
            timeout_wait=timeout_wait,
            seed=seed,
            side_channels=[self.engine_config_channel],
        )

        # Reset to initialize and get behavior specs
        self.unity_env.reset()

        self.behavior_name = list(self.unity_env.behavior_specs.keys())[0]
        self.spec = self.unity_env.behavior_specs[self.behavior_name]

        # Discover agent_ids and num_envs
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        all_agent_ids_at_init = set()
        if decision_steps.agent_id.size > 0:
            all_agent_ids_at_init.update(decision_steps.agent_id)
        if (
            terminal_steps.agent_id.size > 0
        ):  # Should be empty after a full reset if all agents start active
            all_agent_ids_at_init.update(terminal_steps.agent_id)

        if not all_agent_ids_at_init:
            # Try one step if no agents found initially (e.g. environment waits for first step call)
            print(
                "UnityVecEnv: No agents found at initial get_steps. Attempting a step..."
            )
            self.unity_env.step()  # Trigger default actions / decision request
            decision_steps, terminal_steps = self.unity_env.get_steps(
                self.behavior_name
            )
            if decision_steps.agent_id.size > 0:
                all_agent_ids_at_init.update(decision_steps.agent_id)
            if terminal_steps.agent_id.size > 0:
                all_agent_ids_at_init.update(terminal_steps.agent_id)

        if not all_agent_ids_at_init:
            raise RuntimeError(
                "UnityVecEnv: Could not determine agent IDs after initial reset and step."
            )

        self.agent_ids_ordered = sorted(list(all_agent_ids_at_init))
        _num_envs = len(self.agent_ids_ordered)
        self.agent_id_to_vec_idx = {
            agent_id: i for i, agent_id in enumerate(self.agent_ids_ordered)
        }
        print(
            f"UnityVecEnv: Detected {_num_envs} agents with IDs: {self.agent_ids_ordered}"
        )

        # Define observation and action spaces (for a single agent logic, VecEnv handles batching)
        if len(self.spec.observation_specs) > 1:
            print(
                "Warning: Multiple observation sensors found. Using only the first one. For Dict spaces, this wrapper needs extension."
            )
        obs_spec = self.spec.observation_specs[0]

        # Determine observation dtype
        _obs_dtype = np.float32  # Default for ML-Agents vector observations
        # decision_steps and terminal_steps are from earlier in __init__
        if (
            len(decision_steps.obs) > 0
            and decision_steps.obs[0] is not None
            and decision_steps.obs[0].size > 0
        ):
            _obs_dtype = decision_steps.obs[0].dtype
        elif (
            len(terminal_steps.obs) > 0
            and terminal_steps.obs[0] is not None
            and terminal_steps.obs[0].size > 0
        ):
            _obs_dtype = terminal_steps.obs[0].dtype
            # print(f"UnityVecEnv: Inferred observation dtype from terminal_steps as {_obs_dtype}.") # Optional log
        else:
            # This case might happen if no agents send observations on the initial setup steps.
            print(
                "UnityVecEnv: Warning - Could not infer observation dtype from initial agent steps. "
                "Defaulting to np.float32. Check Unity agent configuration and observation sensors if this is unexpected."
            )

        _observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_spec.shape, dtype=_obs_dtype
        )

        # Check for hybrid action space (both continuous and discrete)
        has_continuous = self.spec.action_spec.continuous_size > 0
        has_discrete = len(self.spec.action_spec.discrete_branches) > 0

        if has_continuous and has_discrete:
            # Hybrid action space - use Dict space
            action_spaces = {}
            continuous_size = self.spec.action_spec.continuous_size
            action_spaces["continuous"] = spaces.Box(
                low=-1.0, high=1.0, shape=(continuous_size,), dtype=np.float32
            )
            # For multiple discrete branches, use MultiDiscrete
            if len(self.spec.action_spec.discrete_branches) > 1:
                action_spaces["discrete"] = spaces.MultiDiscrete(
                    self.spec.action_spec.discrete_branches
                )
            else:
                action_spaces["discrete"] = spaces.Discrete(
                    self.spec.action_spec.discrete_branches[0]
                )
            _action_space = spaces.Dict(action_spaces)
        elif has_discrete:
            # Pure discrete
            if len(self.spec.action_spec.discrete_branches) > 1:
                _action_space = spaces.MultiDiscrete(
                    self.spec.action_spec.discrete_branches
                )
            else:
                _action_space = spaces.Discrete(
                    self.spec.action_spec.discrete_branches[0]
                )
        else:
            # Pure continuous
            continuous_size = self.spec.action_spec.continuous_size
            _action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(continuous_size,), dtype=np.float32
            )

        super().__init__(_num_envs, _observation_space, _action_space)
        self.actions = None  # To store actions from step_async

    def reset(self):
        self.unity_env.reset()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        obs_array = np.zeros(
            (self.num_envs, *self.observation_space.shape),
            dtype=self.observation_space.dtype,
        )

        for agent_id in decision_steps.agent_id:
            if agent_id in self.agent_id_to_vec_idx:
                vec_idx = self.agent_id_to_vec_idx[agent_id]
                unity_idx = decision_steps.agent_id_to_index[agent_id]
                obs_array[vec_idx] = decision_steps.obs[0][unity_idx]

        for agent_id in terminal_steps.agent_id:
            if agent_id in self.agent_id_to_vec_idx:
                vec_idx = self.agent_id_to_vec_idx[agent_id]
                unity_idx = terminal_steps.agent_id_to_index[agent_id]
                obs_array[vec_idx] = terminal_steps.obs[0][unity_idx]
        return obs_array

    def step_async(self, actions):
        # SB3 actions are np.ndarray of shape (num_envs, *action_space.shape) for Box
        # or (num_envs,) for Discrete
        self.actions = actions

    def step_wait(self):
        # Get current decision steps to know which agents need actions
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)

        # Only send actions for agents that are requesting decisions
        if len(decision_steps) > 0:
            action_tuple = ActionTuple()

            # Map actions from vec_env indices to Unity's decision_steps order
            actions_for_unity = []
            for agent_id in decision_steps.agent_id:
                if agent_id in self.agent_id_to_vec_idx:
                    vec_idx = self.agent_id_to_vec_idx[agent_id]
                    actions_for_unity.append(self.actions[vec_idx])

            if actions_for_unity:
                # Handle different action space types
                if isinstance(self.action_space, spaces.Dict):
                    # Hybrid action space - extract continuous and discrete parts
                    continuous_actions = []
                    discrete_actions = []

                    for action in actions_for_unity:
                        if "continuous" in action:
                            continuous_actions.append(action["continuous"])
                        if "discrete" in action:
                            discrete_actions.append(action["discrete"])

                    # Add continuous actions if present
                    if continuous_actions:
                        continuous_array = np.array(continuous_actions)
                        action_tuple.add_continuous(continuous_array)

                    # Add discrete actions if present
                    if discrete_actions:
                        discrete_array = np.array(discrete_actions)
                        # Ensure proper shape for discrete actions
                        if len(discrete_array.shape) == 1:
                            discrete_array = discrete_array.reshape(-1, 1)
                        action_tuple.add_discrete(discrete_array)

                elif isinstance(self.action_space, spaces.Discrete):
                    # Pure discrete actions
                    actions_array = np.array(actions_for_unity)
                    # Single branch: reshape from (num_agents,) to (num_agents, 1)
                    action_tuple.add_discrete(actions_array.reshape(-1, 1))

                elif isinstance(self.action_space, spaces.MultiDiscrete):
                    # Multiple discrete branches
                    actions_array = np.array(actions_for_unity)
                    action_tuple.add_discrete(actions_array)

                else:  # Box - pure continuous
                    actions_array = np.array(actions_for_unity)
                    action_tuple.add_continuous(actions_array)

                self.unity_env.set_actions(self.behavior_name, action_tuple)

        self.unity_env.step()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        current_obs_arr = np.zeros(
            (self.num_envs, *self.observation_space.shape),
            dtype=self.observation_space.dtype,
        )
        rewards_arr = np.zeros(self.num_envs, dtype=np.float32)
        dones_arr = np.zeros(self.num_envs, dtype=bool)
        infos_list = [{} for _ in range(self.num_envs)]

        terminated_agents_this_step_vec_indices = set()

        for agent_id in terminal_steps.agent_id:
            if agent_id not in self.agent_id_to_vec_idx:
                continue

            vec_idx = self.agent_id_to_vec_idx[agent_id]
            unity_idx = terminal_steps.agent_id_to_index[agent_id]

            # Store terminal observation in infos for SB3
            infos_list[vec_idx]["terminal_observation"] = terminal_steps.obs[0][
                unity_idx
            ].copy()
            rewards_arr[vec_idx] = terminal_steps.reward[unity_idx]
            dones_arr[vec_idx] = True
            if terminal_steps.interrupted[unity_idx]:
                infos_list[vec_idx]["TimeLimit.truncated"] = True
            terminated_agents_this_step_vec_indices.add(vec_idx)

        for agent_id in decision_steps.agent_id:
            if agent_id not in self.agent_id_to_vec_idx:
                continue
            vec_idx = self.agent_id_to_vec_idx[agent_id]

            unity_idx = decision_steps.agent_id_to_index[agent_id]
            current_obs_arr[vec_idx] = decision_steps.obs[0][unity_idx]

            # If this agent was just reset (was in terminal_steps last step),
            # this is its new initial observation
            if vec_idx not in terminated_agents_this_step_vec_indices:
                rewards_arr[vec_idx] = decision_steps.reward[unity_idx]
                dones_arr[vec_idx] = False

        # No global reset needed! Unity automatically resets individual agents.
        # For done agents, Unity has already called OnEpisodeBegin() and they'll
        # appear in decision_steps next step with their new initial observations.
        # For done agents, we return their last observation from terminal_steps
        # (stored in infos), and dones_arr[i] = True tells SB3 the episode ended.

        return current_obs_arr, rewards_arr, dones_arr, infos_list

    def close(self):
        self.unity_env.close()

    def get_attr(self, attr_name, indices=None):
        # Simplified: attempt to get attr from self.unity_env if not on VecEnv itself
        if hasattr(self, attr_name):
            return [getattr(self, attr_name) for _ in range(self.num_envs)]
        if hasattr(self.unity_env, attr_name):
            return [getattr(self.unity_env, attr_name) for _ in range(self.num_envs)]
        raise AttributeError(
            f"Attribute {attr_name} not found in VecEnv or underlying UnityEnvironment."
        )

    def set_attr(self, attr_name, value, indices=None):
        # Simplified: attempt to set attr on self.unity_env
        # This is tricky; Unity env settings are often global.
        # Assuming `value` is a single value to be set globally if it's for unity_env
        if hasattr(
            self, attr_name
        ):  # Check if it's an attribute of the VecEnv wrapper itself
            setattr(self, attr_name, value)
        elif hasattr(self.unity_env, attr_name):
            setattr(self.unity_env, attr_name, value)
        else:
            # Allow setting new attributes on the VecEnv wrapper itself
            setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # Call a method on the underlying Unity environment.
        # Assumes the method is called once on self.unity_env.
        results = [
            getattr(self.unity_env, method_name)(*method_args, **method_kwargs)
            for _ in range(self.num_envs)
        ]
        return (
            results[0] if self.num_envs == 1 else results
        )  # SB3 might expect a list or single if num_envs=1

    def seed(self, seed=None):
        # Set seed for the Unity environment via side channel
        # SB3 calls this with a single seed integer.
        if seed is not None:
            self.engine_config_channel.set_configuration_parameters(seed=seed)
        # Note: For VecEnv, SB3 might expect a list of seeds if it were to seed sub-envs individually.
        # Here, one seed seeds the entire Unity instance.

    def env_is_wrapped(self, wrapper_class, indices=None):
        # Check if this VecEnv (or its underlying env, though not applicable here) is wrapped by wrapper_class.
        if indices is None:
            indices = range(self.num_envs)
        return [isinstance(self, wrapper_class) for _ in indices]


if __name__ == "__main__":
    # Example Usage (requires a Unity environment to be running or a build path)
    print("Attempting to initialize UnityVecEnv...")
    try:
        # Replace 'your_unity_env.x86_64' with your actual environment file if needed
        # or ensure Unity Editor is running and listening on base_port 5004.
        env = UnityVecEnv(file_name=None, base_port=5004)
        print(f"Initialized UnityVecEnv with {env.num_envs} agents.")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        for i in range(5):
            actions = [env.action_space.sample() for _ in range(env.num_envs)]
            actions = np.array(actions)

            # For discrete actions, SB3 passes (num_envs,), ensure it matches if action_space is Discrete
            if isinstance(env.action_space, spaces.Discrete):
                # Action space sample gives single int, stack them
                pass  # np.array(actions) is already (num_envs,)
            elif isinstance(env.action_space, spaces.Box):
                # Action space sample gives (action_dim,), stack them to (num_envs, action_dim)
                pass  # np.array(actions) is already (num_envs, action_dim)

            new_obs, rewards, dones, infos = env.step(actions)
            print(f"Step {i+1}:")
            print(f"  Observations shape: {new_obs.shape}")
            print(f"  Rewards: {rewards}")
            print(f"  Dones: {dones}")
            # print(f"  Infos: {infos}")
            if np.any(dones):
                print(
                    "  An agent is done. Environment should have reset if logic is correct."
                )
                # obs = env.reset() # Not needed, step handles global reset return values

        env.close()
        print("UnityVecEnv example usage finished.")
    except Exception as e:
        print(f"Error during UnityVecEnv example: {e}")
        import traceback

        traceback.print_exc()
