# AR4 Robot Reinforcement Learning Training Setup

This document explains the Python-based reinforcement learning training pipeline for the AR4 robot using Unity ML-Agents and Stable Baselines3 (SB3).

## Overview

The training system consists of three main components:
1. **Unity Environment** - The AR4 robot simulation running in Unity with ML-Agents
2. **Python Wrappers** - Custom Gym/Gymnasium wrappers to interface with Unity
3. **RL Training** - Stable Baselines3 for training PPO agents

## File Structure

```
Python/
├── train_sb3.py          # Main SB3 training script (current)
├── train.py              # Legacy SBX training script
├── train_sac.py          # Legacy SAC training with SBX
├── unity_vec_env.py      # Vectorized environment wrapper
├── unity_gym_wrapper.py  # Basic single-agent Gym wrapper
├── llm.py                # Azure OpenAI LLM client
├── reward_llm.py         # LLM client for reward generation
├── extract_env_info.py   # Unity environment analysis utilities
├── visualize_training.py # Training visualization and plotting
├── launch_tensorboard.py # TensorBoard launching utility
├── test_simplified_system.py # Testing utilities
├── setup_env.ps1         # Environment setup script
├── requirements.txt      # Python dependencies
├── eureka_main.py        # Main Eureka loop
├── run_eureka.py         # Eureka runner script
├── training_orchestrator.py # Training coordination
├── conversation_utils.py # LLM conversation utilities
├── test_results_feedback.py # Result analysis
├── test_llm_messages.py  # LLM message testing utility
├── eureka_README.md      # Eureka system documentation
├── config/               # Configuration files
│   ├── eureka_config.yaml # Main Eureka configuration
│   ├── eureka_config_T0.yaml # Task T0 config
│   └── eureka_config_test.yaml # Test configuration
└── prompts/              # LLM prompt templates
    ├── system_prompt.py  # System prompts
    ├── code_feedback.py  # Code feedback prompts
    └── policy_feedback.py # Policy feedback prompts
```

## Components Breakdown

### 1. train_sb3.py - Main SB3 Training Script (Current)

This is the current entry point for training using Stable Baselines3. It:
- Creates a vectorized Unity environment with 5x time scale for faster training
- Wraps it with VecNormalize for observation/reward normalization
- Initializes a PPO agent from SB3 (Stable Baselines3)
- Trains for configurable timesteps with progress tracking
- Includes callbacks for evaluation, checkpointing, and monitoring
- Ensures proper environment cleanup

Key features:
- Uses PPO with MLP policy (fully connected neural network)
- Time scale of 5x speeds up simulation during training
- Automatic environment normalization for better learning stability
- TensorBoard integration for training visualization
- Checkpoint saving during training

### 1b. Legacy Training Scripts

- **train.py**: Legacy training script using SBX (Stable Baselines JAX)
- **train_sac.py**: Legacy SAC algorithm training with SBX

### 2. unity_vec_env.py - Vectorized Environment Wrapper

This is the main wrapper that enables efficient parallel training with multiple Unity agents:

**Key Features:**
- **Auto-detection of agents**: Automatically discovers all agents in the Unity scene
- **Vectorized interface**: Implements `VecEnv` API for parallel environment stepping
- **Flexible action/observation spaces**: Supports both discrete and continuous actions
- **Individual agent resets**: Handles Unity's per-agent reset behavior correctly
- **Time scale configuration**: Allows speeding up simulation via side channels
- **Agent ID mapping**: Maintains consistent agent ordering across episodes

**Technical Details:**
- Uses ML-Agents' `EngineConfigurationChannel` for time scale control
- Handles both decision steps (ongoing agents) and terminal steps (done agents)
- Provides proper terminal observations for done agents (SB3 requirement)
- Supports seeding for reproducible training

### 3. unity_gym_wrapper.py - Basic Gymnasium Wrapper

A simpler single-agent wrapper that:
- Provides standard Gymnasium (formerly OpenAI Gym) interface
- Handles observation and action space conversion
- Manages episode resets and step logic
- Best for testing or single-agent scenarios

**Note**: The vectorized wrapper (unity_vec_env.py) is preferred for actual training as it's more efficient.

## 4. Eureka System - LLM-Based Reward Engineering

The Python directory now includes a complete implementation of the Eureka methodology for automatically generating and iterating on reward functions using Large Language Models (LLMs).

### Core Components

#### eureka_main.py - Main Eureka Loop
The central orchestrator that implements the Eureka algorithm:
- **LLM Integration**: Uses Azure OpenAI or OpenAI APIs to generate reward functions
- **Iterative Improvement**: Runs multiple iterations to refine reward functions based on performance
- **Multi-Sample Generation**: Creates multiple reward function candidates per iteration
- **Performance Analysis**: Evaluates and ranks different reward functions
- **Conversation Management**: Maintains context across iterations for better LLM responses

#### training_orchestrator.py - Training Coordination
Manages the training process for different reward functions:
- **Subprocess Management**: Launches Unity training runs with different reward functions
- **Performance Monitoring**: Tracks training metrics and success rates
- **Resource Management**: Handles parallel training runs and cleanup
- **Result Collection**: Aggregates training statistics for feedback

#### reward_llm.py - LLM Client
Specialized LLM client for reward function generation:
- **Multi-Provider Support**: Works with both Azure OpenAI and standard OpenAI
- **Prompt Management**: Handles system prompts and conversation context
- **Code Extraction**: Parses C# reward functions from LLM responses
- **Error Handling**: Robust error handling for LLM API calls
- **Token Management**: Tracks usage and costs

### Configuration System

#### eureka_config.yaml - Main Configuration
Defines the overall Eureka setup:
```yaml
model: o4-mini
iterations: 5
samples: 4
max_training_steps: 100000
tasks:
  T0: "Move to Target"
  T1: "Static Pick"
  T2: "Pick and Place"
  # ... more tasks
```

#### Task-Specific Configs
- **eureka_config_T0.yaml**: Configuration for basic movement tasks
- **eureka_config_test.yaml**: Minimal configuration for testing

### Prompt Engineering

The `prompts/` directory contains carefully crafted prompts for different aspects:

#### system_prompt.py - Core System Prompt
Defines the LLM's role as a reward engineer and provides context about:
- The robot environment and available observations
- Reward function structure and requirements
- Success criteria and evaluation metrics
- C# coding conventions for Unity integration

#### code_feedback.py - Code Feedback Prompts
Templates for providing feedback on generated reward functions:
- Compilation error analysis
- Runtime error debugging
- Performance improvement suggestions
- Code quality feedback

#### policy_feedback.py - Policy Feedback Prompts
Templates for analyzing training performance:
- Success rate analysis
- Learning curve interpretation
- Behavior analysis and debugging
- Convergence diagnostics

### Eureka Workflow

1. **Initialization**: Load task configuration and initialize LLM client
2. **Generation Phase**: LLM generates multiple reward function candidates
3. **Training Phase**: Each reward function is trained in Unity for specified steps
4. **Evaluation Phase**: Training results are collected and analyzed
5. **Feedback Phase**: Performance data is converted to natural language feedback
6. **Iteration**: Process repeats with refined prompts based on previous results
7. **Selection**: Best performing reward function is selected

### Integration with Unity

The Eureka system integrates with Unity through:
- **Dynamic Compilation**: C# reward functions are compiled at runtime using Roslyn
- **Command Line Interface**: Unity accepts reward function files via command line args
- **Result Export**: Training statistics are exported as JSON for analysis
- **File System Communication**: Reward functions and results are exchanged via shared directories

### Supported Tasks

The system supports various robotic manipulation tasks:
- **T0**: Move end effector to fixed target position
- **T1**: Reach and grab a fixed cube object
- **T2**: Pick up cube and place at fixed location
- **T3**: Pick and place with randomized positions
- **T4**: Vision-based pick and place using camera
- **T5**: Multi-object stacking tasks

For detailed Eureka documentation, see `eureka_README.md`.

## 5. Utility Scripts and Development Tools

### extract_env_info.py - Environment Analysis
Parses Unity C# scripts to extract environment information:
- **Agent Analysis**: Extracts observation and action space definitions
- **Variable Detection**: Identifies available variables for reward functions
- **Environment Documentation**: Generates structured information for LLM prompts
- **Code Parsing**: Uses regex patterns to analyze Unity ML-Agents code

### visualize_training.py - Training Visualization
Creates plots and visualizations from training data:
- **TensorBoard Integration**: Loads and processes TensorBoard logs
- **Learning Curves**: Generates reward and performance plots
- **Comparison Charts**: Compares different training runs
- **Export Functionality**: Saves plots as images for documentation

### launch_tensorboard.py - TensorBoard Launcher
Utility to start TensorBoard monitoring:
- **Automatic Discovery**: Finds training log directories
- **Port Management**: Handles port conflicts and multiple instances
- **Browser Integration**: Automatically opens TensorBoard in browser

### test_simplified_system.py - Testing Framework
Testing utilities for the training system:
- **Environment Testing**: Validates Unity connection and communication
- **Wrapper Testing**: Tests vectorized environment functionality
- **Integration Testing**: End-to-end system validation
- **Performance Benchmarks**: Measures training performance metrics

### setup_env.ps1 - Environment Setup
PowerShell script for setting up the Python environment:
- **Conda Environment**: Creates and configures conda environment
- **Dependency Installation**: Installs all required packages from requirements.txt
- **Environment Validation**: Tests installation success

## Dependencies

From requirements.txt:
- **mlagents==1.1.0**: Unity ML-Agents Python package for communication
- **stable-baselines3[extra]**: Stable Baselines3 - PyTorch implementation of RL algorithms
- **tensorboard>=2.10.0**: Training visualization and monitoring
- **matplotlib>=3.5.0**: Plotting and data visualization
- **openai==1.77.0**: OpenAI API client for LLM integration
- **azure-identity==1.23.0**: Azure authentication for Azure OpenAI
- **python-dotenv==1.1.0**: Environment variable management
- **onnx>=1.12.0**: ONNX model format support
- **onnxruntime>=1.13.0**: ONNX runtime for model inference
- **psutil>=5.9.0**: System and process monitoring utilities

### Additional Development Dependencies
The system also supports various optional components:
- **PyTorch**: Required by Stable Baselines3 for neural network training
- **NumPy**: Scientific computing and array operations
- **Gymnasium**: Reinforcement learning environment interface
- **JAX** (legacy): For SBX-based legacy training scripts

## Training Workflow

1. **Unity Setup**: The Unity environment must be running (either in Editor or as a build) with ML-Agents configured agents
2. **Environment Connection**: Python connects to Unity via port 5004 (configurable)
3. **Agent Discovery**: The wrapper automatically finds all training agents in the scene
4. **Training Loop**:
   - Collect observations from all agents
   - Feed to PPO policy network
   - Execute actions in Unity
   - Collect rewards and next observations
   - Update policy using PPO algorithm
5. **Normalization**: VecNormalize tracks running statistics for stable training

## Key Concepts

### Vectorized Environments
The `UnityVecEnv` allows training multiple robot instances in parallel within a single Unity scene. This significantly speeds up data collection and training efficiency.

### Action Spaces
- **Continuous**: Used for joint control (e.g., joint velocities/torques)
- **Discrete**: Used for high-level actions (e.g., predefined movements)

### Observation Spaces
The wrapper expects vector observations from Unity containing the robot's state (joint angles, velocities, target positions, etc.).

### Reset Behavior
Unity ML-Agents handles resets individually for each agent:
1. When an agent calls `EndEpisode()` in Unity, it appears in `terminal_steps`
2. Unity automatically calls `OnEpisodeBegin()` for that agent
3. The agent will appear in `decision_steps` next step with its new initial observation
4. No global reset is needed - each agent manages its own episode lifecycle

## Detailed VecEnv Architecture and Data Flow

### Overview of Communication Layers

The training system involves three layers of communication:

```
Unity (C#) <--> ML-Agents Python API <--> UnityVecEnv Wrapper <--> Stable Baselines 3
```

### 1. Unity Side (C# Agent Implementation)

In Unity, each AR4 robot agent:
- Inherits from `Unity.MLAgents.Agent`
- Implements key methods:
  - `OnEpisodeBegin()`: Resets robot state
  - `CollectObservations()`: Sends state data to Python
  - `OnActionReceived()`: Receives and executes actions from Python
  - `EndEpisode()`: Called when task completes or fails

Example flow in Unity:
```csharp
// Each FixedUpdate cycle:
1. RequestDecision() - Agent asks for an action
2. CollectObservations() - Sends current state
3. Waits for Python response
4. OnActionReceived() - Executes the action
5. AddReward() - Accumulates rewards
6. If done: EndEpisode() - Triggers reset
```

### 2. ML-Agents Python API Layer

ML-Agents provides:
- `UnityEnvironment`: Main communication interface
- `DecisionSteps` & `TerminalSteps`: Agent state containers
- `ActionTuple`: Action container
- Side channels for configuration (time scale, graphics, etc.)

Key data structures:
```python
# DecisionSteps contains agents still active:
- obs: List of observation arrays
- reward: Rewards since last step
- agent_id: Active agent IDs
- action_mask: Optional action masks

# TerminalSteps contains agents that ended:
- obs: Final observations
- reward: Final rewards
- agent_id: Terminated agent IDs
- interrupted: Whether max steps reached
```

### 3. UnityVecEnv Wrapper Implementation

The wrapper bridges ML-Agents and SB3's VecEnv interface:

#### Initialization Phase:
```python
1. Connect to Unity via UnityEnvironment
2. Discover all agents and assign consistent IDs
3. Create agent_id_to_vec_idx mapping
4. Define observation/action spaces
5. Set time scale via side channel
```

#### Step Execution Flow:

**Step 1: Action Distribution**
```python
def step_async(self, actions):
    # SB3 provides actions shape: (num_envs, action_dim)
    # Store for later processing
    self.actions = actions
```

**Step 2: Action Mapping and Execution**
```python
def step_wait(self):
    # Get agents requesting decisions
    decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
    
    # Map actions to active agents only
    action_tuple = ActionTuple()
    for agent_id in decision_steps.agent_id:
        vec_idx = self.agent_id_to_vec_idx[agent_id]
        action = self.actions[vec_idx]
        # Add to action tuple in Unity's expected order
    
    # Send to Unity
    self.unity_env.set_actions(self.behavior_name, action_tuple)
    self.unity_env.step()
```

**Step 3: Observation Collection**
```python
    # After Unity steps, collect results
    decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
    
    # Initialize arrays for SB3
    obs_array = np.zeros((self.num_envs, *self.observation_space.shape))
    rewards_array = np.zeros(self.num_envs)
    dones_array = np.zeros(self.num_envs, dtype=bool)
    
    # Process terminal agents (episode ended)
    for agent_id in terminal_steps.agent_id:
        vec_idx = self.agent_id_to_vec_idx[agent_id]
        rewards_array[vec_idx] = terminal_steps.reward[...]
        dones_array[vec_idx] = True
        # Store terminal obs in infos
    
    # Process active agents
    for agent_id in decision_steps.agent_id:
        vec_idx = self.agent_id_to_vec_idx[agent_id]
        obs_array[vec_idx] = decision_steps.obs[0][...]
        rewards_array[vec_idx] = decision_steps.reward[...]
```

### 4. Data Flow Example

Let's trace a complete step with 3 agents:

**Initial State:**
- Agent IDs: [0, 1, 2]
- Vec indices: [0, 1, 2]

**Step 1: SB3 requests actions**
```python
actions = model.predict(observations)  # Shape: (3, action_dim)
env.step(actions)
```

**Step 2: VecEnv processes**
```python
# Agent 1 just completed its episode this step
decision_steps.agent_id = [0, 2]  # Agents still active
terminal_steps.agent_id = [1]     # Agent 1 just finished

# Map actions for active agents only
actions_for_unity = [actions[0], actions[2]]  # Only for decision_steps agents
```

**Step 3: Unity executes**
- Agents 0 and 2 receive and execute actions
- Agent 1 is in `OnEpisodeBegin()`, preparing for next episode

**Step 4: Next step**
```python
decision_steps.agent_id = [0, 1, 2]  # Agent 1 is back
terminal_steps.agent_id = []

# Agent 1 now has its new initial observation
```

### 5. Critical Implementation Details

#### Agent ID Consistency
- Unity may spawn agents with any ID
- VecEnv maintains `agent_ids_ordered` for consistent indexing
- Maps between vec_env index and Unity agent ID

#### Action Synchronization
- Only send actions for agents in `decision_steps`
- Actions must be in Unity's internal agent order
- Missing agents don't receive actions

#### Observation Handling
- Terminal agents: Last observation stored in `infos['terminal_observation']`
- Reset agents: First observation of new episode in next `decision_steps`
- Active agents: Current observation in `decision_steps`

#### Performance Optimizations
- Time scale acceleration (e.g., 5x)
- Batched agent stepping
- Efficient numpy array operations
- No unnecessary resets

### 6. Common Pitfalls and Solutions

**Pitfall 1: Assuming fixed agent ordering**
- Solution: Always use agent_id mappings

**Pitfall 2: Sending actions for all agents**
- Solution: Filter actions for decision_steps agents only

**Pitfall 3: Global resets on episode end**
- Solution: Let Unity handle individual resets

**Pitfall 4: Mismatched observation shapes**
- Solution: Verify Unity's CollectObservations matches Python space

## Visual Communication Flow

### Training Loop Sequence Diagram

```
┌─────────┐     ┌──────────┐     ┌───────────┐     ┌─────────┐
│   SB3   │     │ VecEnv   │     │ ML-Agents │     │  Unity  │
│  (PPO)  │     │ Wrapper  │     │    API    │     │ (Agents)│
└────┬────┘     └────┬─────┘     └─────┬─────┘     └────┬────┘
     │               │                   │                 │
     │  actions[3]   │                   │                 │
     ├──────────────>│                   │                 │
     │               │                   │                 │
     │               │ get_steps()       │                 │
     │               ├──────────────────>│                 │
     │               │                   │                 │
     │               │<──────────────────┤                 │
     │               │ decision_steps:[0,2]                │
     │               │ terminal_steps:[]                   │
     │               │                   │                 │
     │               │ Map actions       │                 │
     │               │ [0,2] only       │                 │
     │               │                   │                 │
     │               │ set_actions()     │                 │
     │               ├──────────────────>│ actions[0,2]   │
     │               │                   ├────────────────>│
     │               │                   │                 │
     │               │                   │     Execute     │
     │               │                   │     Actions     │
     │               │                   │                 │
     │               │ step()            │                 │
     │               ├──────────────────>│     step()      │
     │               │                   ├────────────────>│
     │               │                   │                 │
     │               │                   │   observations  │
     │               │                   │<────────────────┤
     │               │ get_steps()       │    rewards      │
     │               ├──────────────────>│                 │
     │               │                   │                 │
     │               │<──────────────────┤                 │
     │               │ decision_steps:[0,1,2]              │
     │               │ terminal_steps:[]                   │
     │               │                   │                 │
     │               │ Build arrays      │                 │
     │               │ obs[3], rew[3]   │                 │
     │               │ done[3], info[3] │                 │
     │               │                   │                 │
     │<──────────────┤                   │                 │
     │  obs, rewards │                   │                 │
     │  dones, infos │                   │                 │
     │               │                   │                 │
```

### Agent State Transitions

```
Unity Agent States:
                    ┌─────────────┐
                    │   Active    │
                    │ (decision)  │
                    └──────┬──────┘
                           │
                           │ EndEpisode()
                           ▼
                    ┌─────────────┐
                    │  Terminal   │
                    │   (done)    │
                    └──────┬──────┘
                           │
                           │ Auto-reset
                           ▼
                    ┌─────────────┐
                    │  Resetting  │
                    │ (invisible) │
                    └──────┬──────┘
                           │
                           │ OnEpisodeBegin()
                           ▼
                    ┌─────────────┐
                    │   Active    │
                    │ (decision)  │
                    └─────────────┘

Python View:
- Active agents: Appear in decision_steps
- Terminal agents: Appear in terminal_steps (one step only)
- Resetting agents: Not visible (handled internally by Unity)
```

### Multi-Agent Action Mapping Example

```
Step N:
┌─────────────────────────────────────────────────┐
│ SB3 Actions Array (all agents)                  │
│ ┌────────┬────────┬────────┐                   │
│ │ Act[0] │ Act[1] │ Act[2] │                   │
│ └────────┴────────┴────────┘                   │
└─────────────────────────────────────────────────┘
                    │
                    ▼ VecEnv maps to active agents
┌─────────────────────────────────────────────────┐
│ Unity Decision Steps (active agents only)       │
│ ┌────────┬────────┐                            │
│ │ Act[0] │ Act[2] │  (Agent 1 is resetting)   │
│ └────────┴────────┘                            │
└─────────────────────────────────────────────────┘

Step N+1:
┌─────────────────────────────────────────────────┐
│ Unity Decision Steps (all agents active)        │
│ ┌────────┬────────┬────────┐                   │
│ │ Obs[0] │ Obs[1] │ Obs[2] │  Agent 1 is back │
│ └────────┴────────┴────────┘                   │
└─────────────────────────────────────────────────┘
```

## Usage Example

Basic SB3 training:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from unity_vec_env import UnityVecEnv

# Create environment with 5x time scale
env = UnityVecEnv(time_scale=5.0)
env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

# Setup callbacks for checkpointing and evaluation
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/")

# Train
model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

# Save final model
model.save("ar4_robot_ppo_final")
```

Eureka-based training (handles Unity launching automatically):
```python
# Run the Eureka loop for automatic reward function generation
python eureka_main.py

# Or run with specific configuration
python run_eureka.py --config config/eureka_config_T0.yaml
```

## Configuration Notes

- **Port**: Default is 5004, ensure Unity is configured to use the same port
- **Time Scale**: Higher values (e.g., 5-10) speed up training but may affect physics stability
- **No Graphics**: Can be enabled for headless training on servers
- **Worker ID**: Allows running multiple Unity instances for distributed training

## Troubleshooting

1. **No agents found**: Ensure Unity agents have "Behavior Parameters" component configured
2. **Connection timeout**: Check Unity is running and port matches
3. **Observation shape mismatch**: Verify Unity observation configuration matches expected shape
4. **Action errors**: Ensure action space (continuous/discrete) matches Unity configuration

## Future Enhancements

Potential improvements:
- Add support for visual observations (camera inputs)
- Implement recurrent policies for temporal reasoning
- Add curriculum learning support
- Integrate with more advanced RL algorithms (SAC, TD3)
- Add checkpoint saving/loading during training
- Implement custom reward shaping utilities