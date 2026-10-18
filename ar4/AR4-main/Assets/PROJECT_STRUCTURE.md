# Eureka Unity AR4 Robot Project Structure

## Overview

This project implements Eureka-style dynamic reward function generation for training a Unity-simulated AR4 robot arm using reinforcement learning. The system uses LLMs to generate C# reward functions, which are dynamically compiled and evaluated in Unity while training with Stable Baselines3.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Python/LLM    │────▶│   File System    │────▶│  Unity Build    │
│  Reward Gen     │     │  .cs Functions   │     │  + Roslyn       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                                 │
         │                                                 ▼
         ▼                                        ┌─────────────────┐
┌─────────────────┐                               │ RobotAgentDynamic│
│ Stable Baselines│◀──────── ML-Agents ──────────│  (Pure Behavior)│
│   PPO Training  │         Connection            └─────────────────┘
└─────────────────┘                                        │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  ConfigManager  │
                                                  │  (Configuration)│
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │TrainingStatsLogger│
                                                  │   (Statistics)  │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │RewardFunctionMgr│
                                                  │ (Reward Logic)  │
                                                  └─────────────────┘
```

### Separation of Concerns

The Unity architecture now follows clean separation of concerns:

- **RobotAgentDynamic**: Pure ML-Agents behavior (observations, actions, episodes)
- **ConfigManager**: Centralized configuration management and command line parsing
- **TrainingStatsLogger**: Episode tracking, statistics calculation, and result output
- **RewardFunctionManager**: Dynamic reward compilation and execution

## Project Structure

```
AR4/Assets/
├── Scripts/
│   ├── ml/
│   │   ├── RobotAgentDynamic.cs         # ML-Agents agent with dynamic rewards
│   │   ├── RewardFunctionManager.cs     # Centralized reward compilation/execution
│   │   ├── RewardFunctionManagerInit.cs # Inspector-editable reward functions (MonoBehaviour)
│   │   ├── ConfigManager.cs             # Command line parsing and configuration
│   │   ├── ConfigManagerInit.cs         # Configuration initialization (MonoBehaviour)
│   │   ├── TrainingStatsLogger.cs       # Training statistics and Eureka result tracking
│   │   ├── RoslynScriptTest.cs          # Roslyn testing utilities
│   │   └── UnityExpressionEvaluator.cs  # Expression evaluation helpers
│   ├── RobotManager.cs               # Robot joint control
│   ├── MoveToTarget.cs               # IK implementation
│   └── ArticulationBodyDriver.cs     # Joint physics control
│
├── Python/
│   ├── eureka_unity/
│   │   ├── eureka_main.py            # Main Eureka loop implementation
│   │   ├── training_orchestrator.py   # Unity subprocess + SB3 training
│   │   ├── README.md                 # Eureka integration guide
│   │   ├── config/
│   │   │   └── eureka_config.yaml    # Task and training configurations
│   │   └── prompts/
│   │       └── system_prompt.txt     # LLM prompt for reward generation
│   ├── train_sb3.py                  # Standalone SB3 training script
│   ├── unity_vec_env.py              # Unity ML-Agents environment wrapper
│   ├── test_eureka_integration.py    # Integration tests
│   └── test_sb3_training.py          # Training system tests
│
├── Configs/
│   ├── robot_wp0.yaml                # T0 task ML-Agents config
│   ├── robot_sac.yaml                # SAC algorithm config (ML-Agents)
│   └── ball.yaml                     # Additional task configs
│
└── GeneratedRewards/                 # Runtime-generated directory
    ├── rewards/                      # C# reward function files
    └── results/                      # Training results JSON
```

## Key Components

### Unity Side (C#)

#### RobotAgentDynamic.cs
- **ML-Agents Integration**: Standard Agent implementation with dynamic rewards
- **Configuration Integration**: Uses ConfigManager for all configuration needs
- **Statistics Integration**: Uses TrainingStatsLogger for episode tracking and result output
- **Reward Execution**: Delegates reward compilation/execution to RewardFunctionManager
- **Scene-Level Initialization**: Waits for RewardFunctionManager to be initialized by scene components
- **Clean Separation**: Focuses purely on agent behavior (observations, actions, episodes)

#### ConfigManager.cs
- **Command Line Parsing**: Accepts `--reward-id`, `--run-id`, `--max-steps`, `--output-file`, `--mlagents-port`
- **Configuration Storage**: Centralized storage for all agent configuration
- **Singleton Pattern**: Single instance accessible throughout the application
- **Port Management**: Tracks ML-Agents communication port for logging and debugging
- **Validation**: Ensures required configuration is present and valid
- **Dynamic Updates**: Allows runtime configuration changes

#### RewardFunctionManager.cs
- **Dynamic Compilation**: Uses Roslyn to compile C# reward functions at runtime
- **Multiple Sources**: Supports file-based, inspector, and built-in default rewards
- **Priority System**: File-based → Inspector → Built-in default
- **Runtime Recompilation**: Allows testing different reward functions during development
- **Centralized Execution**: All agents use the same compiled reward function

#### RewardFunctionManagerInit.cs (MonoBehaviour)
- **Inspector Editing**: Large TextArea for easy reward function prototyping
- **Scene-Level Initialization**: Uses Awake() to initialize before agents start
- **Runtime Testing**: Context menu options for recompilation and validation
- **Development Friendly**: Toggle between inspector and file-based rewards
- **Zero Code Changes**: Works seamlessly with Eureka production runs

#### TrainingStatsLogger.cs
- **Dual Mode Support**: Works with both legacy RobotAgent and new RobotAgentDynamic
- **Episode Tracking**: Tracks rewards, lengths, and success rates across episodes
- **Eureka Integration**: Writes Eureka-compatible JSON results for Python integration
- **Real-time Monitoring**: Provides current statistics for monitoring dashboards
- **Legacy Compatibility**: Maintains existing detailed logging format

#### Reward Function Format
```csharp
// Observations available via dictionary
float currentDistanceToGoal = observations["DistanceToGoal"];

// Context provides additional info
float distanceDelta = context.previousDistance - currentDistanceToGoal;

// Return reward components for analysis
var rewards = new Dictionary<string, float>();
rewards["MovingCloser"] = distanceDelta * 50f;
rewards["GoalReached"] = currentDistanceToGoal < context.goalThreshold ? 10f : 0f;

// Calculate total
float total = 0f;
foreach (var r in rewards.Values) total += r;

return new RobotAgentDynamic.RewardResult { Total = total, Components = rewards };
```

### Python Side

#### Training Flow (training_orchestrator.py)
1. **Write Reward**: Save LLM-generated reward to file with unique ID
2. **Launch Unity**: Start Unity subprocess with command line arguments
3. **Connect SB3**: Create ML-Agents connection to running Unity instance
4. **Train PPO**: Run Stable Baselines3 training loop
5. **Collect Results**: Gather statistics from both SB3 and Unity

#### Eureka Loop (eureka_main.py)
1. **Generate Samples**: LLM creates multiple reward function variants
2. **Parallel Evaluation**: Launch multiple Unity instances for concurrent training
3. **Select Best**: Choose highest performing reward function
4. **Generate Feedback**: Analyze results and create improvement prompts
5. **Iterate**: Repeat with refined prompts

## File System Communication

### Directory Structure
```
Application.persistentDataPath/EurekaRewards/
├── rewards/
│   ├── reward_001_1703601234.cs    # Generated reward functions
│   ├── reward_002_1703601235.cs    # with run_id and timestamp
│   └── ...
├── results/
│   ├── results_001.json            # Training statistics
│   ├── results_002.json            # from Unity
│   └── ...
└── tensorboard/
    ├── run_1_reward_001_1703601234/ # TensorBoard logs for each training run
    ├── run_2_reward_002_1703601235/ # Organized by run_id and reward
    └── ...
```

### Platform Paths
- **Windows**: `%userprofile%/AppData/LocalLow/DefaultCompany/AR4/EurekaRewards/`
- **Linux**: `~/.config/unity3d/DefaultCompany/AR4/EurekaRewards/`
- **Mac**: `~/Library/Application Support/DefaultCompany/AR4/EurekaRewards/`

All subdirectories (rewards/, results/, tensorboard/) are organized under this main path.

## Training Configuration

### Task Hierarchy (Work Packages)
- **T0**: Move end effector to fixed target position (implemented)
- **T1**: Reach and grab fixed cube
- **T2**: Pick and place to fixed location
- **T3**: Pick and place with randomized positions
- **T4**: Vision-based pick and place
- **T5**: Multi-object stacking

### PPO Hyperparameters
```python
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "policy_kwargs": {
        "net_arch": [256, 256],
        "activation_fn": nn.Tanh
    }
}
```

## Usage

### Scene Setup
1. **Add RewardFunctionManagerInit** to scene
   - Edit reward function in inspector TextArea
   - Toggle "Use Inspector Reward" for development
   - Use context menu to recompile/validate during runtime

2. **Add ConfigManagerInit** to scene (existing)
   - Handles command line argument parsing

3. **Multiple RobotAgentDynamic** instances work automatically
   - All agents wait for reward manager initialization
   - All agents use the same compiled reward function

### Development Workflow
```bash
# 1. Edit reward in RewardFunctionManagerInit inspector
# 2. Hit play - reward compiles automatically  
# 3. All agents use the new reward function
# 4. Use "Recompile Reward Function" context menu for runtime testing
```

### Production Workflow
```bash
# Same scene setup works for Eureka runs
# When --reward-id provided, automatically uses file-based rewards
# When no args, falls back to inspector reward
# Zero code changes needed between development and production
```

### Testing Integration
```bash
# Test file system and basic integration
python test_eureka_integration.py

# Test full training system
python test_sb3_training.py

# Test single reward function
python eureka_unity/training_orchestrator.py
```

### Running Eureka
```bash
# Configure unity build path
export UNITY_BUILD_PATH="./Builds/AR4.exe"

# Run Eureka iterations
python eureka_unity/eureka_main.py
```

### Monitoring with TensorBoard
```bash
# Launch TensorBoard to monitor training progress
python launch_tensorboard.py

# Or manually with custom port
tensorboard --logdir ~/AppData/LocalLow/DefaultCompany/AR4/EurekaRewards/tensorboard --port 6006
```

### Building Unity
1. Ensure `RobotAgentDynamic` is in the scene
2. Install Roslyn packages via NuGet for Unity
3. Build with ML-Agents support
4. Test with: `./AR4.exe --reward-id test.cs --run-id 1 --mlagents-port 5005`

## Dependencies

### Unity Packages
- ML-Agents 2.3.0+
- Microsoft.CodeAnalysis.CSharp.Scripting (via NuGet)
- Unity 2022.3 LTS

### Python Requirements
- stable-baselines3
- mlagents
- numpy
- torch
- openai (for LLM integration)

## Current Status

### ✅ Implemented
- Dynamic C# reward compilation in Unity
- Clean separation of concerns architecture
- ConfigManager for centralized configuration
- Enhanced TrainingStatsLogger with Eureka support
- File-based reward function loading
- Unity subprocess launching from Python
- SB3 training integration
- Parallel reward evaluation
- Result collection and analysis
- Test suites for all components

### 🚧 TODO
- LLM integration for reward generation
- Feedback analysis system
- Tasks T1-T5 implementation
- Monitoring dashboard
- Cloud deployment support

## Troubleshooting

### Common Issues

1. **"Reward compilation failed"**
   - Check C# syntax in generated reward
   - Verify Roslyn packages are installed
   - Check Unity console for detailed errors

2. **"Unity subprocess timeout"**
   - Ensure Unity build exists at specified path
   - Check Unity isn't already running
   - Verify ML-Agents compatible Unity version

3. **"Cannot connect to Unity"**
   - Check port availability (5004+)
   - Ensure Unity launched successfully
   - Verify ML-Agents package version match

4. **"File not found" errors**
   - Check Application.persistentDataPath permissions
   - Verify directory structure exists
   - Ensure cross-platform path handling