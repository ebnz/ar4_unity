# AR4 Unity Assets Architecture

## Overview

This Unity project implements a sophisticated ML-powered robotics simulation for training an AR4 robot arm using reinforcement learning with dynamically generated reward functions. The system integrates Unity's ML-Agents framework with external Python training orchestration via Stable Baselines3, enabling Eureka-style LLM-based reward evolution.

## Core Architecture

### Separation of Concerns

The Unity architecture follows clean separation of concerns across four main subsystems:

1. **Robot Control** - Physical simulation and movement
2. **ML-Agents Integration** - Reinforcement learning behavior  
3. **Configuration Management** - Command-line parsing and settings
4. **Dynamic Rewards** - Runtime C# compilation and execution

### Communication Flow

```
Python Eureka System ──file──> C# Reward Functions ──runtime──> Unity Agents
                    ──cmd──> Unity Build ──MLAgents──> SB3 Training
```

## Information Encoding for Eureka

### Agent Annotation System

The system uses a sophisticated annotation system to extract environment information for the Eureka LLM process via `Python/extract_env_info.py`. This parser reads Unity C# agent files and extracts structured information using special comment annotations:

#### Task Identification
```csharp
/// @task-name: T0
```
Marks the task name for identification in the Eureka system.

#### Task Description
```csharp
/// <taskdescription>
/// T0 Task: Move the robot end effector to reach a fixed target position.
/// The agent controls the target position that the inverse kinematics system follows.
/// Success is achieved when the end effector reaches within 0.1 units of the goal position.
/// </taskdescription>
```
Provides comprehensive task description for LLM context.

#### Success Criteria
```csharp
/// <successcriteria>
/// Distance between end effector and goal position < 0.1 units
/// </successcriteria>
```
Defines clear success metrics for reward function generation.

#### Observation Annotations
```csharp
currentObservations["DistanceToGoal"] = distanceToGoal; // @obs: Euclidean distance from end effector to goal position
currentObservations["TargetToGoal_X"] = targetToGoal.x; // @obs: X component of vector from current target to goal position
```
Each observation is documented with `// @obs:` comments explaining its meaning and purpose.

#### Context Variable Annotations
```csharp
var context = new Dictionary<string, object>
{
    ["previousDistance"] = previousDistanceToGoal, // @context: Distance to goal in previous timestep
    ["goalThreshold"] = ConfigManager.Instance.GoalThreshold, // @context: Success threshold distance (0.1f)
    ["episodeSteps"] = currentEpisodeSteps, // @context: Current number of steps in this episode
};
```
Context variables provide additional state information with `// @context:` annotations.

### Information Extraction Process

1. **Parse Agent File**: `UnityAgentParser` reads the C# agent implementation
2. **Extract Annotations**: Regex patterns identify and extract annotated information
3. **Structure Data**: Information is organized into `EnvironmentInfo` dataclass
4. **Generate Prompt**: Structured data is formatted for LLM consumption
5. **Reward Generation**: LLM uses environment information to create reward functions

This annotation system ensures that LLMs have complete understanding of the task, observations, actions, and context when generating reward functions.

## Key Components

### Robot Control System

#### RobotManager.cs (`Scripts/RobotManager.cs`)
- **Purpose**: Central robot joint control and physics configuration
- **Key Features**:
  - AR4-specific joint configuration (6 articulation bodies)
  - Physics parameters: stiffness (15k-2k), damping (50-150), force limits (300-1500N)
  - Joint limits: realistic ranges for AR4 arm (-170° to +180° varies by joint)
  - Instant reset capability for episode boundaries
  - End effector position/rotation tracking
- **Critical Methods**:
  - `SetUpRobot()`: Configures physics parameters for all 6 joints
  - `ResetRobotInstantly()`: Zero-latency reset for RL training
  - `MoveTo(Vector3)`: Target-based movement via IK system

#### MoveToTarget.cs (`Scripts/MoveToTarget.cs`)
- **Purpose**: Inverse kinematics implementation for robot movement
- **Integration**: Works with `RobotManager` to translate target positions to joint commands
- **Performance**: Optimized for real-time RL training requirements

#### ArticulationBodyDriver.cs (`Scripts/ArticulationBodyDriver.cs`) 
- **Purpose**: Low-level joint physics control
- **Unity Integration**: Interfaces with Unity's ArticulationBody physics system
- **Real-time**: Handles high-frequency joint updates during training

### ML-Agents Integration

#### RobotAgentT0.cs (`Scripts/ml/RobotAgentT0.cs`)
- **Task Definition**: T0 - Move end effector to fixed target position
- **Task Annotation**: `/// @task-name: T0` for identification
- **Success Criteria**: Distance < 0.1 units from goal position
- **Observations** (4-dimensional):
  - `TargetToGoal_X/Y/Z`: Vector components from current to goal position  
  - `DistanceToGoal`: Euclidean distance to target
- **Actions** (3-dimensional): Continuous XYZ target position deltas
- **Episode Management**: 
  - Max 2000 steps per episode
  - Automatic termination on goal achievement
  - Instant reset via `RobotManager.ResetRobotInstantly()`

#### IEurekaAgent Interface (`Scripts/ml/IEurekaAgent.cs`)
- **Purpose**: Contract for agents supporting dynamic reward functions
- **Methods**: Standardized interface for reward execution and observation access
- **Extensibility**: Enables future task implementations (T1-T5)

### Configuration Management

#### ConfigManager.cs (`Scripts/ml/ConfigManager.cs`)
- **Pattern**: Singleton for global configuration access
- **Command Line Parsing**: 
  - `--reward-id`: Specifies which reward function file to load
  - `--run-id`: Unique identifier for training run
  - `--max-steps`: Training duration override
  - `--output-file`: Results file path
  - `--mlagents-port`: Communication port for ML-Agents
- **Runtime Configuration**: Dynamic parameter updates during training
- **Validation**: Ensures required parameters are present

#### ConfigManagerInit.cs (`Scripts/ml/ConfigManagerInit.cs`)
- **Purpose**: MonoBehaviour wrapper for scene-level initialization
- **Timing**: Initializes in `Awake()` before agents start
- **Integration**: Parses command line arguments passed from Python orchestrator

### Dynamic Reward System

#### RewardFunctionManager.cs (`Scripts/ml/RewardFunctionManager.cs`)
- **Core Innovation**: Runtime C# compilation using Microsoft.CodeAnalysis.CSharp.Scripting
- **Architecture**: Static class for consistent access pattern
- **Priority System**: File-based → Inspector → Built-in default
- **Compilation**: Async compilation with error handling and structured feedback
- **Thread Safety**: Lock-based synchronization for multi-agent scenarios

**Reward Function Format**:
```csharp
// Available observations via dictionary
float currentDistance = observations["DistanceToGoal"];
float deltaX = observations["TargetToGoal_X"];

// Context provides additional state
float previousDistance = (float)context["previousDistance"];
float threshold = (float)context["goalThreshold"];

// Build reward components for analysis
var rewards = new Dictionary<string, float>();
rewards["MovingCloser"] = (previousDistance - currentDistance) * 50f;
rewards["GoalReached"] = currentDistance < threshold ? 100f : 0f;
rewards["StepPenalty"] = -0.1f;

// Auto-calculate total (or set totalReward explicitly)
float total = 0f;
foreach (var r in rewards.Values) total += r;

return new RewardResult { Total = total, Components = rewards };
```

#### RewardFunctionManagerInit.cs (`Scripts/ml/RewardFunctionManagerInit.cs`)
- **Development Tool**: Inspector-based reward editing with large TextArea
- **Runtime Testing**: Context menu options for recompilation during development
- **Seamless Integration**: Works with both development and production workflows
- **Zero-Code Switching**: Same scene setup works for both Eureka runs and manual testing

### Statistics and Monitoring

#### TrainingStatsLogger.cs (`Scripts/ml/TrainingStatsLogger.cs`)
- **Dual Mode**: Supports both legacy `RobotAgent` and new `RobotAgentDynamic`
- **Episode Tracking**: Rewards, lengths, success rates across episodes
- **Eureka Integration**: Writes JSON results compatible with Python analysis
- **Real-time Monitoring**: Current statistics for dashboards and TensorBoard
- **Performance Metrics**: Episode statistics, reward component analysis

### Testing and Validation

#### RoslynScriptTest.cs (`Scripts/ml/RoslynScriptTest.cs`)
- **Purpose**: Validation utilities for Roslyn compilation system
- **Development**: Testing framework for reward function compilation
- **Debugging**: Helps identify compilation issues during development

#### UnityExpressionEvaluator.cs (`Scripts/ml/UnityExpressionEvaluator.cs`)
- **Purpose**: Expression evaluation helpers for reward functions
- **Runtime**: Supports dynamic expression evaluation within reward functions

## Configuration Files

### ML-Agents Training Configuration

#### robot.yaml (`Configs/robot.yaml`)
- **Behavior**: `RobotPickAndPlace` (generic name, used for all tasks)
- **Algorithm**: PPO with optimized hyperparameters
- **Network**: 256-unit hidden layers, 2 layers deep, normalization enabled
- **Training**: 250k max steps, 64-step time horizon
- **Optimizations**: 
  - Batch size 2048 for stable gradients
  - Learning rate 3e-4 with linear schedule
  - GAE lambda 0.95 for variance reduction

**Note**: ML-Agents configuration files in `Configs/` are optional and not the primary training method. The main training approach uses Stable Baselines3 via `Python/train_sb3.py` with direct Unity communication.

### Work Package Definitions

The project implements a progressive task hierarchy:

- **T0** (Current): Move end effector to fixed target - `RobotAgentT0.cs`
- **T1** (Planned): Reach and grab fixed cube
- **T2** (In Progress): Pick and place to fixed location  
- **T3** (Planned): Pick and place with randomized positions
- **T4** (Planned): Vision-based pick and place
- **T5** (Planned): Multi-object stacking

Each task builds complexity while maintaining the same underlying architecture.

## File System Communication

### Directory Structure
```
Application.persistentDataPath/EurekaRewards/
├── rewards/
│   ├── reward_001_1703601234.cs    # LLM-generated reward functions
│   ├── reward_002_1703601235.cs    # with run_id and timestamp
│   └── ...
├── results/
│   ├── results_001.json            # Training statistics from Unity
│   ├── results_002.json            # Compatible with Python analysis
│   └── ...
└── tensorboard/
    ├── run_1_reward_001_1703601234/ # TensorBoard logs per training run
    ├── run_2_reward_002_1703601235/ # Organized by run_id and reward
    └── ...
```

**Analysis Tools**: For advanced analysis of training results and reward evolution, see the [EurekaAnalysis repository](https://github.com/zebleck/EurekaAnalysis) which provides visualization and statistical analysis tools for Eureka training runs.

### Platform-Specific Paths
- **Windows**: `%userprofile%/AppData/LocalLow/DefaultCompany/AR4/EurekaRewards/`
- **Linux**: `~/.config/unity3d/DefaultCompany/AR4/EurekaRewards/`
- **Mac**: `~/Library/Application Support/DefaultCompany/AR4/EurekaRewards/`

### Communication Protocol

1. **Python → Unity**: Write reward function to `rewards/reward_{id}_{timestamp}.cs`
2. **Python → Unity**: Launch Unity with `--reward-id reward_{id}_{timestamp}.cs`
3. **Unity**: Compile and execute reward function via `RewardFunctionManager`
4. **Unity → Python**: Write results to `results/results_{id}.json`
5. **Python**: Analyze results and generate next iteration

## Integration with Python Eureka System

### Command Line Interface
Unity build accepts parameters from Python training orchestrator:
```bash
./AR4.exe --reward-id reward_001.cs --run-id 1 --max-steps 50000 --output-file results_001.json --mlagents-port 5005
```

### Eureka Loop Integration
1. **LLM Generation**: Python generates multiple reward function variants
2. **Parallel Evaluation**: Multiple Unity instances evaluate rewards concurrently
3. **Performance Collection**: Unity writes structured results for analysis
4. **Selection**: Python selects best-performing reward functions
5. **Iteration**: Process repeats with refined prompts

### Real-time Monitoring
- **TensorBoard**: Live training metrics visualization
- **JSON Results**: Structured episode statistics and reward components
- **Console Logging**: Debug information and compilation status

## Development Workflow

### Scene Setup
1. Add `RewardFunctionManagerInit` to scene for inspector-based rewards
2. Add `ConfigManagerInit` for command line parsing  
3. Add `RobotAgentT0` instances (supports multiple agents per scene)
4. Configure robot references and goal positions

### Development Testing (Manual Training)
```bash
# For manual testing without Eureka
# IMPORTANT: Start Python training first
python Python/train_sb3.py

# Then start Unity scene in editor or build
# Edit reward in RewardFunctionManagerInit inspector
# Hit play - reward compiles automatically
# All agents use the new reward function
# Use "Recompile Reward Function" context menu for runtime testing
```

**Critical Setup Note**: For manual training, you must start `train_sb3.py` before launching the Unity scene or build. Otherwise, the robot will not respond to actions as there's no ML-Agents connection established.

**Note**: When using Eureka (`eureka_main.py`), this manual setup is not needed as Eureka handles Unity launching automatically.

### Production Deployment

**Automated Eureka Training**
```bash
# Eureka handles everything automatically
python Python/eureka_main.py
```

**Manual Single Reward Evaluation**
```bash
# For testing specific reward functions
python Python/train_sb3.py
./AR4.exe --reward-id reward_001.cs --run-id 1 --mlagents-port 5005
```

**Workflow Notes**:
- Same scene setup works for both Eureka and manual runs
- When `--reward-id` provided, automatically uses file-based rewards
- When no args, falls back to inspector reward
- Zero code changes needed between development and production

## Dependencies

### Unity Packages
- **ML-Agents 2.3.0+**: Core reinforcement learning framework
- **Microsoft.CodeAnalysis.CSharp.Scripting**: Runtime C# compilation (via NuGet)
- **Unity 2022.3 LTS**: Stable Unity version with ArticulationBody support

### External Integration
- **Stable Baselines3**: Python RL training algorithms
- **OpenAI API**: LLM integration for reward generation
- **TensorBoard**: Training visualization and monitoring

## Performance Characteristics

### Training Speed
- **5x Speedup**: Compared to previous implementations
- **Parallel Evaluation**: Multiple Unity instances for concurrent reward testing with thread-safe coordination
- **Instant Reset**: Zero-latency episode boundaries via `ResetRobotInstantly()`
- **Optimized Physics**: ArticulationBody configuration tuned for RL training

### Parallelization Architecture
- **Multi-Instance Training**: Concurrent Unity builds evaluate different reward functions simultaneously
- **Thread-Safe Coordination**: Robust process management and port allocation for parallel evaluation
- **Asynchronous Compilation**: Dynamic reward compilation doesn't block other training instances
- **Distributed Results**: Each instance writes independent results for centralized analysis

### Scalability
- **Multi-Agent**: Single scene supports multiple robot agents
- **Concurrent Training**: Multiple Unity builds can run simultaneously
- **Resource Management**: Efficient memory usage during long training runs

## Troubleshooting

### Common Issues

**"Reward compilation failed"**
- Check C# syntax in generated reward function
- Verify Roslyn packages installed via NuGet for Unity
- Check Unity console for detailed compilation errors
- Validate observation and context dictionary keys match agent implementation

**"Unity subprocess timeout"**  
- Ensure Unity build exists at specified path
- Check Unity isn't already running on same port
- Verify ML-Agents package version compatibility
- Confirm command line arguments passed correctly

**"Cannot connect to Unity"**
- Check port availability (5004+ range)
- Ensure Unity launched successfully with ML-Agents behavior
- Verify network configuration and firewall settings
- Match ML-Agents package versions between Unity and Python

**"File not found" errors**
- Check `Application.persistentDataPath` permissions
- Verify EurekaRewards directory structure exists
- Ensure cross-platform path handling in file operations
- Validate reward file naming conventions match expectations

## Future Enhancements

### Planned Features
- **Task T1-T5 Implementation**: Progressive complexity in manipulation tasks
- **Vision Integration**: Camera-based observations for T4 tasks
- **Multi-Robot Coordination**: Support for coordinated multi-agent training
- **Cloud Deployment**: Distributed training across cloud instances
- **Real Robot Transfer**: Sim-to-real deployment pipeline

### Architecture Improvements
- **Reward Function Caching**: Compilation caching for repeated evaluations
- **Advanced Monitoring**: Real-time performance dashboards
- **Automated Testing**: Continuous integration for reward function validation
- **Configuration Templates**: Pre-built configurations for common tasks

This architecture provides a robust foundation for LLM-driven robotics research while maintaining clean separation of concerns, efficient training performance, and seamless integration between Unity simulation and Python-based ML training pipelines.