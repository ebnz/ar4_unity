# Eureka Unity Integration

Dynamic reward function generation for Unity AR4 robot using LLM-based reward engineering.

## Overview

This system implements the Eureka methodology for automatically generating and iterating on reward functions for reinforcement learning tasks. The Python side generates C# reward functions using LLMs, while Unity dynamically compiles and evaluates them during training.

## Architecture

```
Python (LLM Generation) ←→ File System ←→ Unity (Training)
├── eureka_main.py          ├── rewards/     ├── RobotAgentDynamic.cs
├── RewardGenerator         ├── results/     ├── Roslyn scripting  
└── TrainingOrchestrator    └── config/      └── Command line args
```

## Quick Start

### 1. Test the Integration

```bash
cd Python
python test_eureka_integration.py
```

This will:
- Test file system setup
- Create sample reward functions
- Verify Unity command generation
- Test result parsing

### 2. Build Unity Project

1. Build your Unity project with `RobotAgentDynamic` 
2. Place the build at `./Builds/AR4.exe` (or update config)
3. Ensure the build includes the Roslyn scripting packages

### 3. Run Eureka Training

```bash
python eureka_unity/eureka_main.py
```

## File Structure

### Generated Files

**Rewards Directory** (`Application.persistentDataPath/EurekaRewards/rewards/`):
```
reward_001_1703601234.cs    # Generated reward functions
reward_002_1703601235.cs    # with timestamp IDs
...
```

**Results Directory** (`Application.persistentDataPath/EurekaRewards/results/`):
```
results_001_1703601234.json # Training statistics
results_002_1703601235.json # from Unity
...
```

### Unity Integration

**RobotAgentDynamic.cs** supports:
- `--reward-id filename.cs` - Load specific reward function
- `--run-id 123` - Set run identifier  
- `--max-training-steps 100000` - Set training duration
- `--output-file results.json` - Output file for results

### Reward Function Format

Generated C# reward functions must follow this structure:

```csharp
// Extract observations
float currentDistanceToGoal = observations["DistanceToGoal"];
float distanceDelta = context.previousDistance - currentDistanceToGoal;

// Initialize reward components
var rewards = new Dictionary<string, float>();

// Add reward components
rewards["MovingCloser"] = distanceDelta * 50f;
rewards["TimePenalty"] = -0.001f;

if (currentDistanceToGoal < context.goalThreshold)
{
    rewards["GoalReached"] = 10.0f;
}

// Calculate total
float total = 0f;
foreach (var r in rewards.Values)
    total += r;
    
return new RobotAgentDynamic.RewardResult { Total = total, Components = rewards };
```

## Configuration

### Python Config (`eureka_main.py`):

```python
config = {
    'task': 'T0',                              # Task identifier
    'model': 'gpt-4',                          # LLM model
    'iterations': 5,                           # Eureka iterations
    'samples': 16,                             # Samples per iteration
    'max_training_steps': 100000,                       # Training steps
    'unity_build_path': './Builds/AR4.exe'  # Unity build path
}
```

### Task Definitions:

- **T0**: Move end effector to fixed target
- **T1**: Reach and grab fixed cube  
- **T2**: Pick and place to fixed location
- **T3**: Pick and place with randomized positions
- **T4**: Pick and place with camera input
- **T5**: Multi-object stacking

## Training Workflow

1. **Generate**: LLM creates reward function code
2. **Write**: Python writes `.cs` file to rewards directory
3. **Launch**: Python launches Unity with command line arguments
4. **Train**: Unity compiles reward function and trains
5. **Results**: Unity writes training statistics to results directory
6. **Analyze**: Python reads results and generates feedback
7. **Iterate**: Process repeats with improved reward functions

## Example Unity Launch Command

```bash
./AR4.exe -batchmode -nographics \
  --reward-id reward_001_1703601234.cs \
  --run-id 1 \
  --max-training-steps 100000 \
  --output-file results_001.json
```

## Result Format

Unity outputs JSON results:

```json
{
  "run_id": 1,
  "reward_id": "reward_001_1703601234.cs",
  "success_rate": 0.75,
  "total_episodes": 200,
  "final_episode_length": 1200,
  "reward_components": {
    "MovingCloser": {"mean": 2.5, "total": 500.0},
    "GoalReached": {"mean": 7.5, "total": 150.0}
  },
  "training_time": 180.5
}
```

## Troubleshooting

### Common Issues:

1. **Reward compilation fails**: Check C# syntax in generated code
2. **File not found**: Verify Unity persistentDataPath and file permissions
3. **Unity crashes**: Check command line arguments and build configuration
4. **No results**: Verify output directory exists and is writable

### Debug Mode:

Set environment variable for detailed logging:
```bash
export EUREKA_DEBUG=1
python eureka_unity/eureka_main.py
```

## Next Steps

1. Implement LLM integration (OpenAI API)
2. Add parallel evaluation for multiple reward samples
3. Implement advanced feedback analysis
4. Add support for different task types (T1-T5)
5. Create monitoring dashboard for training progress

## Dependencies

- **Unity**: 2022.3+ with Roslyn scripting support
- **Python**: 3.8+ with numpy, openai
- **Packages**: Microsoft.CodeAnalysis.CSharp.Scripting in Unity