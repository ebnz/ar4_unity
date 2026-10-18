# AR4 Unity Robotics Simulation

LLM-powered reinforcement learning training for AR4 robot arm with dynamic reward generation.

# Getting Started

## Unity Setup

1. **Clone the Repository**
   ```sh
   git clone git@github.com:zebleck/AR4.git
   ```

2. **Open the Project**
   - Open **Unity Hub** and click **Add** → **Add project from disk**.
   - Select the cloned `AR4` folder and open it.
   - Unity will import all packages automatically.

3. **Enable Run in Background (Recommended)**
   - Go to **Edit → Project Settings → Player → Resolution and Presentation**.
   - Check **"Run in Background"**.

4. **Open the Example Scene**
   - In the Unity Project window, navigate to `Assets/Scenes` and open a scene (e.g., `T0.unity`).

---

## Python Setup

### Windows

1. **Install Miniconda**
   - Download Miniconda:
     ```powershell
     Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "Miniconda3-latest-Windows-x86_64.exe"
     ```
   - Run the installer:
     ```powershell
     Start-Process .\Miniconda3-latest-Windows-x86_64.exe
     ```
   - (Optional) Remove the installer after installation:
     ```powershell
     Remove-Item .\Miniconda3-latest-Windows-x86_64.exe
     ```

2. **Initialize Conda for PowerShell**
   - Run:
     ```powershell
     conda init powershell
     ```
   - Restart PowerShell and navigate to the Python folder:
     ```powershell
     cd Assets/Python
     ```

3. **Set Up the Environment**
   - Run the setup script:
     ```powershell
     .\setup_env.ps1
     ```
   - This will create a Conda environment named `sbx` and install all required packages.

---

### Linux

1. **Install Miniconda**
   - Download Miniconda:
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     ```
   - Run the installer:
     ```bash
     bash Miniconda3-latest-Linux-x86_64.sh
     ```
     When it asks `Do you wish the installer to initialize Miniconda3 by running conda init? [yes|no]`, select `yes`.
   - (Optional) Remove the installer after installation:
     ```bash
     rm Miniconda3-latest-Linux-x86_64.sh
     ```

2. **Initialize Conda for Bash**
   - Restart your terminal and navigate to the Python folder:
     ```bash
     cd Assets/Python
     ```

3. **Set Up the Environment**
   - Run the setup script:
     ```bash
     bash setup_env.sh
     ```
   - This will create a Conda environment named `sbx` and install all required packages.

---

## Testing

1. **Activate the Conda Environment**
   ```powershell
   conda activate sbx
   ```

2. **Start Training**
   - In the `Assets/Python` folder, run:
     ```powershell
     python train_sb3.py
     ```
   - The script will wait for a Unity process to connect.

3. **Start Unity**
   - In Unity, open the `AR4Scene_T0.unity` scene and press **Play**.
   - Training should now begin.

---

## Key Features

- **Dynamic Rewards**: Runtime C# compilation of LLM-generated reward functions
- **Eureka Integration**: Automated reward evolution via LLMs
- **Parallel Training**: Concurrent evaluation of multiple reward functions
- **5x Training Speedup**: Optimized Unity-SB3 communication for parallel training
- **Multi-Task Support**: Progressive complexity from T0→T5
- **Real-time Monitoring**: TensorBoard integration

## Architecture

```
Python Eureka ──rewards──> Unity C# ──ML-Agents──> SB3 Training
```

## Current Tasks

- **T0** ✅: Move end effector to target
- **T1** 🔄: Reach and grab cube
- **T2** 🚧: Pick and place
- **T3-T5** 📋: Advanced manipulation

## Key Files

- `Assets/Scripts/ml/RobotAgentT0.cs` - Main RL agent
- `Assets/Scripts/ml/RewardFunctionManager.cs` - Dynamic reward system
- `Assets/Python/eureka_main.py` - LLM reward generation
- `Assets/Python/train_sb3.py` - SB3 training loop

## Documentation

- **[Assets/EXPLANATION.md](Assets/EXPLANATION.md)** - Complete architecture guide
- **[Assets/Python/EXPLANATION.md](Assets/Python/EXPLANATION.md)** - Python system details
- **[EurekaAnalysis](https://github.com/zebleck/EurekaAnalysis)** - Training analysis tools

## Dependencies

- Unity 2022.3 LTS + ML-Agents
- Python: stable-baselines3, mlagents, openai
- Roslyn (NuGet for Unity)

---

*For detailed setup and architecture information, see [Assets/EXPLANATION.md](Assets/EXPLANATION.md)*
