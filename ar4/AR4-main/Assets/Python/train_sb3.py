"""
Simple training script for Unity AR4 robot using Stable Baselines3

This script provides the same structured logging and results as Eureka runs,
but for single training sessions without LLM-based reward generation.
"""

import os
import sys
import json
import argparse
import shutil
from datetime import datetime
from typing import Dict, Optional
import time

from stable_baselines3 import PPO

# Shared training utilities
from training.common import (
    load_training_config,
    setup_hyperparameters,
    create_unity_environment,
    create_timestamped_run_directory,
    CheckpointWithONNXCallback,
    save_final_model_with_onnx,
    write_unity_results_to_tensorboard,
    copy_agent_file_if_specified,
    find_free_port,
    train_with_unity_subprocess,
    load_results_from_json,
)


# Alias for backward compatibility
load_config = load_training_config


def copy_reward_file_if_specified(config: Dict) -> Optional[str]:
    """Copy reward file to rewards directory if specified, return filename or None"""
    # Get reward_file from config["paths"] if present, fallback to top-level for legacy support.
    reward_file = None
    if "paths" in config and isinstance(config["paths"], dict):
        reward_file = config["paths"].get("reward_file")
    if not reward_file:
        reward_file = config.get("reward_file")

    if reward_file and os.path.exists(reward_file):
        # Copy to rewards directory
        reward_filename = f"reward_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cs"
        reward_path = os.path.join(config["rewards_dir"], reward_filename)

        shutil.copy2(reward_file, reward_path)
        print(f"Using custom reward function: {reward_file}")
        print(f"Copied to: {reward_path}")

        return reward_filename
    elif reward_file:
        print(f"WARNING: Reward file specified but not found: {reward_file}")
        print("Using Unity default reward function")
        return None
    else:
        print("Using Unity default reward function (from editor)")

        # Create a marker file to indicate we're using Unity default
        marker_path = os.path.join(config["rewards_dir"], "using_unity_default.txt")
        with open(marker_path, "w") as f:
            f.write(
                "This training run uses the reward function defined in the Unity agent script."
            )

        return None


def write_unity_editor_config(config: Dict, result_filename: str) -> str:
    """
    Write config file for Unity Editor to read.
    Unity will use this to know where to save results and export rewards.
    """
    editor_config = {
        "run_id": 0,
        "rewards_dir": config["rewards_dir"],
        "results_dir": config["results_dir"],
        "result_filename": result_filename,
        "max_episode_steps": config["max_episode_steps"],
        "use_aggregation": config.get("logging", {}).get("use_aggregation", True),
        "stats_aggregation_interval": config.get("logging", {}).get(
            "stats_aggregation_interval", 5000
        ),
    }

    # Write to StreamingAssets (Unity can read this at runtime)
    # Use relative path from Assets directory
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "StreamingAssets")
    os.makedirs(assets_dir, exist_ok=True)

    config_path = os.path.join(assets_dir, "python_training_config.json")
    with open(config_path, "w") as f:
        json.dump(editor_config, f, indent=2)

    print(f"Unity Editor config written to: {config_path}")
    return config_path


def _train_with_build(
    config: Dict,
    reward_filename: Optional[str],
    hyperparameters: Dict,
    result_filename: str,
    port: int,
) -> Dict:
    """Train by launching Unity build as subprocess"""

    def training_callback(env):
        """Callback function that performs the actual training once environment is connected"""
        # Get action space configuration from environment
        from gymnasium import spaces

        continuous_action_size = 0
        discrete_action_branches = []

        if isinstance(env.action_space, spaces.Box):
            continuous_action_size = env.action_space.shape[0]
            discrete_action_branches = []
        elif isinstance(env.action_space, spaces.MultiDiscrete):
            continuous_action_size = 0
            discrete_action_branches = env.action_space.nvec.tolist()
        elif isinstance(env.action_space, spaces.Discrete):
            continuous_action_size = 0
            discrete_action_branches = [int(env.action_space.n)]
        elif isinstance(env.action_space, spaces.Tuple):
            continuous_action_size = env.action_space[1].shape[0]
            discrete_action_branches = []
        else:
            print(f"Warning: Unknown action space type: {type(env.action_space)}")
            continuous_action_size = 3
            discrete_action_branches = []

        print(f"Detected action space:")
        print(f"  Continuous actions: {continuous_action_size}")
        print(f"  Discrete branches: {discrete_action_branches}")

        # Set up TensorBoard logging
        tensorboard_log_dir = (
            config["tensorboard_dir"]
            if config.get("logging", {}).get("tensorboard", True)
            else None
        )

        # Create PPO model
        model = PPO(env=env, tensorboard_log=tensorboard_log_dir, **hyperparameters)

        # Set up callbacks
        callbacks = []

        # Add checkpoint callback if save_frequency is specified
        save_freq = config.get("logging", {}).get("save_frequency")
        if save_freq:
            checkpoint_callback = CheckpointWithONNXCallback(
                save_freq=save_freq,
                save_path=config["checkpoints_dir"],
                env=env,
                continuous_action_size=continuous_action_size,
                discrete_action_branches=discrete_action_branches,
                verbose=1,
            )
            callbacks.append(checkpoint_callback)
            print(
                f"Will save checkpoints (with ML-Agents ONNX) every {save_freq} steps to {config['checkpoints_dir']}"
            )

        # Train the model
        print(f"\nTraining PPO model for {config['max_training_steps']} steps...")
        start_time = time.time()

        # Set up learn kwargs
        learn_kwargs = {
            "total_timesteps": config["max_training_steps"],
            "progress_bar": True,
            "reset_num_timesteps": True,
            "callback": callbacks if callbacks else None,
        }

        if tensorboard_log_dir:
            learn_kwargs["tb_log_name"] = (
                f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        model.learn(**learn_kwargs)

        training_time = time.time() - start_time

        # Save final model with ONNX export
        if save_freq:
            save_final_model_with_onnx(
                model,
                env,
                config["checkpoints_dir"],
                "final_model",
                continuous_action_size=continuous_action_size,
                discrete_action_branches=discrete_action_branches,
            )

        print(f"\n✅ Training completed in {training_time:.1f}s")

        # Wait a moment for Unity to finish writing results
        time.sleep(1)

        # Load results from Unity
        result_path_full = os.path.join(config["results_dir"], result_filename)
        results = load_results_from_json(result_path_full)

        if results:
            print(f"✅ Results loaded from Unity")
        else:
            print(f"⚠️  Results not found (Unity may not have written them yet)")

        # Write Unity results to TensorBoard
        if tensorboard_log_dir and os.path.exists(result_path_full):
            print(f"\nWriting Unity results to TensorBoard...")
            write_unity_results_to_tensorboard(
                result_path=result_path_full,
                tensorboard_log_dir=tensorboard_log_dir,
                run_name="unity_stats",
            )

        return {
            "training_time": training_time,
            "result_filename": result_filename,
            "results": results,
        }

    # Use common subprocess training function
    return train_with_unity_subprocess(
        config=config,
        port=port,
        result_filename=result_filename,
        reward_filename=reward_filename,
        run_id=0,
        max_training_steps=config["max_training_steps"],
        gamma=hyperparameters["gamma"],
        seed=config.get("seed", 0),
        training_callback=training_callback,
    )


def train_simple(
    config: Dict,
    reward_filename: Optional[str],
    hyperparameters: Dict,
) -> Dict:
    """Train by connecting to Unity Editor or launching Unity build"""
    # Generate result filename
    result_filename = (
        f"results_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Check if we should launch a Unity build or connect to Editor
    unity_build_path = config.get("unity_build_path")

    if unity_build_path and os.path.exists(unity_build_path):
        # Launch Unity build as subprocess
        print(f"\n🚀 Launching Unity build: {unity_build_path}")

        # Find a free port
        preferred_port = config.get("base_port", 5004)
        actual_port = find_free_port(preferred_port)
        print(f"Using port: {actual_port}")

        return _train_with_build(
            config, reward_filename, hyperparameters, result_filename, actual_port
        )

    # Original Editor mode
    print("\n🔌 Connecting to Unity Editor on port 5004...")
    env = None

    # Write config for Unity Editor to read
    write_unity_editor_config(config, result_filename)

    try:
        print("\nConnecting to Unity Editor...")
        print("Make sure Unity Editor is running in Play mode!")

        # Create Unity environment connection (connects to Editor on default port 5004)
        env = create_unity_environment(
            port=5004,
            seed=config.get("seed", 0),
            gamma=hyperparameters["gamma"],
            time_scale=5.0,
            timeout_wait=60,
        )

        print("✅ Connected to Unity Editor!")

        # Get action space configuration from environment
        # The environment is wrapped with HybridActionWrapper, so we need to check the type
        from gymnasium import spaces

        continuous_action_size = 0
        discrete_action_branches = []

        if isinstance(env.action_space, spaces.Box):
            # Continuous actions only (or hybrid converted to Box)
            continuous_action_size = env.action_space.shape[0]
            discrete_action_branches = []
        elif isinstance(env.action_space, spaces.MultiDiscrete):
            # Discrete actions only
            continuous_action_size = 0
            discrete_action_branches = env.action_space.nvec.tolist()
        elif isinstance(env.action_space, spaces.Discrete):
            # Single discrete action
            continuous_action_size = 0
            discrete_action_branches = [int(env.action_space.n)]
        elif isinstance(env.action_space, spaces.Tuple):
            # HybridActionWrapper might create tuple (discrete, continuous)
            continuous_action_size = env.action_space[1].shape[0]
            discrete_action_branches = []
        else:
            # Fallback: assume continuous
            print(f"Warning: Unknown action space type: {type(env.action_space)}")
            continuous_action_size = 3
            discrete_action_branches = []

        print(f"Detected action space:")
        print(f"  Continuous actions: {continuous_action_size}")
        print(f"  Discrete branches: {discrete_action_branches}")

        # Set up TensorBoard logging
        tensorboard_log_dir = (
            config["tensorboard_dir"]
            if config.get("logging", {}).get("tensorboard", True)
            else None
        )

        # Create PPO model
        model = PPO(env=env, tensorboard_log=tensorboard_log_dir, **hyperparameters)

        # Set up callbacks
        callbacks = []

        # Add checkpoint callback if save_frequency is specified
        save_freq = config.get("logging", {}).get("save_frequency")
        if save_freq:
            checkpoint_callback = CheckpointWithONNXCallback(
                save_freq=save_freq,
                save_path=config["checkpoints_dir"],
                env=env,
                continuous_action_size=continuous_action_size,
                discrete_action_branches=discrete_action_branches,
                verbose=1,
            )
            callbacks.append(checkpoint_callback)
            print(
                f"Will save checkpoints (with ML-Agents ONNX) every {save_freq} steps to {config['checkpoints_dir']}"
            )

        # Train the model
        print(f"\nTraining PPO model for {config['max_training_steps']} steps...")
        start_time = time.time()

        # Set up learn kwargs
        learn_kwargs = {
            "total_timesteps": config["max_training_steps"],
            "progress_bar": True,
            "reset_num_timesteps": True,
            "callback": callbacks if callbacks else None,
        }

        if tensorboard_log_dir:
            learn_kwargs["tb_log_name"] = (
                f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        model.learn(**learn_kwargs)

        training_time = time.time() - start_time

        # Save final model with ONNX export
        if save_freq:
            save_final_model_with_onnx(
                model,
                env,
                config["checkpoints_dir"],
                "final_model",
                continuous_action_size=continuous_action_size,
                discrete_action_branches=discrete_action_branches,
            )

        print(f"\n✅ Training completed in {training_time:.1f}s")

        # Wait a moment for Unity to finish writing results
        time.sleep(1)

        # Load results from Unity (TrainingStatsLogger should have written them)
        result_path_full = os.path.join(config["results_dir"], result_filename)

        from training.common import load_results_from_json

        results = load_results_from_json(result_path_full)

        if results:
            print(f"✅ Results loaded from Unity")
        else:
            print(f"⚠️  Results not found (Unity may not have written them yet)")

        # Write Unity results to TensorBoard
        if tensorboard_log_dir and os.path.exists(result_path_full):
            print(f"\nWriting Unity results to TensorBoard...")
            write_unity_results_to_tensorboard(
                result_path=result_path_full,
                tensorboard_log_dir=tensorboard_log_dir,
                run_name="unity_stats",
            )

        return {
            "training_time": training_time,
            "result_filename": result_filename,
            "results": results,
        }

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # Clean up environment
        if env is not None:
            try:
                print("Closing environment...")
                env.close()
            except Exception as env_cleanup_error:
                print(f"Warning: Error closing environment: {env_cleanup_error}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Unity AR4 robot with Stable Baselines3"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Display configuration
    print("=" * 60)
    print("Unity AR4 Robot Training (Stable Baselines3)")
    print("=" * 60)
    print(f"Run directory: {config['run_dir']}")
    if "task_name" in config:
        print(f"Task: {config.get('task')} - {config['task_name']}")
        print(f"Description: {config.get('task_description', 'N/A')}")
    print(f"Max training steps: {config['max_training_steps']}")
    print(f"Max episode steps: {config['max_episode_steps']}")

    # Display connection mode
    unity_build_path = config.get("unity_build_path")
    if unity_build_path and os.path.exists(unity_build_path):
        print(f"Mode: Unity Build (subprocess)")
        print(f"Build path: {unity_build_path}")
    else:
        print(f"Mode: Unity Editor (port 5004)")

    print(f"Seed: {config.get('seed', 0)}")
    print("-" * 60)

    # Save the complete config to the run directory for reproducibility
    config_save_path = os.path.join(config["run_dir"], "config.json")
    with open(config_save_path, "w") as f:
        # Convert config to JSON-serializable format
        config_copy = config.copy()
        # Remove non-serializable objects if any
        json.dump(config_copy, f, indent=2, default=str)
    print(f"Configuration saved to: {config_save_path}")

    # Copy agent file for posterity if specified in task config
    copy_agent_file_if_specified(config)

    # Copy reward file if specified
    reward_filename = copy_reward_file_if_specified(config)

    # Set up hyperparameters
    training_config = config.get("training", {})
    hyperparameters = setup_hyperparameters(training_config)

    print("\nHyperparameters:")
    for key, value in hyperparameters.items():
        if key != "policy_kwargs":
            print(f"  {key}: {value}")
    print(f"  policy_kwargs: {hyperparameters['policy_kwargs']}")

    # Start training
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        training_results = train_simple(config, reward_filename, hyperparameters)

        print("\n" + "=" * 60)
        print(
            f"Training completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 60)

        # Display results summary
        print(f"\n⏱️  Training time: {training_results['training_time']:.1f}s")

        # Display results if available
        if training_results.get("results"):
            results = training_results["results"]
            print(f"\n📊 Training Results:")
            print(f"  Success rate: {results.get('success_rate', 0.0):.2%}")
            print(f"  Total episodes: {results.get('total_episodes', 0)}")

            # Episode statistics
            if "episode_reward_stats" in results:
                stats = results["episode_reward_stats"]
                print(f"\n  Episode Rewards:")
                print(f"    Mean: {stats.get('mean', 0.0):.3f}")
                print(f"    Std: {stats.get('std', 0.0):.3f}")
                print(f"    Min: {stats.get('min', 0.0):.3f}")
                print(f"    Max: {stats.get('max', 0.0):.3f}")

            if "episode_length_stats" in results:
                stats = results["episode_length_stats"]
                print(f"\n  Episode Lengths:")
                print(f"    Mean: {stats.get('mean', 0.0):.1f}")
                print(f"    Std: {stats.get('std', 0.0):.1f}")
                print(f"    Min: {stats.get('min', 0.0):.0f}")
                print(f"    Max: {stats.get('max', 0.0):.0f}")

        # Print output locations
        print(f"\n📁 All outputs saved to: {config['run_dir']}")
        print(f"  Results: {config['results_dir']}")
        print(f"  Reward files: {config['rewards_dir']}")
        print(f"  Checkpoints: {config['checkpoints_dir']}")
        if config.get("logging", {}).get("tensorboard", True):
            print(f"  TensorBoard logs: {config['tensorboard_dir']}")

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 2

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
