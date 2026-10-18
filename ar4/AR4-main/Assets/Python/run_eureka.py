"""
Main entry point for running Eureka with Unity AR4 robot
"""

import os
import sys
import yaml
import json
import argparse
import re
import glob
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from eureka_main import EurekaUnity
from training.common import copy_agent_file_if_specified


def detect_run_state(run_dir: str) -> dict:
    """
    Detect the state of an existing Eureka run directory.

    Returns:
        dict with:
        - resume_iteration: The iteration to resume from
        - needs_training: Whether training needs to run for existing rewards
        - reward_files: List of reward files at the resume iteration (if needs_training)
    """
    rewards_dir = os.path.join(run_dir, "rewards")
    results_dir = os.path.join(run_dir, "results")

    if not os.path.exists(rewards_dir):
        raise ValueError(f"No rewards directory found at {rewards_dir}")

    # Pattern: reward_iter{XX}_sample{YYY}_attempt{Z}.cs
    reward_pattern = re.compile(r"reward_iter(\d+)_sample(\d+)_attempt(\d+)\.cs")
    result_pattern = re.compile(r"results_iter(\d+)_sample(\d+)_attempt(\d+)\.json")

    # Parse all reward files
    reward_files = glob.glob(os.path.join(rewards_dir, "reward_iter*_sample*_attempt*.cs"))
    result_files = glob.glob(os.path.join(results_dir, "results_iter*_sample*_attempt*.json"))

    if not reward_files:
        raise ValueError(f"No reward files found in {rewards_dir}")

    # Extract iterations from reward files
    reward_iterations = {}
    for f in reward_files:
        match = reward_pattern.match(os.path.basename(f))
        if match:
            iteration = int(match.group(1))
            sample = int(match.group(2))
            if iteration not in reward_iterations:
                reward_iterations[iteration] = set()
            reward_iterations[iteration].add(sample)

    # Extract iterations from result files
    result_iterations = {}
    for f in result_files:
        match = result_pattern.match(os.path.basename(f))
        if match:
            iteration = int(match.group(1))
            sample = int(match.group(2))
            if iteration not in result_iterations:
                result_iterations[iteration] = set()
            result_iterations[iteration].add(sample)

    max_reward_iter = max(reward_iterations.keys())
    rewards_at_max = reward_iterations[max_reward_iter]
    results_at_max = result_iterations.get(max_reward_iter, set())

    print(f"Detected state for run: {run_dir}")
    print(f"  Max iteration with rewards: {max_reward_iter}")
    print(f"  Samples with rewards at iter {max_reward_iter}: {len(rewards_at_max)}")
    print(f"  Samples with results at iter {max_reward_iter}: {len(results_at_max)}")

    # Determine if we need to run training or start next iteration
    if len(results_at_max) >= len(rewards_at_max):
        # All rewards have results - start next iteration
        return {
            "resume_iteration": max_reward_iter + 1,
            "needs_training": False,
            "completed_iterations": max_reward_iter + 1,
        }
    else:
        # Missing results - need to run training for existing rewards
        # Find the latest attempt for each sample
        sample_rewards = {}
        for f in reward_files:
            match = reward_pattern.match(os.path.basename(f))
            if match:
                iteration = int(match.group(1))
                sample = int(match.group(2))
                attempt = int(match.group(3))
                if iteration == max_reward_iter:
                    if sample not in sample_rewards or attempt > sample_rewards[sample][1]:
                        sample_rewards[sample] = (f, attempt)

        return {
            "resume_iteration": max_reward_iter,
            "needs_training": True,
            "reward_files": {sample: info[0] for sample, info in sample_rewards.items()},
            "completed_iterations": max_reward_iter,
        }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine base output directory
    base_dir = None
    if "paths" in config and "output_dir" in config["paths"]:
        base_dir = config["paths"]["output_dir"]
    else:
        # Fall back to platform-specific default
        if sys.platform == "win32":
            base_dir = os.path.join(
                os.environ.get("USERPROFILE", ""),
                "AppData/LocalLow/DefaultCompany/AR4/EurekaRewards",
            )
        else:
            base_dir = os.path.expanduser(
                "~/.config/unity3d/DefaultCompany/AR4/EurekaRewards"
            )

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"eureka_run_{timestamp}")

    # Set all subdirectory paths based on timestamped run directory
    config["run_dir"] = run_dir
    config["rewards_dir"] = os.path.join(run_dir, "rewards")
    config["results_dir"] = os.path.join(run_dir, "results")
    config["conversations_dir"] = os.path.join(run_dir, "conversations")

    # Only set tensorboard dir if tensorboard logging is enabled
    if config.get("logging", {}).get("tensorboard", False):
        config["tensorboard_dir"] = os.path.join(run_dir, "tensorboard")
    else:
        config["tensorboard_dir"] = None

    # Note: Task-specific settings will be applied in main() after task selection

    # Ensure directories exist
    os.makedirs(config["rewards_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)
    os.makedirs(config["conversations_dir"], exist_ok=True)

    # Only create tensorboard dir if tensorboard logging is enabled
    if config["tensorboard_dir"]:
        os.makedirs(config["tensorboard_dir"], exist_ok=True)

    return config


def load_config_for_continue(run_dir: str) -> dict:
    """Load configuration from an existing run directory."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"No config.json found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Ensure directories still exist (they should)
    os.makedirs(config["rewards_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)
    os.makedirs(config["conversations_dir"], exist_ok=True)
    if config.get("tensorboard_dir"):
        os.makedirs(config["tensorboard_dir"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run Eureka reward generation for Unity AR4 robot"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="T0",
        help="Task to run (default: T0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/eureka_config_test.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--continue-run",
        type=str,
        metavar="PATH",
        dest="continue_run",
        help="Resume an interrupted Eureka run from the given path",
    )

    args = parser.parse_args()

    # Handle continue mode vs new run mode
    run_state = None
    if args.continue_run:
        # Continue mode: load config from existing run
        if not os.path.exists(args.continue_run):
            print(f"ERROR: Run directory not found: {args.continue_run}")
            return 1

        print(f"Continuing run from: {args.continue_run}")
        config = load_config_for_continue(args.continue_run)
        run_state = detect_run_state(args.continue_run)

        print(f"\nResume state:")
        print(f"  Resume from iteration: {run_state['resume_iteration']}")
        print(f"  Needs training: {run_state['needs_training']}")
        if run_state['needs_training']:
            print(f"  Reward files to train: {len(run_state['reward_files'])}")
    else:
        # New run mode: load config from YAML
        config = load_config(args.config)

        # Apply task-specific config
        config["task"] = args.task
        if args.task not in config.get("tasks", {}):
            print(f"ERROR: Task '{args.task}' not found in tasks section of config")
            print(f"Available tasks: {list(config.get('tasks', {}).keys())}")
            return 1

    task_config = config["tasks"][config["task"]]
    config["task_name"] = task_config.get("name", config["task"])
    config["task_description"] = task_config.get("description", "")

    # Use task-specific settings (with defaults if not specified)
    config["max_training_steps"] = task_config.get("training_steps", 100000)
    config["max_episode_steps"] = task_config.get("max_episode_steps", 2000)

    # Get Unity build path from task config
    config["unity_build_path"] = task_config.get("unity_build", "")

    # Get optional settings from config
    config["no_graphics"] = config.get("no_graphics", False)
    config["parallel_envs"] = config.get("parallel_envs", 4)

    # Display configuration
    print("=" * 60)
    if run_state:
        print("EUREKA - Unity AR4 Robot Reward Generation (CONTINUE MODE)")
    else:
        print("EUREKA - Unity AR4 Robot Reward Generation")
    print("=" * 60)
    print(f"Run directory: {config['run_dir']}")
    print(f"Task: {config['task']} - {config['task_name']}")
    print(f"Description: {config['task_description']}")
    print(f"Model: {config['model']}")
    print(f"Iterations: {config['iterations']}")
    print(f"Samples per iteration: {config['samples']}")
    print(f"Max training steps: {config['max_training_steps']}")
    print(f"Max episode steps: {config['max_episode_steps']}")
    print(f"Unity build: {config['unity_build_path']}")
    print(f"Parallel environments: {config.get('parallel_envs', 4)}")
    print(f"No graphics: {config.get('no_graphics', False)}")
    if run_state:
        print(f"Resuming from iteration: {run_state['resume_iteration']}")
        print(f"Needs training first: {run_state['needs_training']}")
    print("-" * 60)

    # Check prerequisites
    if not os.path.exists(config["unity_build_path"]):
        print(f"ERROR: Unity build not found at {config['unity_build_path']}")
        print("Please build the Unity project or specify path with --unity-build")
        return 1

    # Check LLM credentials
    if config.get("llm_provider", "azure") == "azure":
        if not os.getenv("AZURE_ENDPOINT_URL") or not os.getenv("AZURE_OPENAI_API_KEY"):
            print("ERROR: Azure OpenAI credentials not found.")
            print("Please set AZURE_ENDPOINT_URL and AZURE_OPENAI_API_KEY in .env file")
            return 1

    # Only save config and copy agent file for new runs
    if not run_state:
        # Save the complete config to the run directory for reproducibility
        os.makedirs(config["run_dir"], exist_ok=True)
        config_save_path = os.path.join(config["run_dir"], "config.json")
        with open(config_save_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nConfiguration saved to: {config_save_path}")

        # Copy agent file for posterity if specified in task config
        copy_agent_file_if_specified(config)

    # Run Eureka
    if run_state:
        print(f"\nResuming Eureka at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\nStarting Eureka at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        eureka = EurekaUnity(config)
        eureka.run(run_state=run_state)

        print("\n" + "=" * 60)
        print(
            f"Eureka completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 60)

        # Print results location
        print(f"\nAll outputs saved to: {config['run_dir']}")
        print(f"  Results: {config['results_dir']}")
        print(f"  Reward functions: {config['rewards_dir']}")
        if config.get("tensorboard_dir"):
            print(f"  TensorBoard logs: {config['tensorboard_dir']}")
        if config.get("conversations_dir"):
            print(f"  Conversation histories: {config['conversations_dir']}")

        return 0

    except KeyboardInterrupt:
        print("\n\nEureka interrupted by user")
        return 2

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
