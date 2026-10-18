"""
Reward parameter sweep runner.

Runs systematic parameter sweeps over reward function parameters,
training multiple configurations in parallel and collecting results.
"""

import os
import sys
import json
import shutil
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

# Add parent directory to path for direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from sweeps.sweep_config import SweepConfig, ParameterGrid
from training.common import (
    find_free_port,
    train_with_unity_subprocess,
    create_timestamped_run_directory,
    setup_hyperparameters,
    load_training_config,
    load_results_from_json,
)


@dataclass
class SweepRun:
    """Represents a single run in a sweep."""

    run_id: int
    params: Dict[str, Any]
    seed: int
    run_name: str
    run_dir: str
    reward_path: str
    result_path: str


@dataclass
class SweepResult:
    """Result from a single sweep run."""

    run_id: int
    run_name: str
    params: Dict[str, Any]
    seed: int
    success_rate: float
    mean_reward: float
    training_time: float
    error: Optional[str] = None


class RewardSweep:
    """
    Manages a parameter sweep over reward function parameters.

    Example usage:
        config = SweepConfig(...)
        sweep = RewardSweep(config)
        results = sweep.run()
        sweep.save_summary()
    """

    def __init__(self, config: SweepConfig, base_dir: Optional[str] = None):
        """
        Initialize the sweep.

        Args:
            config: SweepConfig defining the sweep parameters
            base_dir: Base directory for sweep outputs (default: platform-specific)
        """
        self.config = config

        # Set up base directory
        if base_dir is None:
            if sys.platform == "win32":
                base_dir = os.path.join(
                    os.environ.get("USERPROFILE", ""),
                    "AppData/LocalLow/DefaultCompany/AR4/Sweeps",
                )
            else:
                base_dir = os.path.expanduser(
                    "~/.config/unity3d/DefaultCompany/AR4/Sweeps"
                )

        self.sweep_dir = os.path.join(base_dir, config.output_dir)
        self.results: List[SweepResult] = []

        # Load base training config
        self.base_config = self._load_base_config()

    def _load_base_config(self) -> Dict:
        """Load the base training configuration."""
        config_path = self.config.base_training_config

        # Try relative to Assets directory
        if not os.path.isabs(config_path):
            assets_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(os.path.dirname(assets_dir), config_path)

        if not os.path.exists(config_path):
            # Try relative to Python directory
            python_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(python_dir, self.config.base_training_config)

        # Use common config loader without creating run directories (sweep manages its own)
        return load_training_config(config_path, create_run_dir=False)

    def _load_template(self) -> str:
        """Load the reward function template."""
        template_path = self.config.template_path

        # Try relative to Assets directory
        if not os.path.isabs(template_path):
            assets_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_path = os.path.join(os.path.dirname(assets_dir), template_path)

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Reward template not found: {template_path}")

        with open(template_path, "r") as f:
            return f.read()

    def _render_reward(self, template: str, params: Dict[str, Any]) -> str:
        """Render the reward function template with parameters."""
        result = template
        for key, value in params.items():
            placeholder = "{" + key + "}"
            # Format floats appropriately for C#
            if isinstance(value, float):
                str_value = f"{value}f"
            else:
                str_value = str(value)
            result = result.replace(placeholder, str_value)
        return result

    def _setup_run(self, run_id: int, params: Dict[str, Any], seed: int) -> SweepRun:
        """Set up directories and files for a single run."""
        run_name = self.config.get_run_name(params, seed)
        run_dir = os.path.join(self.sweep_dir, f"run_{run_name}")

        # Create run directory structure
        os.makedirs(run_dir, exist_ok=True)
        rewards_dir = os.path.join(run_dir, "rewards")
        results_dir = os.path.join(run_dir, "results")
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        tensorboard_dir = os.path.join(run_dir, "tensorboard")

        for d in [rewards_dir, results_dir, checkpoints_dir, tensorboard_dir]:
            os.makedirs(d, exist_ok=True)

        # Render and save reward function
        template = self._load_template()
        reward_code = self._render_reward(template, params)
        reward_path = os.path.join(rewards_dir, "reward.txt")
        with open(reward_path, "w") as f:
            f.write(reward_code)

        # Save run config
        run_config = {
            "run_id": run_id,
            "run_name": run_name,
            "params": params,
            "seed": seed,
            "sweep_name": self.config.name,
        }
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(run_config, f, indent=2)

        result_path = os.path.join(results_dir, "results.json")

        return SweepRun(
            run_id=run_id,
            params=params,
            seed=seed,
            run_name=run_name,
            run_dir=run_dir,
            reward_path=reward_path,
            result_path=result_path,
        )

    def _run_single(self, sweep_run: SweepRun) -> SweepResult:
        """Execute a single training run."""
        start_time = time.time()

        try:
            # Build config for this run
            run_config = dict(self.base_config)
            run_config["run_dir"] = sweep_run.run_dir
            run_config["rewards_dir"] = os.path.join(sweep_run.run_dir, "rewards")
            run_config["results_dir"] = os.path.join(sweep_run.run_dir, "results")
            run_config["tensorboard_dir"] = os.path.join(
                sweep_run.run_dir, "tensorboard"
            )
            run_config["checkpoints_dir"] = os.path.join(
                sweep_run.run_dir, "checkpoints"
            )

            if self.config.max_training_steps:
                run_config["max_training_steps"] = self.config.max_training_steps

            # Find free port
            port = find_free_port(5004 + sweep_run.run_id)

            # Training callback
            def training_callback(env):
                from stable_baselines3 import PPO
                from training.common import (
                    CheckpointWithONNXCallback,
                    save_final_model_with_onnx,
                )

                # Get hyperparameters from config using common setup
                training_config = run_config.get("training", {})
                hyperparameters = setup_hyperparameters(training_config)
                hyperparameters["verbose"] = 0  # Override for sweep runs

                model = PPO(
                    env=env,
                    tensorboard_log=run_config["tensorboard_dir"],
                    **hyperparameters,
                )

                model.learn(
                    total_timesteps=run_config.get("max_training_steps", 50000),
                    progress_bar=False,
                )

                # Save final model
                save_final_model_with_onnx(
                    model, env, run_config["checkpoints_dir"], "final_model"
                )

                return {"completed": True}

            # Run training
            train_with_unity_subprocess(
                config=run_config,
                port=port,
                result_filename="results.json",
                reward_filename="reward.txt",
                run_id=sweep_run.run_id,
                max_training_steps=run_config.get("max_training_steps", 50000),
                gamma=run_config.get("training", {}).get("gamma", 0.99),
                seed=sweep_run.seed,
                training_callback=training_callback,
            )

            training_time = time.time() - start_time

            # Load results using common loader (handles both 'results' and 'checkpoints' keys)
            success_rate = 0.0
            mean_reward = 0.0

            result_data = load_results_from_json(sweep_run.result_path)
            if result_data:
                success_rate = result_data.get("success_rate", 0.0)
                if "episode_reward_stats" in result_data:
                    mean_reward = result_data["episode_reward_stats"].get("mean", 0.0)

            return SweepResult(
                run_id=sweep_run.run_id,
                run_name=sweep_run.run_name,
                params=sweep_run.params,
                seed=sweep_run.seed,
                success_rate=success_rate,
                mean_reward=mean_reward,
                training_time=training_time,
            )

        except Exception as e:
            return SweepResult(
                run_id=sweep_run.run_id,
                run_name=sweep_run.run_name,
                params=sweep_run.params,
                seed=sweep_run.seed,
                success_rate=0.0,
                mean_reward=0.0,
                training_time=time.time() - start_time,
                error=str(e),
            )

    def setup(self):
        """Set up the sweep directory structure."""
        os.makedirs(self.sweep_dir, exist_ok=True)

        # Save sweep config
        self.config.save(os.path.join(self.sweep_dir, "sweep_config.json"))

        # Create combined tensorboard directory
        os.makedirs(os.path.join(self.sweep_dir, "tensorboard_combined"), exist_ok=True)

        print(f"Sweep directory: {self.sweep_dir}")
        print(f"Total runs: {self.config.get_total_runs()}")

    def run(self, sequential: bool = False) -> List[SweepResult]:
        """
        Run the sweep.

        Args:
            sequential: If True, run sequentially instead of parallel

        Returns:
            List of SweepResult objects
        """
        self.setup()

        # Generate all runs
        runs: List[SweepRun] = []
        run_id = 0

        for params in self.config.parameter_grid:
            for seed in self.config.seeds:
                sweep_run = self._setup_run(run_id, params, seed)
                runs.append(sweep_run)
                run_id += 1

        print(f"\nStarting {len(runs)} training runs...")

        if sequential:
            # Sequential execution
            for run in runs:
                print(f"\nRunning: {run.run_name}")
                result = self._run_single(run)
                self.results.append(result)
                self._print_result(result)
        else:
            # Parallel execution
            with mp.Pool(processes=self.config.parallel_runs) as pool:
                for result in pool.imap_unordered(self._run_single, runs):
                    self.results.append(result)
                    self._print_result(result)

        return self.results

    def _print_result(self, result: SweepResult):
        """Print a single result."""
        status = "ERROR" if result.error else "OK"
        print(
            f"  [{status}] {result.run_name}: success={result.success_rate:.2%}, "
            f"reward={result.mean_reward:.2f}, time={result.training_time:.1f}s"
        )
        if result.error:
            print(f"       Error: {result.error}")

    def save_summary(self):
        """Save summary of all results."""
        summary = {
            "sweep_name": self.config.name,
            "description": self.config.description,
            "total_runs": len(self.results),
            "completed_runs": sum(1 for r in self.results if r.error is None),
            "failed_runs": sum(1 for r in self.results if r.error is not None),
            "results": [
                {
                    "run_name": r.run_name,
                    "params": r.params,
                    "seed": r.seed,
                    "success_rate": r.success_rate,
                    "mean_reward": r.mean_reward,
                    "training_time": r.training_time,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        # Add aggregated stats per parameter combination
        param_stats = {}
        for r in self.results:
            if r.error:
                continue
            key = json.dumps(r.params, sort_keys=True)
            if key not in param_stats:
                param_stats[key] = {
                    "params": r.params,
                    "success_rates": [],
                    "rewards": [],
                }
            param_stats[key]["success_rates"].append(r.success_rate)
            param_stats[key]["rewards"].append(r.mean_reward)

        summary["aggregated"] = []
        for key, stats in param_stats.items():
            import numpy as np

            summary["aggregated"].append(
                {
                    "params": stats["params"],
                    "mean_success_rate": float(np.mean(stats["success_rates"])),
                    "std_success_rate": float(np.std(stats["success_rates"])),
                    "mean_reward": float(np.mean(stats["rewards"])),
                    "std_reward": float(np.std(stats["rewards"])),
                    "n_runs": len(stats["success_rates"]),
                }
            )

        # Sort by success rate
        summary["aggregated"].sort(key=lambda x: x["mean_success_rate"], reverse=True)

        summary_path = os.path.join(self.sweep_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")

        # Print best results
        if summary["aggregated"]:
            print("\nTop 3 parameter combinations:")
            for i, agg in enumerate(summary["aggregated"][:3]):
                print(f"  {i+1}. {agg['params']}")
                print(
                    f"     Success: {agg['mean_success_rate']:.2%} +/- {agg['std_success_rate']:.2%}"
                )


def run_sweep(config: SweepConfig, sequential: bool = False) -> List[SweepResult]:
    """Convenience function to run a sweep."""
    sweep = RewardSweep(config)
    results = sweep.run(sequential=sequential)
    sweep.save_summary()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run reward parameter sweep")
    parser.add_argument("--config", type=str, help="Path to sweep config JSON")
    parser.add_argument("--template", type=str, help="Path to reward template")
    parser.add_argument(
        "--training-config",
        type=str,
        default="config/training_config_T0.yaml",
        help="Base training config",
    )
    parser.add_argument(
        "--param",
        nargs="+",
        action="append",
        help="Parameter to sweep: --param name val1 val2 val3",
    )
    parser.add_argument(
        "--steps", type=int, default=50000, help="Training steps per run"
    )
    parser.add_argument("--parallel", type=int, default=4, help="Parallel runs")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Seeds for each parameter combo",
    )
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument("--name", type=str, default="reward_sweep", help="Sweep name")

    args = parser.parse_args()

    if args.config:
        # Load config from file
        config = SweepConfig.load(args.config)
    else:
        # Build config from arguments
        if not args.template:
            print("Error: Must specify --template or --config")
            sys.exit(1)

        grid = ParameterGrid()
        if args.param:
            for param_spec in args.param:
                name = param_spec[0]
                values = [float(v) if "." in v else int(v) for v in param_spec[1:]]
                grid.add(name, values)

        config = SweepConfig(
            name=args.name,
            template_path=args.template,
            base_training_config=args.training_config,
            parameter_grid=grid,
            max_training_steps=args.steps,
            parallel_runs=args.parallel,
            seeds=args.seeds,
        )

    print(f"Running sweep: {config.name}")
    print(f"Total runs: {config.get_total_runs()}")
    print(f"Parallel: {config.parallel_runs}")

    run_sweep(config, sequential=args.sequential)
