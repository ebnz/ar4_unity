"""
Training Orchestrator that launches Unity subprocesses and connects SB3 training
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import multiprocessing as mp

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch.nn as nn
import torch
from training.common import RewardCompilationError
import psutil

# Unity environment
import sys
from env.unity_vec_env import UnityVecEnv
from env.hybrid_action_wrapper import HybridActionWrapper

# Shared training utilities
from training.common import (
    find_free_port,
    ensure_port_is_free,
    train_with_unity_subprocess,
    CheckpointWithONNXCallback,
    save_final_model_with_onnx,
)


class ResultEnricher:
    """Enriches raw checkpoint data with aggregated metrics for LLM feedback and selection"""

    @staticmethod
    def enrich_result(result_data: Dict, all_checkpoints: List[Dict]) -> Dict:
        """
        Adds aggregated metrics to the result data.

        Args:
            result_data: The final checkpoint data (last checkpoint)
            all_checkpoints: All checkpoints from the training run

        Returns:
            Enriched result with additional fields:
            - final_success_rate: Success rate at last checkpoint
            - max_success_rate: Highest success rate achieved during training
            - mean_success_rate: Average success rate across all checkpoints
        """
        if not all_checkpoints:
            return result_data

        # Extract success rates from all checkpoints
        success_rates = [cp.get("success_rate", 0.0) for cp in all_checkpoints]

        # Add enriched metrics
        enriched = dict(result_data)  # Copy the final checkpoint data
        enriched["final_success_rate"] = result_data.get("success_rate", 0.0)
        enriched["max_success_rate"] = max(success_rates)
        enriched["mean_success_rate"] = np.mean(success_rates)

        return enriched

    @staticmethod
    def format_for_llm(result: Dict) -> str:
        """
        Formats the enriched result data for LLM consumption.

        Args:
            result: Enriched result data (output from enrich_result)

        Returns:
            Formatted string with performance metrics, reward components, and observations
        """
        lines = ["Policy Performance Metrics:"]

        # Success rate metrics
        final_success = result.get(
            "final_success_rate", result.get("success_rate", 0.0)
        )
        lines.append(f"- Final Success Rate (last checkpoint): {final_success:.2%}")

        if "max_success_rate" in result:
            lines.append(
                f"- Maximum Success Rate (peak during training): {result['max_success_rate']:.2%}"
            )

        if "mean_success_rate" in result:
            lines.append(
                f"- Average Success Rate (across training): {result['mean_success_rate']:.2%}"
            )

        # Reward component analysis
        if "reward_components" in result and result["reward_components"]:
            lines.append("\nReward Component Analysis:")
            for component_data in result["reward_components"]:
                component_name = component_data.get("name", "Unnamed Component")
                stats = component_data.get("stats", {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                lines.append(f"- {component_name}: mean={mean:.3f}, std={std:.3f}")
        else:
            lines.append("\nNo detailed reward component data available.")

        # Observation statistics
        if "observations" in result and result["observations"]:
            lines.append("\nObservation Statistics:")
            for obs in result["observations"]:
                name = obs.get("name", "Unnamed Observation")
                stats = obs.get("stats", {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                min_ = stats.get("min", 0.0)
                max_ = stats.get("max", 0.0)
                lines.append(
                    f"- {name}: mean={mean:.3f}, std={std:.3f}, min={min_:.3f}, max={max_:.3f}"
                )

        return "\n".join(lines)

    @staticmethod
    def select_best(enriched_results: List[Dict]) -> int:
        """
        Selects the best result from a list of enriched results.

        Selection logic:
        1. Use final success rate if any result has > 0%
        2. If all results have 0% final success, use max success rate (peak during training)

        Args:
            enriched_results: List of enriched results (output from enrich_result)

        Returns:
            Index of the best result
        """
        if not enriched_results:
            raise ValueError("Cannot select best from empty results list")

        # Get final success rates
        final_success_rates = [
            r.get("final_success_rate", r.get("success_rate", 0.0))
            for r in enriched_results
        ]
        max_final_success = max(final_success_rates)

        if max_final_success > 0:
            # At least one result achieved success - use final success rate
            best_idx = int(np.argmax(final_success_rates))
            print(
                f"Selected best by final success rate: {final_success_rates[best_idx]:.2%}"
            )
        else:
            # All results have 0% final success - use max success rate (shows learning potential)
            max_success_rates = [
                r.get("max_success_rate", 0.0) for r in enriched_results
            ]
            best_idx = int(np.argmax(max_success_rates))
            print(
                f"All final success rates are 0%. Selected best by max success rate: {max_success_rates[best_idx]:.2%}"
            )

        # Log detailed info about the selected result
        best = enriched_results[best_idx]
        print(f"  Selected index: {best_idx}")
        print(f"    Final: {best.get('final_success_rate', 0.0):.2%}")
        if "max_success_rate" in best:
            print(f"    Max: {best['max_success_rate']:.2%}")
        if "mean_success_rate" in best:
            print(f"    Mean: {best['mean_success_rate']:.2%}")

        return best_idx


class TrainingOrchestrator:
    """Manages training runs with different reward functions using actual SB3 training"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Load training hyperparameters from config, with fallbacks
        training_config = self.config.get("training", {})

        # Default hyperparameters
        self.hyperparameters = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": nn.Tanh,
            },
            "verbose": 0,  # Reduce verbosity for multiple runs
            "device": "auto",
        }

        # Merge config settings over defaults, handling nested policy_kwargs
        self.hyperparameters.update(
            {k: v for k, v in training_config.items() if k != "policy_kwargs"}
        )
        if "policy_kwargs" in training_config:
            self.hyperparameters["policy_kwargs"].update(
                training_config["policy_kwargs"]
            )

        # Convert activation_fn from string to class if needed
        if "activation_fn" in self.hyperparameters["policy_kwargs"]:
            act_fn_str = self.hyperparameters["policy_kwargs"]["activation_fn"]
            if isinstance(act_fn_str, str):
                if act_fn_str.lower() == "relu":
                    self.hyperparameters["policy_kwargs"]["activation_fn"] = nn.ReLU
                elif act_fn_str.lower() == "tanh":
                    self.hyperparameters["policy_kwargs"]["activation_fn"] = nn.Tanh
                else:
                    print(
                        f"Warning: Unknown activation function '{act_fn_str}'. Defaulting to Tanh."
                    )
                    self.hyperparameters["policy_kwargs"]["activation_fn"] = nn.Tanh

        # The algorithm name is not a PPO hyperparameter
        if "algorithm" in self.hyperparameters:
            del self.hyperparameters["algorithm"]

    def get_reward_filename(self, iteration: int, run_id: int, attempt: int) -> str:
        """Gets the filename for a reward function, including the attempt number."""
        # Special case for final evaluation (iteration 999)
        if iteration == 999:
            return "final_evaluation.cs"
        return f"reward_iter{iteration:02d}_sample{run_id:03d}_attempt{attempt}.cs"

    def get_result_filename(self, iteration: int, run_id: int, attempt: int) -> str:
        """Gets the filename for a result JSON, including the attempt number."""
        # Special case for final evaluation (iteration 999)
        if iteration == 999:
            return f"results_final_seed{run_id}.json"
        return f"results_iter{iteration:02d}_sample{run_id:03d}_attempt{attempt}.json"

    def run_evaluation_with_retries(
        self,
        reward_samples: List[str],
        max_steps: int,
        iteration: int,
        llm,
        save_callback,
        task_prompt: str,
        conversation_history: List[Dict],
        conversations_dir: str = None,
    ) -> List[Dict]:
        """
        Run parallel evaluation with an asynchronous retry mechanism for compilation errors.
        """
        max_retries = self.config.get("compilation_retries", 2)
        num_samples = len(reward_samples)

        # State tracking for each run_id
        run_states = {
            i: {
                "attempt": 1,
                "status": "pending",
                "history": list(conversation_history) if conversation_history else [],
            }
            for i in range(num_samples)
        }

        tasks_to_run = list(range(num_samples))
        collected_metadata = {}

        with mp.Pool(processes=min(num_samples, mp.cpu_count())) as pool:
            active_procs = {}  # {run_id: AsyncResult}

            while len(collected_metadata) < num_samples:
                # Submit new tasks from the queue if there's pool capacity
                while tasks_to_run and len(active_procs) < pool._processes:
                    run_id = tasks_to_run.pop(0)
                    seed = run_id
                    attempt = run_states[run_id]["attempt"]

                    async_result = pool.apply_async(
                        self.train_with_reward,
                        args=(max_steps, seed, run_id, iteration, attempt),
                    )
                    active_procs[run_id] = async_result
                    run_states[run_id]["status"] = "running"

                # Check for completed processes
                for run_id, result_proc in list(active_procs.items()):
                    if result_proc.ready():
                        try:
                            metadata = result_proc.get()

                            if metadata.get("is_compilation_error"):
                                if run_states[run_id]["attempt"] <= max_retries:
                                    print(
                                        f"Run {run_id} failed compilation. Retrying (attempt {run_states[run_id]['attempt']}/{max_retries})..."
                                    )

                                    # Construct a new prompt with the error feedback
                                    compilation_error = metadata.get(
                                        "error", "Unknown compilation error."
                                    )
                                    faulty_code = reward_samples[run_id]

                                    error_feedback_prompt = (
                                        f"The previously generated reward function had a compilation error:\n"
                                        f"```\n{compilation_error}\n```\n"
                                        f"Here is the faulty code:\n"
                                        f"```csharp\n{faulty_code}\n```\n"
                                        f"Please fix the error and provide a corrected reward function. "
                                        f"Remember: Write ONLY the code body, do NOT include any function signature like 'void ComputeReward(...)'. "
                                        f"The code will be wrapped automatically."
                                    )

                                    # Add the faulty response and the error feedback to the conversation history
                                    run_states[run_id]["history"].append(
                                        {"role": "assistant", "content": faulty_code}
                                    )
                                    run_states[run_id]["history"].append(
                                        {
                                            "role": "user",
                                            "content": error_feedback_prompt,
                                        }
                                    )

                                    print(f"Retrying run {run_id} with new prompt...")

                                    retry_result = llm.generate(
                                        prompt=error_feedback_prompt,
                                        conversation_history=run_states[run_id][
                                            "history"
                                        ],
                                    )

                                    # Extract code from the result
                                    from llm.conversation_utils import (
                                        extract_code_from_llm_result,
                                    )

                                    new_sample = extract_code_from_llm_result(
                                        retry_result
                                    )
                                    reward_samples[run_id] = (
                                        new_sample  # Update the sample in-memory
                                    )

                                    # Increment attempt number first
                                    run_states[run_id]["attempt"] += 1

                                    # Save retry conversation
                                    if conversations_dir:
                                        from llm.conversation_utils import (
                                            save_llm_result_conversation,
                                        )

                                        save_llm_result_conversation(
                                            conversations_dir=conversations_dir,
                                            iteration=iteration,
                                            run_id=run_id,
                                            attempt=run_states[run_id]["attempt"],
                                            conv_type="retry",
                                            llm_result=retry_result,
                                            error_context=compilation_error,
                                        )

                                    # Save the new reward code
                                    save_callback(
                                        new_sample,
                                        iteration,
                                        run_id,
                                        run_states[run_id]["attempt"],
                                    )

                                    tasks_to_run.append(run_id)  # Add back to queue
                                    run_states[run_id]["status"] = "pending"
                                else:
                                    print(
                                        f"Run {run_id} failed after max retries. Storing error metadata."
                                    )
                                    collected_metadata[run_id] = metadata
                                    run_states[run_id]["status"] = "failed"
                            else:
                                collected_metadata[run_id] = metadata
                                run_states[run_id]["status"] = "completed"
                        except Exception as e:
                            print(
                                f"CRITICAL ERROR getting result for run {run_id}: {e}"
                            )
                            collected_metadata[run_id] = {
                                "run_id": run_id,
                                "error": str(e),
                                "is_compilation_error": False,
                            }
                            run_states[run_id]["status"] = "failed"

                        # Remove from active processes dict
                        del active_procs[run_id]

                time.sleep(1)

        print("All training runs completed!")

        # Load results from files, using the final metadata for each run
        final_metadata_list = [collected_metadata[i] for i in range(num_samples)]
        results = self.load_final_results(final_metadata_list)

        return results

    def train_with_reward(
        self,
        max_training_steps: int,
        seed: int = 0,
        run_id: int = 0,
        iteration: int = 0,
        attempt: int = 1,
    ) -> Dict:
        """Train with a specific reward function by launching Unity and connecting SB3 - returns metadata only"""
        try:
            # 1. Find a free port for this run_id
            preferred_port = 5004 + run_id
            actual_port = find_free_port(preferred_port)

            print(
                f"Starting training run {run_id} (attempt {attempt}) with {max_training_steps} steps on port {actual_port}..."
            )

            # 2. Generate reward and result filenames with attempt number
            reward_filename = self.get_reward_filename(iteration, run_id, attempt)
            result_filename = self.get_result_filename(iteration, run_id, attempt)

            # 3. Start SB3 environment creation in thread, then launch Unity
            self._train_with_threading_approach(
                reward_filename,
                max_training_steps,
                run_id,
                result_filename,
                seed,
                actual_port,
            )

            # 4. Return minimal metadata - Unity results will be loaded later
            training_metadata = {
                "run_id": run_id,
                "iteration": iteration,
                "attempt": attempt,
                "reward_filename": reward_filename,
                "result_filename": result_filename,
                "start_time": time.time(),
                "port": actual_port,
                "max_training_steps": max_training_steps,
                "seed": seed,
                "config_snapshot": {
                    "unity_build_path": self.config.get("unity_build_path"),
                    "max_episode_steps": self.config.get("max_episode_steps", 2000),
                    "parallel_envs": self.config.get("parallel_envs", 4),
                    "no_graphics": self.config.get("no_graphics", False),
                },
            }

            return training_metadata

        except RewardCompilationError as e:
            print(
                f"Training run {run_id} (attempt {attempt}) failed due to compilation error: {e}"
            )
            return {
                "run_id": run_id,
                "iteration": iteration,
                "attempt": attempt,
                "error": str(e),
                "is_compilation_error": True,  # Flag for retry logic
                "start_time": time.time(),
            }
        except Exception as e:
            print(f"Training run {run_id} failed: {e}")
            return {
                "run_id": run_id,
                "iteration": iteration,
                "attempt": attempt,
                "error": str(e),
                "start_time": time.time(),
            }

    def _train_with_threading_approach(
        self,
        reward_filename: str,
        max_training_steps: int,
        run_id: int,
        result_filename: str,
        seed: int,
        port: int,
    ) -> Dict:
        """Train using threading approach: start environment in thread, then launch Unity"""

        def training_callback(env):
            """Callback that performs SB3 training once environment is connected"""
            return self._train_with_sb3(
                env, reward_filename, run_id, max_training_steps
            )

        # Use common subprocess training function
        return train_with_unity_subprocess(
            config=self.config,
            port=port,
            result_filename=result_filename,
            reward_filename=reward_filename,
            run_id=run_id,
            max_training_steps=max_training_steps,
            gamma=self.hyperparameters["gamma"],
            seed=seed,
            training_callback=training_callback,
        )

    def _train_with_sb3(
        self,
        env: VecNormalize,
        reward_filename: str,
        run_id: int,
        max_training_steps: int,
    ) -> Dict:
        """Perform actual training with Stable Baselines3"""

        # Set up TensorBoard logging if enabled
        tensorboard_log_dir = self.config.get("tensorboard_dir")
        if not tensorboard_log_dir:
            tensorboard_log_dir = None  # Disable TensorBoard if no dir specified

        # Create PPO model with optional TensorBoard logging
        model = PPO(
            env=env, tensorboard_log=tensorboard_log_dir, **self.hyperparameters
        )

        # Set up callbacks
        callbacks = []

        # Add checkpoint callback if save_frequency is specified
        save_freq = self.config.get("logging", {}).get("save_frequency")
        if save_freq:
            checkpoints_dir = os.path.join(
                self.config.get("results_dir", "."),
                "checkpoints",
                f"run_{run_id}_{reward_filename.replace('.cs', '')}",
            )
            os.makedirs(checkpoints_dir, exist_ok=True)

            # Use shared checkpoint callback
            checkpoint_callback = CheckpointWithONNXCallback(
                save_freq=save_freq, save_path=checkpoints_dir, env=env, verbose=0
            )
            callbacks.append(checkpoint_callback)
            print(
                f"Will save checkpoints (with ONNX) every {save_freq} steps to {checkpoints_dir}"
            )

        # Train the model
        print(f"Training PPO model for {max_training_steps} steps (run {run_id})...")
        start_time = time.time()

        # Only pass tb_log_name if tensorboard is enabled
        learn_kwargs = {
            "total_timesteps": max_training_steps,
            "progress_bar": False,  # Disable for cleaner output in parallel
            "reset_num_timesteps": True,
            "callback": callbacks if callbacks else None,
        }

        if tensorboard_log_dir:
            learn_kwargs["tb_log_name"] = (
                f"run_{run_id}_{reward_filename.replace('.cs', '')}"
            )

        model.learn(**learn_kwargs)

        training_time = time.time() - start_time

        # Save final model and export to ONNX if save_frequency is set
        if save_freq:
            save_final_model_with_onnx(model, env, checkpoints_dir, "final_model")

        print(f"Training run {run_id} completed in {training_time:.1f}s")

        # Return basic info - Unity will provide detailed results
        return {
            "run_id": run_id,
            "training_time": training_time,
            "max_training_steps": max_training_steps,
            "reward_filename": reward_filename,
        }

    def load_final_results(self, final_metadata_list: List[Dict]) -> List[Dict]:
        """Loads detailed results from JSON files based on the final metadata from each run."""
        results = []
        results_dir = self.config["results_dir"]

        for metadata in final_metadata_list:
            run_id = metadata["run_id"]

            # If the run ended with an error (compilation or otherwise), there's no result file.
            if "error" in metadata:
                results.append(
                    {
                        "run_id": run_id,
                        "success_rate": 0.0,
                        "total_episodes": 0,
                        "reward_components": {},
                        "is_compilation_error": metadata.get(
                            "is_compilation_error", False
                        ),
                        "error": metadata.get("error", "Unknown error"),
                    }
                )
                continue

            result_filename = metadata.get("result_filename")
            if not result_filename:
                print(
                    f"No result filename in metadata for run {run_id}, assuming failure."
                )
                results.append(
                    {
                        "run_id": run_id,
                        "success_rate": 0.0,
                        "error": "Missing result filename.",
                    }
                )
                continue

            result_path = os.path.join(results_dir, result_filename)

            try:
                with open(result_path, "r") as f:
                    data = json.load(f)

                if "checkpoints" in data and data["checkpoints"]:
                    last_checkpoint = data["checkpoints"][-1]
                    # Enrich with aggregated metrics
                    enriched_result = ResultEnricher.enrich_result(
                        last_checkpoint, data["checkpoints"]
                    )
                    # Add metadata from top-level data (not in checkpoint)
                    enriched_result["run_id"] = data.get("run_id", run_id)
                    if "reward_path" in data:
                        enriched_result["reward_path"] = data["reward_path"]
                    results.append(enriched_result)
                else:
                    results.append(
                        {
                            "run_id": run_id,
                            "success_rate": 0.0,
                            "error": "Result file is empty.",
                        }
                    )

            except FileNotFoundError:
                print(f"Result file not found: {result_filename}, assuming failure.")
                results.append(
                    {
                        "run_id": run_id,
                        "success_rate": 0.0,
                        "error": "Result file not found.",
                    }
                )
            except Exception as e:
                print(f"Failed to load {result_filename}: {e}")
                results.append(
                    {
                        "run_id": run_id,
                        "success_rate": 0.0,
                        "error": f"Failed to load result file: {e}",
                    }
                )

        return results
