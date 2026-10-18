"""
Eureka-style reward generation main loop for Unity AR4 robot
"""

import os
import sys
import json
import subprocess
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import yaml

# Import the enhanced training orchestrator and LLM client
from prompts.system_prompt import system_prompt
from prompts.policy_feedback import policy_feedback_prompt
from prompts.code_feedback import code_feedback_prompt
from training.orchestrator import TrainingOrchestrator
from llm.reward_llm import RewardLLM
from llm.conversation_utils import (
    save_llm_result_conversation,
    extract_code_from_llm_result,
)
from utils.extract_env_info import (
    UnityAgentParser,
    EnvironmentInfo,
    ActionInfo,
)


class EurekaUnity:
    def __init__(self, config: Dict):
        self.config = config
        self.task = config["task"]
        self.model = config.get("model", "gpt-4")
        self.iterations = config.get("iterations", 5)
        self.samples = config.get("samples", 16)
        self.max_training_steps = config.get("max_training_steps", 100000)

        # Initialize components
        llm_config = {
            "provider": config.get("llm_provider", "azure"),
            "model": self.model,
            "temperature": config.get("temperature", 1.0),
        }
        # Only set max_completion_tokens if explicitly provided in config
        # Otherwise, let RewardLLM auto-detect based on model type (o1 models need 16000+)
        if "max_completion_tokens" in config:
            llm_config["max_completion_tokens"] = config["max_completion_tokens"]

        self.llm = RewardLLM(llm_config)
        self.training_orchestrator = TrainingOrchestrator(config)
        self.feedback_analyzer = FeedbackAnalyzer(self.llm)
        self.task_info = self.load_task_info(self.task)
        self.conversations_dir = config.get("conversations_dir", ".")

    def run(self, run_state: dict = None):
        """Main Eureka loop

        Args:
            run_state: Optional resume state from detect_run_state(). If provided:
                - resume_iteration: Iteration to resume from
                - needs_training: If True, run training for existing rewards first
                - reward_files: Dict mapping sample index to reward file path (if needs_training)
        """
        # Determine starting iteration
        start_iteration = 0
        if run_state:
            start_iteration = run_state["resume_iteration"]

            # If we need to run training first, handle that before entering the main loop
            if run_state.get("needs_training"):
                self._resume_training(run_state)
                # After training, move to next iteration
                start_iteration = run_state["resume_iteration"] + 1

        # Check if we've already completed all iterations
        if start_iteration >= self.iterations:
            print(f"All {self.iterations} iterations already completed!")
            # TODO: Could optionally run final evaluation here
            return None

        # Initialize conversation history for in-context learning
        messages = [{"role": "system", "content": system_prompt}]

        for iteration in range(start_iteration, self.iterations):
            print(f"\n=== Eureka Iteration {iteration + 1}/{self.iterations} ===")

            # 1. Generate multiple reward function samples
            # Format task prompt based on whether it's first iteration or not
            if len(messages) == 1:  # First iteration (only system message)
                task_prompt = self.task_info.to_prompt()
                conversation_history = None
            else:
                # Subsequent iterations: use the combined feedback+instructions as prompt
                # Conversation history: [system, original_task, best_reward]
                task_prompt = messages[-1][
                    "content"
                ]  # This is the combined feedback + code_feedback_prompt
                conversation_history = messages[
                    :-1
                ]  # [system, original_task, best_reward]

            # Generate samples with temperature variation
            reward_results = self.llm.generate_batch(
                task_prompt,
                self.samples,
                conversation_history=conversation_history,
                temperature_range=(self.llm.temperature, self.llm.temperature),
            )

            # Extract code from results and save conversations
            reward_samples = []
            for i, result in enumerate(reward_results):
                # Extract code from the result
                code = extract_code_from_llm_result(result)
                reward_samples.append(code)

                # Save conversation
                save_llm_result_conversation(
                    conversations_dir=self.conversations_dir,
                    iteration=iteration,
                    run_id=i,
                    attempt=1,
                    conv_type="initial",
                    llm_result=result,
                )

            print(f"Generated {len(reward_samples)} reward samples")

            # Save reward samples immediately for debugging and monitoring
            self._save_initial_reward_samples(reward_samples, iteration)

            # 2. Parallel evaluation with retry mechanism for compilation errors
            evaluation_results = self.training_orchestrator.run_evaluation_with_retries(
                reward_samples,
                self.max_training_steps,
                iteration,
                self.llm,
                self._save_reward_sample_on_retry,
                task_prompt,
                conversation_history,
                self.conversations_dir,
            )

            # 3. Generate feedback for next iteration
            (
                best_result,
                best_reward_code,
                feedback_prompt,
            ) = self.feedback_analyzer.prepare_feedback(
                evaluation_results, reward_samples, self.task_info
            )

            # Generate feedback using LLM
            feedback_result = self.llm.generate(feedback_prompt)
            feedback = feedback_result.content

            # Save feedback conversation
            save_llm_result_conversation(
                conversations_dir=self.conversations_dir,
                iteration=iteration,
                run_id=best_result["run_id"],
                attempt=1,
                conv_type="feedback",
                llm_result=feedback_result,
            )

            print(f"Feedback: {feedback}")
            print(f"Best result: {best_result['run_id']}")

            # 4. Update conversation with feedback
            # Structure: [system, original_task, best_reward, feedback+code_feedback_prompt]
            if len(messages) == 1:
                # First iteration: add original task prompt
                messages.append({"role": "user", "content": self.task_info.to_prompt()})
            else:
                # Subsequent iterations: keep only [system, original_task]
                messages = messages[:2]

            # Add latest best reward and combined feedback message
            messages.append({"role": "assistant", "content": best_reward_code})
            # Combine feedback and code_feedback_prompt into one user message
            combined_feedback = f"{feedback}\n\n{code_feedback_prompt}"
            messages.append({"role": "user", "content": combined_feedback})

            print(
                f"Best success rate this iteration: {best_result.get('success_rate', 0.0):.2%}"
            )

        # Final evaluation with multiple seeds
        final_results = self.run_final_evaluation(best_reward_code)

        return final_results

    def _resume_training(self, run_state: dict):
        """
        Resume training for existing reward files when run was interrupted mid-training.

        Args:
            run_state: State dict containing:
                - resume_iteration: The iteration to resume
                - reward_files: Dict mapping sample index to reward file path
        """
        iteration = run_state["resume_iteration"]
        reward_files = run_state["reward_files"]

        print(f"\n=== Resuming Training for Iteration {iteration} ===")
        print(f"Found {len(reward_files)} reward files to train")

        # Load reward code from files
        reward_samples = []
        for sample_idx in sorted(reward_files.keys()):
            reward_path = reward_files[sample_idx]
            print(f"  Loading: {os.path.basename(reward_path)}")
            with open(reward_path, "r", encoding="utf-8") as f:
                reward_samples.append(f.read())

        # Run training (without retry mechanism since we're resuming)
        # Note: We use run_evaluation_with_retries but pass empty conversation history
        # since we don't need LLM for retries (code already exists)
        task_prompt = self.task_info.to_prompt()

        evaluation_results = self.training_orchestrator.run_evaluation_with_retries(
            reward_samples,
            self.max_training_steps,
            iteration,
            self.llm,
            self._save_reward_sample_on_retry,
            task_prompt,
            None,  # No conversation history for resume
            self.conversations_dir,
        )

        # Generate feedback for next iteration
        (
            best_result,
            best_reward_code,
            feedback_prompt,
        ) = self.feedback_analyzer.prepare_feedback(
            evaluation_results, reward_samples, self.task_info
        )

        # Generate feedback using LLM
        feedback_result = self.llm.generate(feedback_prompt)
        feedback = feedback_result.content

        # Save feedback conversation
        save_llm_result_conversation(
            conversations_dir=self.conversations_dir,
            iteration=iteration,
            run_id=best_result["run_id"],
            attempt=1,
            conv_type="feedback",
            llm_result=feedback_result,
        )

        print(f"\nResume training completed for iteration {iteration}")
        print(f"Best result: sample {best_result['run_id']}")
        print(f"Best success rate: {best_result.get('success_rate', 0.0):.2%}")

    def _save_initial_reward_samples(self, reward_samples: List[str], iteration: int):
        """Save all reward samples immediately after generation for debugging"""
        rewards_dir = self.config["rewards_dir"]
        os.makedirs(rewards_dir, exist_ok=True)

        print(f"Saving {len(reward_samples)} initial reward samples to {rewards_dir}")

        for i, reward_code in enumerate(reward_samples):
            # Initial attempt is always 1
            reward_filename = self.training_orchestrator.get_reward_filename(
                iteration, i, 1
            )
            reward_path = os.path.join(rewards_dir, reward_filename)

            with open(reward_path, "w", encoding="utf-8") as f:
                f.write(reward_code)

            print(f"  Saved: {reward_filename}")

    def _save_reward_sample_on_retry(
        self, reward_code: str, iteration: int, run_id: int, attempt: int
    ):
        """Save a new reward sample during a retry"""
        rewards_dir = self.config["rewards_dir"]
        reward_filename = self.training_orchestrator.get_reward_filename(
            iteration, run_id, attempt
        )
        reward_path = os.path.join(rewards_dir, reward_filename)

        with open(reward_path, "w", encoding="utf-8") as f:
            f.write(reward_code)

        print(f"  Saved retry: {reward_filename}")

    def load_task_info(self, task_name: str) -> EnvironmentInfo:
        """Load task-specific information from config and parse agent file"""
        # Use already loaded config from constructor
        task_config = self.config["tasks"].get(task_name, {})

        # Parse agent file
        assert (
            "agent_file" in task_config
        ), f"Task '{task_name}' must have 'agent_file' specified in config"

        agent_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            task_config["agent_file"],
        )

        assert os.path.exists(
            agent_file_path
        ), f"Agent file not found: {agent_file_path}"

        parser = UnityAgentParser(agent_file_path)
        env_info = parser.parse()

        return env_info

    def run_final_evaluation(self, reward_code: str):
        """Run final evaluation with multiple seeds"""
        print("\n=== Running Final Evaluation ===")

        # Save the final reward code (shared by all seeds)
        reward_path = os.path.join(self.config["rewards_dir"], "final_evaluation.cs")
        with open(reward_path, "w", encoding="utf-8") as f:
            f.write(reward_code)
        print(f"Saved final reward code to: {reward_path}")

        training_metadata = []

        # 1. Run training with special iteration number for custom naming
        for seed in range(5):
            print(f"Running final evaluation with seed {seed}...")
            metadata = self.training_orchestrator.train_with_reward(
                max_training_steps=self.max_training_steps
                * 2,  # Longer training for final eval
                seed=seed,
                run_id=seed,  # Use seed as run_id for clear naming
                iteration=999,  # Special marker for final evaluation
                attempt=1,
            )
            training_metadata.append(metadata)

        # 2. Load actual results from Unity files (gets success_rate)
        final_results = self.training_orchestrator.load_final_results(training_metadata)

        # 3. Extract success rates
        success_rates = []
        for result in final_results:
            success_rates.append(result.get("success_rate", 0.0))

        final_summary = {
            "mean_success_rate": np.mean(success_rates),
            "std_success_rate": np.std(success_rates),
            "individual_results": final_results,
            "best_reward_code": reward_code,
        }

        print(
            f"Final success rate: {final_summary['mean_success_rate']:.2%} ± {final_summary['std_success_rate']:.2%}"
        )

        return final_summary


class FeedbackAnalyzer:
    """Analyzes training results and generates feedback"""

    def __init__(self, llm: RewardLLM):
        self.llm = llm

    def construct_policy_feedback_prompt(
        self, best_result: Dict, best_reward_code: str, task_info: EnvironmentInfo
    ) -> str:
        """Constructs the prompt for policy feedback generation."""
        from training.orchestrator import ResultEnricher

        policy_feedback_input = task_info.to_prompt()
        policy_feedback_input += (
            f"Reward Function Code:\n```csharp\n{best_reward_code}\n```\n\n"
        )

        # Use ResultEnricher to format performance metrics
        policy_feedback_input += ResultEnricher.format_for_llm(best_result)
        policy_feedback_input += "\n\n"

        # Add policy feedback prompt template
        policy_feedback_input += policy_feedback_prompt

        return policy_feedback_input

    def get_best_result_and_reward_code(
        self, evaluation_results: List[Dict], reward_samples: List[str]
    ) -> Tuple[Dict, str]:
        """Finds the best result and reward code from the evaluation results using ResultEnricher"""
        from training.orchestrator import ResultEnricher

        # Use ResultEnricher to select the best result
        best_idx = ResultEnricher.select_best(evaluation_results)
        best_result = evaluation_results[best_idx]
        best_reward_code = reward_samples[best_idx]

        return best_result, best_reward_code

    def prepare_feedback(
        self,
        evaluation_results: List[Dict],
        reward_samples: List[str],
        task_info: EnvironmentInfo,
    ) -> Tuple[Dict, str, str]:
        """Prepares feedback prompt (no LLM call)"""
        # Find best result from this iteration
        best_result, best_reward_code = self.get_best_result_and_reward_code(
            evaluation_results, reward_samples
        )

        # Construct feedback prompt
        policy_feedback_input = self.construct_policy_feedback_prompt(
            best_result, best_reward_code, task_info
        )

        return best_result, best_reward_code, policy_feedback_input


if __name__ == "__main__":
    config = {
        "task": "T0",
        "model": "gpt-4",
        "iterations": 5,
        "samples": 16,
        "max_training_steps": 100000,
        "unity_build_path": "./Builds/AR4.exe",  # Adjust path as needed
    }

    eureka = EurekaUnity(config)
    eureka.run()
