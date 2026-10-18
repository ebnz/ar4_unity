from run_eureka import load_config
import os
import sys
import numpy as np
import json

from training.orchestrator import TrainingOrchestrator
from eureka_main import FeedbackAnalyzer, EurekaUnity
from llm.reward_llm import RewardLLM


def load_reward_codes(
    rewards_dir: str, run_ids: list[int], iteration: int
) -> list[str]:
    """Load reward codes from files."""
    reward_codes = []
    for i in run_ids:
        reward_filename = f"reward_iter{iteration:02d}_sample{i:03d}.cs"
        reward_path = os.path.join(rewards_dir, reward_filename)
        try:
            with open(reward_path, "r", encoding="utf-8") as f:
                reward_codes.append(f.read())
        except FileNotFoundError:
            print(f"Warning: Reward file not found: {reward_path}")
            reward_codes.append(None)  # Keep list length consistent
    return [code for code in reward_codes if code is not None]


# --- Test Configuration ---
run_ids = [0, 1, 2, 3]
iteration = 0
CONFIG_PATH = "config/eureka_config_test.yaml"

# --- Load Configuration & Components ---
config = load_config(CONFIG_PATH)

# Initialize LLM and Feedback Analyzer
llm_config = {
    "provider": config.get("llm_provider", "azure"),
    "model": config.get("model", "gpt-4"),
    "temperature": config.get("temperature", 0.0),  # Use 0 for deterministic feedback
    "max_completion_tokens": config.get("max_completion_tokens", 2000),
}
reward_llm = RewardLLM(llm_config)
feedback_analyzer = FeedbackAnalyzer(reward_llm)

# --- Load Test Data ---
print(f"--- Loading data for iteration {iteration}, runs {run_ids} ---")
orchestrator = TrainingOrchestrator(config)
results = orchestrator.load_all_results(run_ids, iteration, timeout=30)

rewards_dir = config.get("rewards_dir", "GeneratedRewards")
reward_codes = load_reward_codes(rewards_dir, run_ids, iteration)

# Instantiate a dummy EurekaUnity to use its task loader
# This is easier than re-implementing the logic here
eureka_dummy = EurekaUnity(config)
task_info = eureka_dummy.load_task_info("T0")

print(f"Loaded {len(results)} results and {len(reward_codes)} reward codes.")

if not results or not reward_codes or len(results) != len(reward_codes):
    print(
        "Error: Mismatch between number of results and reward codes, or no data loaded. Exiting."
    )
    sys.exit(1)


# --- Run Feedback Generation Step-by-Step ---
print("\n\n" + "=" * 80)
print("--- TEST RESULTS ---")
print("=" * 80)

best_result, best_reward_code = feedback_analyzer.get_best_result_and_reward_code(
    results, reward_codes
)

print(f"\nBest Success Rate: {best_result.get('success_rate', 0.0):.2%}")

print("\n--- Best Performing Reward Code ---")
print(best_reward_code)

# --- Step 1: Construct Policy Feedback Prompt ---
print("\n--- Step 1: Constructed Policy Feedback Prompt ---")
policy_prompt = feedback_analyzer.construct_policy_feedback_prompt(
    best_result, best_reward_code, task_info
)
print(policy_prompt)

# --- Step 2: Generate Policy Analysis from LLM ---
print("\n--- Step 2: Generated Policy Analysis from LLM ---")
policy_analysis = feedback_analyzer.generate_policy_feedback(
    results, reward_codes, task_info
)
print(policy_analysis)


print("\n" + "=" * 80)
print("--- TEST COMPLETE ---")
print("=" * 80)
