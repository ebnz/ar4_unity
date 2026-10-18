"""
Test script to preview the messages that will be sent to the LLM
without running the full Eureka loop
"""

import os
import sys
import yaml
import json
from typing import Dict, List

from prompts.system_prompt import system_prompt
from utils.extract_env_info import UnityAgentParser, EnvironmentInfo
from eureka_main import FeedbackAnalyzer


def preview_llm_messages(config_path: str = None, task_name: str = "T0"):
    """
    Preview the messages that would be sent to the LLM for a given task

    Args:
        config_path: Path to eureka config file (optional)
        task_name: Task name to preview (default: T0)
    """

    # Load config
    if config_path is None:
        config_path = "config/eureka_config_test.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get task config
    task_config = config["tasks"].get(task_name, {})

    if "agent_file" not in task_config:
        print(f"Error: Task '{task_name}' must have 'agent_file' specified in config")
        return

    # Parse agent file
    agent_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        task_config["agent_file"],
    )

    if not os.path.exists(agent_file_path):
        print(f"Error: Agent file not found: {agent_file_path}")
        return

    parser = UnityAgentParser(agent_file_path)
    env_info = parser.parse()

    print("=" * 80)
    print("ITERATION 1 - Initial Request")
    print("=" * 80)

    # Show system message
    print("\n[SYSTEM MESSAGE]:")
    print("-" * 40)
    print(system_prompt)

    # Show user message for first iteration
    print("\n[USER MESSAGE]:")
    print("-" * 40)
    task_prompt = env_info.to_prompt()
    print(task_prompt)

    # Simulate a reward function response
    print("\n[SIMULATED ASSISTANT RESPONSE]:")
    print("-" * 40)
    sample_reward = """// Distance-based reward
float distanceToGoal = observations["DistanceToGoal"];
float distanceReward = -distanceToGoal;
rewards["distance"] = distanceReward;

// Progress reward
float progress = context.previousDistance - distanceToGoal;
rewards["progress"] = progress * 10.0f;

// Success bonus
if (distanceToGoal < context.goalThreshold)
{
    rewards["success"] = 100.0f;
}

// Small time penalty
rewards["time_penalty"] = -0.01f;"""
    print(sample_reward)

    # Simulate feedback for iteration 2
    print("\n" + "=" * 80)
    print("ITERATION 2 - With Feedback")
    print("=" * 80)

    # Create mock evaluation results for feedback
    mock_results = [
        {
            "success_rate": 0.15,
            "reward_components": {
                "distance": {"mean": -0.5, "min": -1.0, "max": -0.1},
                "progress": {"mean": 0.001, "min": -0.1, "max": 0.1},
                "success": {"mean": 15.0, "min": 0.0, "max": 100.0},
                "time_penalty": {"mean": -0.5, "min": -1.0, "max": -0.01},
            },
        }
    ]

    # Generate feedback
    feedback_analyzer = FeedbackAnalyzer()
    feedback = feedback_analyzer.generate_feedback(
        mock_results,
        [sample_reward],
        {
            "name": task_name,
            "description": env_info.task_description,
            "success_criteria": env_info.success_criteria,
            "observations": env_info.observations,
            "context": env_info.context,
            "action_space": {
                "continuous_actions": env_info.actions.continuous_actions,
                "description": env_info.actions.description,
            },
        },
    )

    print("\n[USER MESSAGE - Feedback]:")
    print("-" * 40)
    print(feedback)

    # Show full message list
    print("\n" + "=" * 80)
    print("FULL MESSAGE LIST (for iteration 2)")
    print("=" * 80)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
        {"role": "assistant", "content": sample_reward},
        {"role": "user", "content": feedback},
    ]

    print("\nTotal messages:", len(messages))
    for i, msg in enumerate(messages):
        print(f"\nMessage {i+1} [{msg['role'].upper()}]:")
        print(f"Length: {len(msg['content'])} characters")
        if len(msg["content"]) > 200:
            print(f"Preview: {msg['content'][:200]}...")
        else:
            print(f"Full content:\n{msg['content']}")

    # Save to file for inspection
    output_file = f"llm_messages_preview_{task_name}.json"
    with open(output_file, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"\n\nFull messages saved to: {output_file}")


def test_all_tasks(config_path: str = None):
    """Test all tasks defined in the config"""
    if config_path is None:
        config_path = "config/eureka_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for task_name in config["tasks"].keys():
        print(f"\n{'#'*80}")
        print(f"# Testing task: {task_name}")
        print(f"{'#'*80}")
        try:
            preview_llm_messages(config_path, task_name)
        except Exception as e:
            print(f"Error testing task {task_name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview LLM messages for Eureka")
    parser.add_argument("--config", type=str, help="Path to eureka config file")
    parser.add_argument("--task", type=str, default="T0", help="Task name to preview")
    parser.add_argument("--all", action="store_true", help="Test all tasks in config")

    args = parser.parse_args()

    if args.all:
        test_all_tasks(args.config)
    else:
        preview_llm_messages(args.config, args.task)
