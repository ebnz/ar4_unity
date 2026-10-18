"""
Simple conversation logging utilities
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional


def save_conversation(
    conversations_dir: str,
    iteration: int,
    run_id: int,
    attempt: int,
    conv_type: str,
    messages: List[Dict],
    generated_content: str = None,
    error_context: str = None,
    metadata: dict = None,
) -> str:
    """
    Save a conversation to a JSON file

    Args:
        conversations_dir: Directory to save conversations
        iteration: Eureka iteration number
        run_id: Sample/run ID within iteration
        attempt: Attempt number (for retries)
        conv_type: Type of conversation ('initial', 'retry', 'feedback')
        messages: List of conversation messages
        generated_content: Generated content (code or feedback text)
        error_context: Error context for retry conversations
        metadata: Optional metadata dict

    Returns:
        Path to saved conversation file
    """
    # Ensure directory exists
    os.makedirs(conversations_dir, exist_ok=True)

    # Generate filename
    filename = (
        f"conv_iter{iteration:02d}_run{run_id:03d}_attempt{attempt}_{conv_type}.json"
    )
    filepath = os.path.join(conversations_dir, filename)

    # Create conversation data
    conversation_data = {
        "iteration": iteration,
        "run_id": run_id,
        "attempt": attempt,
        "conversation_type": conv_type,
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "generated_content": generated_content,
        "error_context": error_context,
        "metadata": metadata or {},
    }

    # Save to JSON file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)

    return filepath


def save_llm_result_conversation(
    conversations_dir: str,
    iteration: int,
    run_id: int,
    attempt: int,
    conv_type: str,
    llm_result,
    error_context: str = None,
) -> str:
    """
    Convenience function to save an LLMResult as a conversation

    Args:
        conversations_dir: Directory to save conversations
        iteration: Eureka iteration number
        run_id: Sample/run ID within iteration
        attempt: Attempt number
        conv_type: Type of conversation
        llm_result: LLMResult object from RewardLLM
        error_context: Error context for retry conversations

    Returns:
        Path to saved conversation file
    """
    # Convert LLMMetadata to dict for JSON serialization
    metadata_dict = {
        "model": llm_result.metadata.model,
        "temperature": llm_result.metadata.temperature,
        "prompt_tokens": llm_result.metadata.prompt_tokens,
        "completion_tokens": llm_result.metadata.completion_tokens,
        "total_tokens": llm_result.metadata.total_tokens,
        "latency_ms": llm_result.metadata.latency_ms,
        "timestamp": llm_result.metadata.timestamp,
    }

    return save_conversation(
        conversations_dir=conversations_dir,
        iteration=iteration,
        run_id=run_id,
        attempt=attempt,
        conv_type=conv_type,
        messages=llm_result.messages,
        generated_content=llm_result.content,
        error_context=error_context,
        metadata=metadata_dict,
    )


def extract_code_from_llm_result(llm_result) -> str:
    """
    Extract C# code from an LLMResult containing code generation

    Args:
        llm_result: LLMResult object

    Returns:
        Extracted C# code
    """
    # Import the extract function from RewardLLM
    from llm.reward_llm import RewardLLM

    # Call the static method directly without creating an instance
    return RewardLLM.extract_csharp_code(llm_result.content)
