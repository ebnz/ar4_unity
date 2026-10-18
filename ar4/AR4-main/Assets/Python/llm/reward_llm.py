"""
RewardLLM: LLM client for generating reward functions
Supports both Azure OpenAI and standard OpenAI endpoints
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import logging
from datetime import datetime
from prompts.system_prompt import system_prompt as system_prompt_reward_function
from utils.extract_env_info import EnvironmentInfo


class LLMMetadata:
    """Simple metadata for LLM calls"""

    def __init__(
        self,
        model: str,
        temperature: float,
        prompt_tokens: int = None,
        completion_tokens: int = None,
        latency_ms: float = None,
    ):
        self.model = model
        self.temperature = temperature
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        self.latency_ms = latency_ms
        self.timestamp = datetime.now().isoformat()


class LLMResult:
    """Generic result from LLM generation"""

    def __init__(self, content: str, messages: List[Dict], metadata: LLMMetadata):
        self.content = content  # Raw LLM response
        self.messages = messages  # Full conversation including response
        self.metadata = metadata  # Simple metadata


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardLLM:
    """LLM client for generating C# reward functions for Unity environments"""

    def __init__(self, config: Dict):
        """
        Initialize the LLM client

        Args:
            config: Configuration dict with keys:
                - provider: 'azure' or 'openai'
                - model: Model name (e.g., 'gpt-4', 'o4-mini')
                - temperature: Sampling temperature
                - max_completion_tokens: Maximum tokens for completion
        """
        load_dotenv()

        self.config = config
        self.provider = config.get("provider", "azure")
        self.model = config.get("model", "o4-mini")
        self.temperature = config.get("temperature", 1.0)

        # o1 models need much higher token limits due to internal reasoning
        # They use many tokens for "thinking" before generating the visible output
        default_max_tokens = (
            16000 if any(x in self.model.lower() for x in ["o1-", "o4-"]) else 2000
        )
        self.max_completion_tokens = config.get(
            "max_completion_tokens", default_max_tokens
        )

        logger.info(
            f"Initialized RewardLLM: model={self.model}, max_completion_tokens={self.max_completion_tokens}"
        )

        # Initialize client based on provider
        if self.provider == "azure":
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_ENDPOINT_URL"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2025-01-01-preview",
            )
            self.deployment = self.model
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.deployment = self.model

    def generate(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        """
        Generate a single response (works for both code and feedback)

        Args:
            prompt: The prompt to send to the LLM
            conversation_history: Previous messages for in-context learning
            temperature: Override default temperature

        Returns:
            LLMResult with content, messages, and metadata
        """
        temperature = temperature or self.temperature
        start_time = time.time()

        # Check if this is an o1 model (o1-mini, o1-preview, o4-mini, etc.)
        is_o1_model = any(x in self.model.lower() for x in ["o1-", "o4-"])

        # Build messages
        if conversation_history:
            # Use existing conversation history
            messages = list(conversation_history)
            # Add new prompt
            messages.append({"role": "user", "content": prompt})
        else:
            # First time - build initial messages
            if is_o1_model:
                # o1 models don't support system messages - embed system prompt in user message
                messages = [
                    {
                        "role": "user",
                        "content": f"{system_prompt_reward_function}\n\n{prompt}",
                    }
                ]
            else:
                # Standard models use system message
                messages = [
                    {"role": "system", "content": system_prompt_reward_function},
                    {"role": "user", "content": prompt},
                ]

        try:
            # Build API call parameters based on model type
            api_params = {
                "model": self.deployment,
                "messages": messages,
                "max_completion_tokens": self.max_completion_tokens,
                "stream": False,
            }

            # o1 models don't support temperature or stop parameters
            if not is_o1_model:
                api_params["temperature"] = temperature
                api_params["stop"] = None

            # Call LLM
            response = self.client.chat.completions.create(**api_params)

            # Extract response content
            content = response.choices[0].message.content

            # Add assistant response to messages
            full_messages = messages + [{"role": "assistant", "content": content}]

            # Create metadata
            end_time = time.time()
            usage = getattr(response, "usage", None)
            metadata = LLMMetadata(
                model=self.deployment,
                temperature=temperature,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                latency_ms=(end_time - start_time) * 1000,
            )

            return LLMResult(content=content, messages=full_messages, metadata=metadata)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_batch(
        self,
        prompt: str,
        n_samples: int,
        conversation_history: Optional[List[Dict]] = None,
        temperature_range: Tuple[float, float] = None,
        max_workers: int = 4,
    ) -> List[LLMResult]:
        """
        Generate multiple samples in parallel with temperature variation

        Args:
            prompt: The prompt to send to the LLM
            n_samples: Number of samples to generate
            conversation_history: Previous messages
            temperature_range: Min and max temperature for variation (defaults to self.temperature)
            max_workers: Maximum number of parallel threads

        Returns:
            List of LLMResult objects
        """
        # Set default temperature range
        if temperature_range is None:
            temperature_range = (self.temperature, self.temperature)

        # Calculate temperature steps
        min_temp, max_temp = temperature_range
        if n_samples == 1:
            temps = [self.temperature]
        else:
            temps = [
                min_temp + (max_temp - min_temp) * i / (n_samples - 1)
                for i in range(n_samples)
            ]

        logger.info(
            f"Generating {n_samples} samples in parallel with {max_workers} workers"
        )

        # Generate samples in parallel
        results = [None] * n_samples  # Pre-allocate to maintain order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, temp in enumerate(temps):
                future = executor.submit(
                    self.generate, prompt, conversation_history, temp
                )
                future_to_index[future] = i

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Failed to generate sample {index + 1}: {e}")
                    raise

        return results

    @staticmethod
    def extract_csharp_code(response: str) -> str:
        """
        Extract C# code from LLM response

        Args:
            response: Full LLM response

        Returns:
            Extracted C# code (just the function body)
        """
        # Try to find code blocks
        code_block_pattern = r"```(?:csharp|c#|cs)?\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            code = matches[0].strip()
        else:
            # If no code block, assume the whole response is code
            code = response.strip()

        # Remove function signature if present
        # Look for common patterns like "public float ComputeReward" or "float ComputeReward"
        signature_patterns = [
            r"public\s+float\s+ComputeReward\s*\([^)]*\)\s*{(.*)}",
            r"float\s+ComputeReward\s*\([^)]*\)\s*{(.*)}",
            r"public\s+.*\s+ComputeReward\s*\([^)]*\)\s*{(.*)}",
        ]

        for pattern in signature_patterns:
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1).strip()
                break

        # Remove any leading/trailing braces
        code = code.strip()
        if code.startswith("{") and code.endswith("}"):
            code = code[1:-1].strip()

        return code
