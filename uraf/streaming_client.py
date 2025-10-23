"""
Streaming LLM Client - Real-time Token-by-Token Evaluation

Extends LLMClient with streaming support for real-time evaluation.
"""

import asyncio
import aiohttp
from typing import AsyncIterator, Dict, Optional, Any, Callable
from loguru import logger
import json


class StreamingLLMClient:
    """LLM client with streaming support."""

    def __init__(
        self,
        model: str = "qwen2.5-7b-instruct-1m",
        api_url: str = "http://localhost:1234/v1/completions",
        **kwargs
    ):
        """
        Initialize streaming client.

        Args:
            model: Model identifier
            api_url: API endpoint URL
            **kwargs: Additional LLM parameters
        """
        self.model = model
        self.api_url = api_url
        self.parameters = kwargs
        logger.info(f"Initialized StreamingLLMClient: {model}")

    async def stream_query(
        self,
        prompt: str,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncIterator[str]:
        """
        Stream LLM response token-by-token.

        Args:
            prompt: Input prompt
            callback: Optional callback for each token

        Yields:
            Individual tokens or chunks
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **self.parameters
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Streaming failed: {response.status} - {error_text}")
                        yield f"Error: {response.status}"
                        return

                    # Stream response
                    async for line in response.content:
                        if not line:
                            continue

                        line = line.decode('utf-8').strip()

                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix

                            if data_str == '[DONE]':
                                break

                            try:
                                chunk = json.loads(data_str)
                                token = chunk.get('choices', [{}])[0].get('text', '')

                                if token:
                                    if callback:
                                        callback(token)
                                    yield token

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

    async def stream_with_evaluation(
        self,
        prompt: str,
        evaluator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Stream response with real-time partial evaluation.

        Args:
            prompt: Input prompt
            evaluator: Optional evaluator for partial responses

        Returns:
            Complete response with evaluation history
        """
        full_response = ""
        partial_evaluations = []
        chunk_count = 0

        async for token in self.stream_query(prompt):
            full_response += token
            chunk_count += 1

            # Evaluate every N tokens
            if evaluator and chunk_count % 50 == 0:
                try:
                    partial_eval = await evaluator.evaluate_response(full_response)
                    partial_evaluations.append({
                        "tokens_so_far": chunk_count,
                        "evaluation": partial_eval
                    })
                except Exception as e:
                    logger.warning(f"Partial evaluation failed: {e}")

        # Final evaluation
        final_eval = None
        if evaluator:
            try:
                final_eval = await evaluator.evaluate_response(full_response)
            except Exception as e:
                logger.error(f"Final evaluation failed: {e}")

        return {
            "full_response": full_response,
            "token_count": chunk_count,
            "partial_evaluations": partial_evaluations,
            "final_evaluation": final_eval
        }


class RealTimeEvaluator:
    """Evaluates streaming responses in real-time."""

    def __init__(self):
        """Initialize real-time evaluator."""
        self.evaluation_history = []
        logger.info("Initialized RealTimeEvaluator")

    async def evaluate_partial(self, partial_text: str) -> Dict[str, float]:
        """
        Evaluate partial response.

        Args:
            partial_text: Incomplete response

        Returns:
            Partial evaluation metrics
        """
        # Simple metrics for partial text
        word_count = len(partial_text.split())
        has_structure = '*' in partial_text or '#' in partial_text

        completeness_estimate = min(1.0, word_count / 200.0)
        structure_score = 0.5 if has_structure else 0.0

        return {
            "completeness": completeness_estimate,
            "structure_score": structure_score,
            "word_count": word_count
        }
