"""
Hugging Face Serverless Inference API Reasoner.

Sends OCR-extracted text (no images) to a Gemma model hosted on HF infrastructure.
This avoids the 5MB payload limit that makes image-based requests impossible.

Why text-only:
  - HF free-tier Serverless API has a strict ~5MB request body limit
  - Base64-encoding even 1 document image exceeds this
  - Instead, we rely on high-quality OCR text from PaddleOCR and send that

Default model: google/gemma-3-4b-it
  - Reliably served on HF free tier (4B params fits comfortably)
  - google/gemma-3-27b-it frequently returns 502/503 (too large to keep warm)

Requires: HF_TOKEN in .env file (get one at https://huggingface.co/settings/tokens)
"""

import os
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class HuggingFaceAPIReasoner:
    """
    Text-only HF API reasoner via huggingface_hub InferenceClient.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.model_id = config.get("model_id", "google/gemma-2-9b-it")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_output_tokens", 256)

        self.api_key = os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "HF_TOKEN not found. "
                "Get a token at https://huggingface.co/settings/tokens "
                "and add it to your .env file."
            )

        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface_hub is not installed. "
                "Run: pip install huggingface-hub>=0.24.0"
            )

        self._client = InferenceClient(model=self.model_id, token=self.api_key)
        logger.info(f"HuggingFaceAPIReasoner configured (model: {self.model_id}, text-only)")

    def answer(
        self,
        pages: List,
        question: str,
        parsed_context: str,
        domain: str,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        """
        Answer a question using only OCR text (no images sent to API).

        Args:
            pages: List of PIL.Image objects (unused — kept for interface compat).
            question: Question string.
            parsed_context: OCR text from the parser.
            domain: Document domain.

        Returns:
            Dict with "answer" and "full_answer".
        """
        from src.reasoner.prompt_builder import build_prompt
        from src.reasoner.answer_formatter import extract_and_format_answer

        # HF Free Tier Serverless APIs often have strict max input token limits (e.g. 4k-8k tokens)
        # 15,000 chars is roughly 3,750 tokens (4 chars/token avg), which safely avoids Bad Request errors
        MAX_CHARS = 15000
        if len(parsed_context) > MAX_CHARS:
            logger.warning(f"OCR text truncated from {len(parsed_context)} to {MAX_CHARS} chars to fit API limits")
            parsed_context = parsed_context[:MAX_CHARS] + "\n\n[TEXT TRUNCATED DUE TO LENGTH LIMITS]"

        prompt_text = build_prompt(
            question=question,
            parsed_context=parsed_context,
            domain=domain,
        )

        # Text-only message (no images — avoids payload limit)
        messages = [{"role": "user", "content": prompt_text}]

        for attempt in range(3):
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    max_tokens=max_new_tokens or self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else 0.001,
                )
                full_output = response.choices[0].message.content.strip()
                break
            except Exception as e:
                error_str = str(e).lower()
                if "413" in error_str or "payload" in error_str:
                    logger.error(f"Payload too large even in text-only mode: {e}")
                    return {"answer": "Unknown", "full_answer": f"[PAYLOAD ERROR: {e}]"}
                elif "429" in error_str or "rate" in error_str:
                    wait = 15 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Sleeping {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                elif "502" in error_str or "503" in error_str or "bad gateway" in error_str:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"Server error. Sleeping {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                elif "402" in error_str or "payment" in error_str or "prepaid" in error_str:
                    # Paid provider / account exhausted — attempt local Gemma fallback if available
                    logger.error(f"Payment/Quota error contacting HF inference provider: {e}")
                    try:
                        from src.reasoner.gemma_reasoner import GemmaReasoner
                        logger.info("Attempting local GemmaReasoner fallback due to HF payment error")
                        gemma = GemmaReasoner({})
                        return gemma.answer(
                            pages=pages,
                            question=question,
                            parsed_context=parsed_context,
                            domain=domain,
                            max_new_tokens=max_new_tokens,
                        )
                    except Exception as ge:
                        logger.error(f"Local Gemma fallback failed: {ge}")
                        return {"answer": "Unknown", "full_answer": f"[API ERROR: {e}] [LOCAL FALLBACK FAILED: {ge}]"}
                else:
                    logger.error(f"HF API call failed: {e}")
                    return {"answer": "Unknown", "full_answer": f"[API ERROR: {e}]"}
        else:
            return {"answer": "Unknown", "full_answer": "[API ERROR: All retries exhausted]"}

        formatted_answer = extract_and_format_answer(full_output)
        return {
            "answer": formatted_answer,
            "full_answer": full_output,
        }

    def answer_batch(self, qa_batch: List[dict]) -> List[dict]:
        return [self.answer(**qa) for qa in qa_batch]
