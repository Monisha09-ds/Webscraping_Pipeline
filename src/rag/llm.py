# src/rag/llm.py
#
# Unified LLM interface.
#   provider="groq"  -> open-source models served by Groq (default)
#   provider="local" -> a HuggingFace causal LM from a local folder (offline)
#
# Failures raise LLMError rather than returning "". An empty string would be
# indistinguishable from a genuine "I don't know", which makes a dead API key
# look like a retrieval miss.

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path
from typing import Optional

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LOCAL_LLM_PATH,
)

logger = logging.getLogger(__name__)

# Substrings that mark an error as worth retrying.
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "rate limit", "overloaded")


class LLMError(RuntimeError):
    """Raised when generation fails for an operational reason."""


class LLMWrapper:
    """
    Usage:
        llm = LLMWrapper(provider="groq", model_name="llama-3.3-70b-versatile")
        text = llm.generate(prompt, max_new_tokens=512)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
        model_dir: Path | str | None = None,
        mode: Optional[str] = None,  # legacy alias for `provider`
    ):
        resolved = (provider or mode or LLM_PROVIDER or "groq").strip().lower()
        # Older callers/UIs said "api" when they meant "the hosted provider".
        if resolved == "api":
            resolved = "groq"
        if resolved not in {"groq", "local"}:
            raise ValueError(f"provider must be 'groq' or 'local', got {resolved!r}")

        self.provider = resolved
        self.mode = resolved  # backwards-compatible attribute
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

        self.client = None
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        if self.provider == "groq":
            self.model_name = (model_name or GROQ_MODEL).strip()
            self._init_groq(api_key)
        else:
            self.model_name = model_name or str(model_dir or LOCAL_LLM_PATH)
            self._init_local(model_dir or LOCAL_LLM_PATH)

    # ----------------------------
    # Initialization
    # ----------------------------
    def _init_groq(self, api_key: Optional[str]) -> None:
        from groq import Groq

        key = (api_key or GROQ_API_KEY or "").strip()
        if not key:
            raise LLMError(
                "Missing GROQ_API_KEY. Set it in .env, pass it in the UI, or export it "
                "in the environment. Get a free key at https://console.groq.com/keys"
            )
        self.client = Groq(api_key=key)
        logger.info("[LLM] Groq ready, model=%s", self.model_name)

    def _init_local(self, model_dir: Path | str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = Path(model_dir).resolve()
        if not model_dir.exists():
            raise LLMError(
                f"Local model folder not found: {model_dir}. "
                "Either download a model there or use provider='groq'."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[LLM] Loading local model %s on %s", model_dir, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
            )
            .to(self.device)
            .eval()
        )
        logger.info("[LLM] Local model loaded")

    # ----------------------------
    # Generation
    # ----------------------------
    def generate(self, prompt: str, max_new_tokens: int = LLM_MAX_TOKENS) -> str:
        if not prompt:
            return ""
        if self.provider == "groq":
            return self._groq_generate(prompt, max_new_tokens)
        return self._local_generate(prompt, max_new_tokens)

    def _groq_generate(self, prompt: str, max_new_tokens: int) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_new_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - re-raised as LLMError below
                last_error = e
                msg = str(e).lower()
                if attempt < self.max_retries and any(m in msg for m in _TRANSIENT_MARKERS):
                    delay = self.retry_backoff_sec * attempt
                    logger.warning(
                        "[LLM] Transient Groq error (attempt %d/%d): %s -> retrying in %.1fs",
                        attempt, self.max_retries, e, delay,
                    )
                    time.sleep(delay)
                    continue
                break

        raise LLMError(f"Groq generation failed ({self.model_name}): {last_error}") from last_error

    def _local_generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.temperature > 0,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            raise LLMError(f"Local generation failed: {e}") from e

    # ----------------------------
    # Cleanup
    # ----------------------------
    def cleanup(self) -> None:
        logger.info("[LLM] Cleanup")
        if self.provider == "local":
            self.model = None
            self.tokenizer = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
