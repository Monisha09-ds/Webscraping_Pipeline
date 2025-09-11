####-------------####
# rag/llm.py
# rag/llm.py
from __future__ import annotations
from pathlib import Path
import logging
import os
import gc
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Google Gemini (google-genai)
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve()
DEFAULT_LOCAL_MODEL_DIR = ROOT.parents[2] / "models" / "gemma-3-4b-it"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # optional legacy env; we prefer GOOGLE_API_KEY


class LLMWrapper:
    """
    Unified LLM interface:
      - mode='local' -> loads Gemma-3-4b-it from a local folder
      - mode='api'   -> calls Gemini via google-genai

    Usage:
        llm = LLMWrapper(mode="api", model_name="gemini-1.5-flash")
        text = llm.generate(prompt, max_new_tokens=256)
    """

    def __init__(
        self,
        mode: str = os.getenv("LLM_MODE", "api"),
        model_dir: Path | str = DEFAULT_LOCAL_MODEL_DIR,
        model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash"),
        temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
    ):
        if mode not in {"local", "api"}:
            raise ValueError("mode must be 'local' or 'api'")
        self.mode = mode
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.client = None

        if self.mode == "local":
            self._init_local(model_dir)
        else:
            self._init_api()

    # ----------------------------
    # Initialization helpers
    # ----------------------------
    def _init_local(self, model_dir: Path | str):
        model_dir = Path(model_dir).resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"Local model folder not found: {model_dir}")
        if model_dir.name != "gemma-3-4b-it":
            raise ValueError(f"Expected folder 'gemma-3-4b-it', got '{model_dir.name}'")

        # # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"  # Force CPU usage for LLM
        self.device = "cuda"
        logger.info(f"[LLM] Local mode on device: {self.device}")
        logger.info(f"[LLM] Loading local model: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
        )
        # Ensure pad token exists to avoid warnings/errors in generate()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 

        # dtype = torch.float16 if self.device == "cuda" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device).eval()
        logger.info("[LLM] Local model loaded ✅")

    def _init_api(self):
        # Prefer GOOGLE_API_KEY; warn if both are set
        if GOOGLE_API_KEY and GEMINI_API_KEY:
            logger.warning("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
        api_key = GOOGLE_API_KEY or GEMINI_API_KEY
        if not api_key:
            raise EnvironmentError(
                "Missing GOOGLE_API_KEY (or GEMINI_API_KEY). Set it in your environment or .env for API mode."
            )
        # Some client versions auto-read env; we pass explicitly for clarity
        self.client = genai.Client(api_key=api_key)
        logger.info(f"[LLM] API mode with model: {self.model_name} ✅")

    # ----------------------------
    # Public generate
    # ----------------------------
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not prompt:
            return ""

        if self.mode == "local":
            return self._local_generate(prompt, max_new_tokens)
        else:
            return self._api_generate(prompt, max_new_tokens)

    # ----------------------------
    # Local path
    # ----------------------------
    def _local_generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        out_ids = outputs[0]
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = out_ids[input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # # ----------------------------
    # # API path (version-agnostic)
    # # ----------------------------
    # def _api_generate(self, prompt: str, max_new_tokens: int) -> str:
    #     """
    #     Works across google-genai versions:
    #       - new: client.models.generate_content(...)
    #       - old: client.generate_content(...)device 
    #     Retries on transient 429/5xx/timeout errors.
    #     """
    #     attempt = 0
    #     while True:
    #         attempt += 1
    #         try:
    #             if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
    #                 # NEWER clients
    #                 resp = self.client.models.generate_content(
    #                     model=self.model_name,
    #                     contents=prompt,
    #                     generation_config={
    #                         "max_output_tokens": max_new_tokens,
    #                         "temperature": self.temperature,
    #                     },
    #                 )
    #             else:
    #                 # OLDER clients
    #                 resp = self.client.generate_content(
    #                     model=self.model_name,
    #                     contents=prompt,
    #                     generation_config={
    #                         "max_output_tokens": max_new_tokens,
    #                         "temperature": self.temperature,
    #                     },
    #                 )
    #             text = (getattr(resp, "text", "") or "").strip()
    #             return text
    #         except Exception as e:
    #             msg = str(e)
    #             transient = any(code in msg for code in ("429", "502", "503", "504", "timeout"))
    #             if attempt < self.max_retries and transient:
    #                 sleep_for = self.retry_backoff_sec * attempt
    #                 logger.warning(
    #                     f"[LLM] API transient error (attempt {attempt}): {msg} -> retrying in {sleep_for:.1f}s"
    #                 )
    #                 time.sleep(sleep_for)
    #                 continue
    #             logger.exception("[LLM] API generation failed: %s", e)
    #             return ""

    def _api_generate(self, prompt: str, max_new_tokens: int) -> str:
        """
        Works across google-genai versions:
        - newer: client.models.generate_content(..., generation_config=...)
        - older: client.models.generate_content(...)  # no generation_config
        - much older: client.generate_content(...)
        Retries on 429/5xx/timeouts.
        """
        attempt = 0

        # choose the callable once
        if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
            def _call(**kw):  # newer namespace
                return self.client.models.generate_content(**kw)
        else:
            def _call(**kw):  # older flat client
                return self.client.generate_content(**kw)

        while True:
            attempt += 1
            try:
                payload = {
                    "model": self.model_name,
                    "contents": prompt,
                }
                gen_cfg = {
                    "max_output_tokens": max_new_tokens,
                    "temperature": self.temperature,
                }

                # Try with generation_config (newer clients)
                try:
                    resp = _call(**payload, generation_config=gen_cfg)
                except TypeError:
                    # Fallback for older clients (no generation_config kwarg)
                    resp = _call(**payload)

                text = (getattr(resp, "text", "") or "").strip()
                return text
            except Exception as e:
                msg = str(e)
                transient = any(code in msg for code in ("429", "502", "503", "504", "timeout"))
                if attempt < self.max_retries and transient:
                    sleep_for = self.retry_backoff_sec * attempt
                    logging.warning(
                        f"[LLM] API transient error (attempt {attempt}): {msg} -> retrying in {sleep_for:.1f}s"
                    )
                    time.sleep(sleep_for)
                    continue
                logging.exception("[LLM] API generation failed: %s", e)
                return ""

    # ----------------------------
    # Cleanup
    # ----------------------------
    def cleanup(self):
        logger.info("[LLM] Cleanup...")
        if self.mode == "local":
            for attr in ("model", "tokenizer"):
                if getattr(self, attr, None) is not None:
                    delattr(self, attr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
