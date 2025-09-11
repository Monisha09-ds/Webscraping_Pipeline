# ####---------Local LLM --------####
# from __future__ import annotations
# from pathlib import Path
# import logging
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import gc
# from transformers import BitsAndBytesConfig

# ROOT = Path(__file__).resolve()
# MODEL_DIR = ROOT.parents[2] / "models" / "gemma-3-4b-it"

# # Logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# class LocalLLM:
#     """
#     Local wrapper around Gemma-3-4b-it.
#     Loads model from local directory.
#     Uses GPU if available, otherwise CPU.
#     """

#     def __init__(self, model_dir: Path = MODEL_DIR):
#         self.model_dir = Path(model_dir).resolve()

#         # Strict validation: model folder must exist and be correct
#         if not self.model_dir.exists():
#             raise FileNotFoundError(f"Model folder not found: {self.model_dir}")
#         if self.model_dir.name != "gemma-3-4b-it":
#             raise ValueError(
#                 f"Expected model folder to be 'gemma-3-4b-it', got '{self.model_dir.name}'"
#             )
        
#         self.device = "cpu"
#         logger.info(f"[LLM] Using device: {self.device}")

#         logger.info(f"[LLM] Loading model from local folder: {self.model_dir}")

       
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_dir, local_files_only=True, trust_remote_code=True
#         )

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_dir,
#             local_files_only=True,
#             trust_remote_code=True,
#             torch_dtype=torch.float16 
            
#         )

#         # Ensure model is fully on the chosen device
#         self.model.to(self.device)

#         self.model.eval()
#         logger.info(f"[LLM] Model loaded successfully on {self.device} ✅")

#     def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
#         """Generate text from prompt."""
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 temperature=0.0,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )

#         out_ids = outputs[0]
#         input_len = inputs["input_ids"].shape[-1]
#         generated_ids = out_ids[input_len:]
#         return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    

#     def cleanup(self):
#         """Release memory to avoid CPU/GPU memory issues."""
#         logging.info("[Embeddings] Cleaning up model and tokenizer...")
#         for attr in ["model", "tokenizer"]:
#             if getattr(self, attr, None) is not None:
#                 delattr(self, attr)
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         logging.info("[Embeddings] Cleanup completed.")


# # if __name__ == "__main__":
# #     logger.info("[TEST] Initializing LocalLLM...")
# #     llm = LocalLLM()

# #     # Enhanced prompt integrating Chain of Thought (CoT) and Instruction-Tuning principles:
# #     # - Instruction-Tuning: The model (Gemma-3-4b-it) is already instruction-tuned, so we craft the prompt as a clear, structured instruction to leverage this alignment for better adherence to user intent, reducing ambiguity and improving output quality (e.g., via explicit directives like "be concise" or "focus on key elements").
# #     # - Chain of Thought: Added step-by-step reasoning directive to elicit logical breakdown, enhancing accuracy and depth (empirical proof: CoT improves reasoning benchmarks by 10-40% as per Wei et al., 2022; here, it breaks down quote creation into ideation, refinement, and finalization steps).
# #     # - Deep Thinking: This integration ensures the LLM processes contextually (e.g., for RAG: retrieve chunks → instruct with CoT to reason over them), leading to precise answers; proof via ablation: without CoT, outputs are shallower (e.g., direct quotes vs. reasoned ones); with instruction-tuning leverage, preference rates increase (Ouyang et al., 2022: 85% human preference for tuned responses).
# #     # - For RAG Goal: Adapt this pattern in your pipeline's LLM call: Wrap user query + retrieved chunks in a CoT-instructed prompt, e.g., "Using this context: [chunks]. Follow instructions: Think step by step to answer [query]. Step 1: Summarize context. Step 2: Relate to query. Step 3: Conclude accurately."
# #     test_prompt = """You are an AI expert tasked with generating a short motivational quote about learning AI. Follow these instructions precisely: Be concise, inspirational, and original. Use Chain of Thought reasoning to ensure depth.

# #         Step 1: Identify core themes in learning AI (e.g., persistence, curiosity, innovation).
# #         Step 2: Brainstorm a quote structure: Start with challenge, end with reward.
# #         Step 3: Refine for brevity and impact.
# #         Step 4: Output only the final quote."""
# #     logger.info(f"[TEST] Prompt: {test_prompt}")

# #     response = llm.generate(test_prompt)
# #     print("\n[TEST OUTPUT]\n", response)



####---------LLM Wrapper (local + API) --------####
# from __future__ import annotations
# from pathlib import Path
# import logging
# import os
# import gc
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Gemini (new google-genai client)
# from google import genai

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ROOT = Path(__file__).resolve()
# DEFAULT_LOCAL_MODEL_DIR = ROOT.parents[2] / "models" / "gemma-3-4b-it"

# class LLMWrapper:
#     """
#     Unified LLM interface for:
#       - mode='local'  -> Gemma-3-4b-it from local folder
#       - mode='api'    -> Gemini 1.5 Flash via google-genai client

#     Usage: text = llm.generate(prompt, max_new_tokens=256)
#     """

#     def __init__(
#         self,
#         mode: str = os.getenv("LLM_MODE", "api"),
#         model_dir: Path = DEFAULT_LOCAL_MODEL_DIR,
#         model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash"),
#         temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
#     ):
#         if mode not in ("local", "api"):
#             raise ValueError("mode must be 'local' or 'api'")
#         self.mode = mode
#         self.temperature = temperature

#         self.model = None
#         self.tokenizer = None
#         self.device = "cpu"

#         if self.mode == "local":
#             model_dir = Path(model_dir).resolve()
#             if not model_dir.exists():
#                 raise FileNotFoundError(f"Local model folder not found: {model_dir}")
#             if model_dir.name != "gemma-3-4b-it":
#                 raise ValueError(f"Expected folder 'gemma-3-4b-it', got '{model_dir.name}'")

#             # prefer GPU if available
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             logger.info(f"[LLM] Local mode on device: {self.device}")
#             logger.info(f"[LLM] Loading local model: {model_dir}")

#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_dir, local_files_only=True, trust_remote_code=True
#             )
#             dtype = torch.float16 if self.device == "cuda" else torch.float32
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_dir, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
#             ).to(self.device).eval()
#             logger.info("[LLM] Local model loaded ✅")

#         else:
#             # API mode (Gemini with new google-genai)
#             # NOTE: The client reads GOOGLE_API_KEY or GEMINI_API_KEY from env.
#             # If both are set, SDK prioritizes GOOGLE_API_KEY (you saw that warning).
#             self.client = genai.Client()
#             self.model_name = model_name
#             logger.info(f"[LLM] API mode with model: {self.model_name} ✅")

#     def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
#         if self.mode == "local":
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens,
#                     do_sample=False,
#                     temperature=0.0,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                 )
#             out_ids = outputs[0]
#             input_len = inputs["input_ids"].shape[-1]
#             generated_ids = out_ids[input_len:]
#             return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

#         # API mode: call through client.models.generate_content (no configure / no get)
#         resp = self.client.models.generate_content(
#             model=self.model_name,
#             contents=prompt,
#             generation_config=genai.types.GenerationConfig(
#                 max_output_tokens=max_new_tokens,
#                 temperature=self.temperature,
#             ),
#         )
#         return (resp.text or "").strip()

#     def cleanup(self):
#         logger.info("[LLM] Cleanup...")
#         if self.mode == "local":
#             for attr in ("model", "tokenizer"):
#                 if getattr(self, attr, None) is not None:
#                     delattr(self, attr)
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
