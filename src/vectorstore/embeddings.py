import logging
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import os 
import sys
import gc
from transformers import __version__ as transformers_version


PROJ_ROOT = Path(__file__).resolve().parents[1]  
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.config import EMBED_MODEL_PATH, USE_LOCAL_EMBEDDINGS


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"[Embeddings] Torch {torch.__version__}, Transformers {transformers_version}")


class LocalTextEmbedder:
    """
    Local CPU/GPU-based text embedder for scraped markdown content.
    Loads HuggingFace model from local folder OR downloads from HF if missing.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = None, torch_dtype=None):
        # Default to a lightweight portable model if not found locally
        DEFAULT_PORTABLE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        
        raw_path = model_path or EMBED_MODEL_PATH
        self.model_path_or_name = raw_path
        
        # Check if it's a local directory
        self.is_local = Path(raw_path).exists() and Path(raw_path).is_dir()
        
        if not self.is_local:
            if USE_LOCAL_EMBEDDINGS:
                raise FileNotFoundError(f"Local embedding model not found at {raw_path} and USE_LOCAL_EMBEDDINGS is True")
            else:
                logging.info(f"[Embeddings] Local path {raw_path} not found. Falling back to downloadable model: {DEFAULT_PORTABLE_MODEL}")
                self.model_path_or_name = DEFAULT_PORTABLE_MODEL

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        # Use float32 on CPU for better compatibility, float16 on GPU
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        
        self.model = None
        self.tokenizer = None

        logging.info(f"[Embeddings] Using model: {self.model_path_or_name} on {self.device}")
        self._load_model()

    def _load_model(self):
        """Load HuggingFace model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path_or_name,
                local_files_only=self.is_local,
                trust_remote_code=True,
            )

            self.model = AutoModel.from_pretrained(
                self.model_path_or_name,
                local_files_only=self.is_local,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
            )
        except Exception as e:
            logging.error(f"[Embeddings] Failed to load model {self.model_path_or_name}: {e}")
            # Final fallback to standard MiniLM if anything fails
            if self.model_path_or_name != "sentence-transformers/all-MiniLM-L6-v2":
                logging.info("[Embeddings] Attempting final fallback to sentence-transformers/all-MiniLM-L6-v2")
                self.model_path_or_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.is_local = False
                self._load_model()
                return
            raise e

        self.model.to(self.device)
        self.model.eval()

        logging.info("[Embeddings] Model loaded and ready.")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 16,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Processes in batches to avoid OOM on GPU.
        Returns embeddings as list of lists (CPU, float32).
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state  # [batch, seq, hidden]
                embeddings = last_hidden.mean(dim=1)     # mean pooling -> [batch, hidden]

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Move to CPU and convert to list
            all_embeddings.extend(embeddings.cpu().tolist())

        logging.info(f"[Embeddings] Generated {len(all_embeddings)} vectors of size {len(all_embeddings[0])}")
        return all_embeddings

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """
        Generate an embedding for a single query string.
        Returns a single embedding vector (CPU, float32).
        """
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state
            embedding = last_hidden.mean(dim=1)

            if normalize:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().squeeze(0).tolist()
    

    def cleanup(self):
        """Release memory to avoid CPU/GPU memory issues."""
        logging.info("[Embeddings] Cleaning up model and tokenizer...")
        for attr in ["model", "tokenizer"]:
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("[Embeddings] Cleanup completed.")
