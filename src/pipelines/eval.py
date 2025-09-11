
##--------------------------###
src/rag/eval.py
from pathlib import Path
import logging
import json
import sys
import torch
import gc


# ---- Path bootstrap ------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[3]      # .../website_content_scrapper
SRC_DIR   = PROJ_ROOT / "webscraper_pipeline" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
from vectorstore.store import ChromaStore, build_from_sitecontent
from rag.llm import LocalLLM
from webscraper_pipeline.src.rag.retrieve import retrieve_chunks
from rag.answer import generate_answer


logging.basicConfig(level=logging.INFO)


# ===== Paths =====
SITECONTENT_DIR = Path(__file__).resolve().parents[2] / "sitecontent"
PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_index"
EMBED_MODEL_PATH = Path(__file__).resolve().parents[2] / "models/colnomic-embed-multimodal-3b"

# ===== Load Chroma store =====
store = build_from_sitecontent(SITECONTENT_DIR, PERSIST_DIR, EMBED_MODEL_PATH)
retriever = retrieve_chunks(store, top_k=5)

# ===== Load LLM =====
llm = LocalLLM()  # uses device_map auto and FP16 if GPU available

# ===== Evaluation Dataset =====
# Replace with your actual queries and ground truth answers
EVAL_DATA = [
    {
        "query": "What is the latest policy on exports?",
        "ground_truth": "The export policy was updated in 2025 to include..."
    },
    {
        "query": "List the main steps of site scraping.",
        "ground_truth": "The steps include frontier initialization, BFS traversal, pagination handling..."
    }
]

results = []

for item in EVAL_DATA:
    query = item["query"]
    ground_truth = item["ground_truth"]

    # Retrieve top-k chunks
    chunks = retriever.retrieve(query)

    # Generate LLM answer
    answer = generate_answer(query, chunks)

    # ===== Compute simple metrics =====
    # Faithfulness: fraction of answer present in chunks (simple overlap)
    context_text = " ".join([c["text"] for c in chunks])
    overlap = sum([1 for w in answer.split() if w in context_text.split()])
    faithfulness = overlap / max(len(answer.split()), 1)

    # Context precision: fraction of retrieved context words used in answer
    context_words = context_text.split()
    used_words = [w for w in context_words if w in answer.split()]
    context_precision = len(used_words) / max(len(context_words), 1)

    # Context recall: fraction of answer words found in context
    answer_words = answer.split()
    retrieved_words = [w for w in answer_words if w in context_words]
    context_recall = len(retrieved_words) / max(len(answer_words), 1)

    results.append({
        "query": query,
        "ground_truth": ground_truth,
        "answer": answer,
        "faithfulness": round(faithfulness, 3),
        "context_precision": round(context_precision, 3),
        "context_recall": round(context_recall, 3)
    })

# ===== Save results =====
output_file = Path("rag_eval_results.json")
output_file.write_text(json.dumps(results, indent=2))
logging.info(f"[Eval] Results saved to {output_file}")
