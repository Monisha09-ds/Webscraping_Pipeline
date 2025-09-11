from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class Chunk(BaseModel):
    doc_id: str
    chunk_id: int
    text: str
    source_path: str
    rel_path: str
    url: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)   

class EmbeddingRecord(BaseModel):
    vector: List[float]
    chunk: Chunk

class IndexMeta(BaseModel):
    dim: int
    model_name: str
    normalize: bool = True
