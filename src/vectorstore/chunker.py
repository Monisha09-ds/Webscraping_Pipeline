####--------(Updated Chunking Strategy (Heading-aware + Token Overlap)------------####



from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re
import tiktoken
from dataclasses import dataclass

# ========= Config =========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _get_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        class Dummy:
            def encode(self, s): return s.split()
            def decode(self, tokens): return " ".join(tokens)
        return Dummy()

enc = _get_encoder()

@dataclass
class Chunk:
    text: str
    section_title: str
    metadata: Dict

def split_by_headings(markdown_text: str) -> List[Dict]:
    """Split markdown into sections by headings."""
    sections = re.split(r"(?m)^#{1,6}\s", markdown_text)
    titles = re.findall(r"(?m)^#{1,6}\s(.*)", markdown_text)
    chunks = []
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        title = titles[i] if i < len(titles) else "Untitled"
        chunks.append({"title": title, "content": section.strip()})
    return chunks

def tokenize_and_chunk(text: str, section_index: int, section_title: str, base_metadata: Dict, start_chunk_id: int = 0) -> List[Chunk]:
    """Break long sections into token windows with overlap."""
    tokens = enc.encode(text)
    chunks = []
    chunk_index = start_chunk_id
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        window = tokens[i: i + CHUNK_SIZE]
        chunk_text = enc.decode(window) if hasattr(enc, "decode") else " ".join(window)
        # ✅ Use section_index + chunk_index for unique ID
        chunk_id = f"{base_metadata['doc_id']}_{section_index}_{chunk_index}"
        chunks.append(
            Chunk(
                text=chunk_text,
                section_title=section_title,
                metadata={**base_metadata, "chunk_id": chunk_id}
            )
        )
        chunk_index += 1
    return chunks

def chunk_markdown(md_text: str, doc_id: str, source: str) -> List[Chunk]:
    """High-level API: Markdown → structured chunks."""
    sections = split_by_headings(md_text)
    final_chunks = []
    for sec_idx, sec in enumerate(sections):
        base_metadata = {"doc_id": doc_id, "source": source, "section": sec["title"]}
        sec_tokens = enc.encode(sec["content"])
        if len(sec_tokens) <= CHUNK_SIZE:
            # ✅ single chunk, unique ID
            final_chunks.append(
                Chunk(
                    text=sec["content"],
                    section_title=sec["title"],
                    metadata={**base_metadata, "chunk_id": f"{doc_id}_{sec_idx}_0"}
                )
            )
        else:
            final_chunks.extend(tokenize_and_chunk(
                sec["content"],
                section_index=sec_idx,
                section_title=sec["title"],
                base_metadata=base_metadata,
                start_chunk_id=0
            ))
    return final_chunks
