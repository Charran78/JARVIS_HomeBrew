# models/data_models.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class SmartChunk:
    text: str
    chunk_id: str
    source_document: str
    document_id: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None
    batch_id: Optional[str] = None
    processed_time: Optional[float] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class Document:
    path: str
    document_id: str
    file_type: str
    size: int
    last_modified: float
    processed: bool = False
    chunks: List[SmartChunk] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []

    def to_dict(self):
        return {
            "path": self.path,
            "document_id": self.document_id,
            "file_type": self.file_type,
            "size": self.size,
            "last_modified": self.last_modified,
            "processed": self.processed,
            "chunks_count": len(self.chunks)
        }