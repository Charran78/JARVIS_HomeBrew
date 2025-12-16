try:
    import chromadb
except Exception:
    chromadb = None
from typing import List, Dict, Any, Optional
import logging
from models.data_models import SmartChunk

logger = logging.getLogger("ChromaDBManager")

class ChromaDBManager:
    def __init__(self, path: str = "./chroma_db"):
        self.client = None
        if chromadb is None:
            self.collection = None
            self._mem = []
            return
        try:
            if hasattr(chromadb, "PersistentClient"):
                self.client = chromadb.PersistentClient(path=path)
            else:
                from chromadb.config import Settings
                self.client = chromadb.Client(
                    Settings(
                        persist_directory=path,
                        chroma_db_impl="duckdb+parquet",
                        anonymized_telemetry=False
                    )
                )
            self.collection = self.client.get_or_create_collection(name="documents")
        except Exception as e:
            self.collection = None
            self._mem = []
            logger.error(f"Chroma init error: {e}")
        
    def add_embeddings(self, chunks: List[SmartChunk]):
        if self.collection is not None:
            embeddings = []
            metadatas = []
            documents = []
            ids = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
                    meta = chunk.to_dict()
                    if "embedding" in meta:
                        meta.pop("embedding", None)
                    metadatas.append({k: v for k, v in meta.items() if v is None or isinstance(v, (str, int, float, bool))})
                    documents.append(chunk.text)
                    ids.append(chunk.chunk_id)
            if embeddings:
                try:
                    self.collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        ids=ids
                    )
                    logger.info(f"Added {len(embeddings)} chunks to ChromaDB")
                except Exception as e:
                    logger.error(f"Chroma add error: {e}")
        else:
            for chunk in chunks:
                if chunk.embedding is not None:
                    self._mem.append(chunk)
        
    def query(self, query_embedding: List[float], n_results: int = 10, where: Optional[Dict] = None) -> Dict[str, Any]:
        if self.collection is not None:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
        docs = [c.text for c in self._mem][-n_results:]
        metas = [c.to_dict() for c in self._mem][-n_results:]
        ids = [c.chunk_id for c in self._mem][-n_results:]
        return {"documents": [docs], "metadatas": [metas], "ids": [ids]}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        if self.collection is not None:
            try:
                count = self.collection.count()
            except Exception:
                count = None
            return {"count": count, "metadata": getattr(self.collection, "metadata", {})}
        return {"count": len(getattr(self, "_mem", [])), "metadata": {}}
    
    def delete_document(self, document_id: str):
        if self.collection is not None:
            self.collection.delete(where={"document_id": document_id})
        else:
            self._mem = [c for c in getattr(self, "_mem", []) if c.document_id != document_id]
