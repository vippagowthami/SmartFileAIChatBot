import json
import time
import uuid
from pathlib import Path

import chromadb


class VectorDatabase:
    """Persistent local document store backed by ChromaDB."""

    def __init__(self, db_path: str = "../data/chroma_db", collection_name: str = "documents"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._migrate_legacy_store()

    # ------------------------------------------------------------------ #
    # Migration from old JSON store
    # ------------------------------------------------------------------ #
    def _legacy_store_path(self) -> Path:
        return self.db_path / "documents.json"

    def _migrate_legacy_store(self) -> None:
        legacy_path = self._legacy_store_path()
        if not legacy_path.exists() or self.collection.count() > 0:
            return
        try:
            legacy_documents = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            return

        documents, embeddings, metadatas, ids = [], [], [], []
        for item in legacy_documents:
            text = item.get("text", "")
            embedding = item.get("embedding", [])
            if not text or not embedding:
                continue
            documents.append(text)
            embeddings.append(embedding)
            metadatas.append(item.get("metadata", {}) or {})
            ids.append(item.get("id") or str(uuid.uuid4()))

        if documents:
            self.collection.add(ids=ids, documents=documents,
                                embeddings=embeddings, metadatas=metadatas)
        try:
            legacy_path.unlink()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> float:
        start = time.time()
        if not documents:
            return 0.0
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return time.time() - start

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query_embedding: list[float], num_results: int = 5, where: dict | None = None) -> dict:
        start = time.time()
        count = self.collection.count()
        if count == 0:
            return {"results": [], "retrieval_time": time.time() - start}

        n = min(num_results, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        formatted = [
            {
                "text": doc,
                "metadata": meta or {},
                "distance": float(dist) if dist is not None else 1.0,
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        return {"results": formatted, "retrieval_time": time.time() - start}

    # ------------------------------------------------------------------ #
    # Live stats (always read from ChromaDB — never stale)
    # ------------------------------------------------------------------ #
    def document_count(self) -> int:
        """Total number of chunks stored."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_collection_stats(self) -> dict:
        """Returns live accurate statistics directly from ChromaDB."""
        try:
            total_chunks = self.collection.count()
            # Count unique source files
            records = self.collection.get(include=["metadatas"])
            metas = records.get("metadatas", []) or []
            unique_files: set[str] = set()
            for m in metas:
                if not m:
                    continue
                name = m.get("source_name") or m.get("source", "")
                if name:
                    unique_files.add(str(name))
            indexed_files = len(unique_files)
        except Exception:
            total_chunks = 0
            indexed_files = 0

        return {
            "total_documents": indexed_files,
            "indexed_files": indexed_files,
            "total_chunks": total_chunks,
            "storage_path": str(self.db_path),
            "backend": "chromadb",
            "collection_name": self.collection_name,
        }

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def list_indexed_file_names(self) -> list[str]:
        """Return unique source file names from stored metadata."""
        try:
            records = self.collection.get(include=["metadatas"])
            metas = records.get("metadatas", []) or []
            names: set[str] = set()
            for m in metas:
                if not m:
                    continue
                name = m.get("source_name")
                if not name:
                    source = m.get("source", "")
                    name = str(source).replace("\\", "/").split("/")[-1] if source else ""
                if name:
                    names.add(str(name))
            return sorted(names)
        except Exception:
            return []

    def clear_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_database(self) -> None:
        self.clear_collection()