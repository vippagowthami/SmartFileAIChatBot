import time
import uuid
import os
import re
try:
    from .file_processor import FileProcessor
    from .db import VectorDatabase
    from .llm import OllamaLLM
    from .text_vectorizer import tokenize
except Exception:
    from file_processor import FileProcessor
    from db import VectorDatabase
    from llm import OllamaLLM
    from text_vectorizer import tokenize


class RAGPipeline:
    """Orchestrates Retrieval-Augmented Generation pipeline"""

    def __init__(
        self,
        llm: OllamaLLM,
        db: VectorDatabase,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        num_retrieval: int = 3,
        relevance_threshold: float = 0.45,
    ):
        """
        Initialize RAG pipeline optimized for LOCAL LLM.
        
        Args:
            llm: OllamaLLM instance
            db: VectorDatabase instance
            chunk_size: Target tokens per chunk (default 600 = ~2400 chars)
            chunk_overlap: Overlap tokens (default 120 = ~480 chars)
            num_retrieval: Top-k chunks to retrieve (LOCAL LLM prefers fewer, higher quality)
            relevance_threshold: Min similarity to consider chunk relevant (default 0.45 = stricter)
        """
        self.llm = llm
        self.db = db
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_retrieval = num_retrieval
        # Strict threshold for local models (they hallucinate if given too much noise)
        self.relevance_threshold = relevance_threshold

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_file(self, file_path: str, original_filename: str | None = None) -> dict:
        start_time = time.time()

        FileProcessor.validate_file(file_path)
        text = FileProcessor.extract_text(file_path)
        text = FileProcessor.clean_text(text)
        chunks = FileProcessor.chunk_text(
            text, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )

        embeddings, total_embedding_time = self.llm.get_embeddings(chunks)

        if len(embeddings) != len(chunks):
            embeddings = [self.llm.get_embedding(chunk)[0] for chunk in chunks]

        file_id = str(uuid.uuid4())
        source_name = original_filename or os.path.basename(file_path)

        metadata_list = [
            {
                "source": file_path,
                "source_name": source_name,
                "file_id": file_id,
                "chunk_index": index,
                "chunk_length": len(chunk),
            }
            for index, chunk in enumerate(chunks)
        ]

        ids = [str(uuid.uuid4()) for _ in chunks]
        db_time = self.db.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata_list,
            ids=ids,
        )

        total_time = time.time() - start_time
        return {
            "success": True,
            "filename": source_name,
            "file_id": file_id,
            "chunks_created": len(chunks),
            "embedding_time": round(total_embedding_time, 4),
            "db_store_time": round(db_time, 4),
            "total_time": round(total_time, 4),
        }

    # ------------------------------------------------------------------ #
    # Query  (RAG path)
    # ------------------------------------------------------------------ #
    def query(
        self,
        question: str,
        temperature: float = 0.1,
        *,
        intent: str | None = None,
        verbosity: str | None = None,
        original_question: str | None = None,
    ) -> dict:
        total_start = time.time()

        # Route greetings / trivial prompts directly to the LLM
        if self._should_use_general_llm(question):
            answer, generation_time = self.llm.generate(
                prompt=question, context="", temperature=temperature, intent=intent, verbosity=verbosity
            )
            return {
                "question": original_question or question,
                "normalized_question": question,
                "answer": answer,
                "retrieved_documents": 0,
                "retrieved_chunks": [],
                "retrieved_sources": [],
                "timings": {
                    "generation": round(generation_time, 4),
                    "total": round(time.time() - total_start, 4),
                },
            }

        # If the database is empty, go straight to general LLM
        if self.db.document_count() == 0:
            answer, generation_time = self.llm.generate(
                prompt=question, context="", temperature=temperature, intent=intent, verbosity=verbosity
            )
            return {
                "question": original_question or question,
                "normalized_question": question,
                "answer": answer,
                "retrieved_documents": 0,
                "retrieved_chunks": [],
                "retrieved_sources": [],
                "timings": {
                    "generation": round(generation_time, 4),
                    "total": round(time.time() - total_start, 4),
                },
            }

        # Embed the query
        query_embedding, query_emb_time = self.llm.get_embedding(question)

        # Vector search with metadata filtering if specific files are mentioned
        indexed_files = self.db.list_indexed_file_names()
        where_filter = None
        
        # Check if any indexed filenames appear in the question (case-insensitive)
        mentioned_files = []
        lower_question = question.lower()
        for filename in indexed_files:
            if filename.lower() in lower_question:
                mentioned_files.append(filename)
        
        if mentioned_files:
            if len(mentioned_files) == 1:
                where_filter = {"source_name": mentioned_files[0]}
            else:
                where_filter = {"$or": [{"source_name": f} for f in mentioned_files]}

        search_result = self.db.search(
            query_embedding=query_embedding, 
            num_results=self.num_retrieval,
            where=where_filter
        )
        
        # If filtered search returns nothing but we mentioned files, 
        # it might be a mismatch in retrieval. Fallback to general search.
        if mentioned_files and not search_result["results"]:
            search_result = self.db.search(
                query_embedding=query_embedding, 
                num_results=self.num_retrieval
            )

        retrieval_time = search_result["retrieval_time"]
        retrieved_docs = search_result["results"]

        # Filter out low-similarity chunks to avoid mixing unrelated documents
        # Similarity is (1 - distance) because Chroma returns cosine distance for our config.
        filtered_docs = []
        for d in retrieved_docs:
            try:
                sim = 1.0 - float(d.get("distance", 1.0))
            except Exception:
                sim = 0.0
            if sim >= self.relevance_threshold:
                filtered_docs.append(d)
        retrieved_docs = filtered_docs

        # Compute top-1 cosine similarity
        top_similarity = 0.0
        if retrieved_docs:
            try:
                top_similarity = 1.0 - float(retrieved_docs[0].get("distance", 1.0))
            except Exception:
                top_similarity = 0.0

        # For broad knowledge questions, require a stronger match before using documents.
        if not self._should_use_document_context(question, intent, mentioned_files, top_similarity):
            answer, generation_time = self.llm.generate(
                prompt=question, context="", temperature=temperature, intent=intent, verbosity=verbosity
            )
            return {
                "question": original_question or question,
                "normalized_question": question,
                "answer": answer,
                "retrieved_documents": 0,
                "retrieved_chunks": [],
                "retrieved_sources": [],
                "timings": {
                    "query_embedding": round(query_emb_time, 4),
                    "retrieval": round(retrieval_time, 4),
                    "generation": round(generation_time, 4),
                    "total": round(time.time() - total_start, 4),
                },
            }

        # Merge adjacent chunks to avoid fragmentation
        merged_docs = self._merge_adjacent_chunks(retrieved_docs)

        # Build rich context with source labels and merged chunks
        context_parts = []
        for i, doc in enumerate(merged_docs, 1):
            src = (
                doc["metadata"].get("source_name")
                or os.path.basename(doc["metadata"].get("source", "Unknown"))
            )
            chunk_text = self._clean_text(doc["text"])
            context_parts.append(f"[Source {i}: {src}]\n{chunk_text}")

        context_text = "\n\n---\n\n".join(context_parts)

        retrieved_sources = [
            {
                "source": (
                    doc["metadata"].get("source_name")
                    or os.path.basename(doc["metadata"].get("source", "Unknown"))
                ),
                "snippet": self._clean_text(doc["text"][:350]).strip(),
                "similarity": round(1.0 - float(doc.get("distance", 1.0)), 4),
            }
            for doc in merged_docs
        ]

        answer, generation_time = self.llm.generate(
            prompt=question,
            context=context_text,
            temperature=temperature,
            intent=intent,
            verbosity=verbosity,
        )

        return {
            "question": original_question or question,
            "normalized_question": question,
            "answer": answer,
            "retrieved_documents": len(merged_docs),
            "retrieved_chunks": [
                self._clean_text(doc["text"][:300]) + "..." for doc in merged_docs
            ],
            "retrieved_sources": retrieved_sources,
            "timings": {
                "query_embedding": round(query_emb_time, 4),
                "retrieval": round(retrieval_time, 4),
                "generation": round(generation_time, 4),
                "total": round(time.time() - total_start, 4),
            },
        }

    # ------------------------------------------------------------------ #
    # Direct LLM (no RAG)
    # ------------------------------------------------------------------ #
    def query_without_rag(
        self,
        question: str,
        temperature: float = 0.5,
        *,
        intent: str | None = None,
        verbosity: str | None = None,
        original_question: str | None = None,
    ) -> dict:
        total_start = time.time()
        answer, generation_time = self.llm.generate(
            prompt=question, context="", temperature=temperature, intent=intent, verbosity=verbosity
        )
        return {
            "question": original_question or question,
            "normalized_question": question,
            "answer": answer,
            "retrieved_documents": 0,
            "retrieved_sources": [],
            "timings": {
                "generation": round(generation_time, 4),
                "total": round(time.time() - total_start, 4),
            },
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _merge_adjacent_chunks(self, retrieved_docs: list, merge_threshold: int = 1500) -> list:
        """
        Merge adjacent chunks from the same source to avoid fragmentation.
        
        If consecutive chunks are from the same file and their combined size is < threshold,
        merge them to preserve context.
        
        Args:
            retrieved_docs: List of retrieved documents
            merge_threshold: Max chars before stopping merges (default 1500 ≈ 375 tokens)
        
        Returns:
            Potentially merged list of documents
        """
        if len(retrieved_docs) <= 1:
            return retrieved_docs

        merged = []
        current_batch = [retrieved_docs[0]]
        current_size = len(retrieved_docs[0]["text"])

        for doc in retrieved_docs[1:]:
            same_file = (
                doc["metadata"].get("source_name")
                == current_batch[0]["metadata"].get("source_name")
            )
            combined_size = current_size + len(doc["text"])

            if same_file and combined_size < merge_threshold and len(current_batch) < 3:
                # Add to current batch
                current_batch.append(doc)
                current_size = combined_size
            else:
                # Save batch and start new one
                if current_batch:
                    merged.append(self._merge_batch(current_batch))
                current_batch = [doc]
                current_size = len(doc["text"])

        # Add final batch
        if current_batch:
            merged.append(self._merge_batch(current_batch))

        return merged

    def _merge_batch(self, batch: list) -> dict:
        """Merge a batch of documents from the same source."""
        if len(batch) == 1:
            return batch[0]

        # Combined text with clear separation
        merged_text = "\n\n".join(doc["text"] for doc in batch)
        
        return {
            "text": merged_text,
            "metadata": batch[0]["metadata"],
            "distance": min(float(doc.get("distance", 1.0)) for doc in batch),
        }

    def _has_document_cues(self, question: str) -> bool:
        """Detect whether the user is explicitly asking about uploaded files/documents."""
        lower = (question or "").lower()
        cues = (
            "uploaded",
            "document",
            "file",
            "pdf",
            "doc",
            "docx",
            "csv",
            "json",
            "spreadsheet",
            "sheet",
            "table",
            "in the document",
            "in this document",
            "according to",
            "from the file",
            "source",
        )
        return any(cue in lower for cue in cues)

    def _should_use_document_context(
        self,
        question: str,
        intent: str | None,
        mentioned_files: list[str],
        top_similarity: float,
    ) -> bool:
        """Decide whether retrieved chunks are strong enough to answer from files."""
        if mentioned_files:
            return True

        if not self._has_document_cues(question):
            if (intent or "").strip().lower() in {"definition", "explanation", "comparison", "list", "deep_dive"}:
                return top_similarity >= 0.68
            return top_similarity >= 0.60

        return top_similarity >= self.relevance_threshold

    def _should_use_general_llm(self, question: str) -> bool:
        """Route only true greetings / empty prompts to LLM directly."""
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        if not normalized:
            return True

        greeting_patterns = {
            "hi", "hello", "hey", "helo",
            "good morning", "good afternoon", "good evening",
            "thanks", "thank you",
            "bye", "goodbye",
        }
        return normalized in greeting_patterns

    def get_statistics(self) -> dict:
        db_stats = self.db.get_collection_stats()
        return {
            "database": db_stats,
            "llm_model": self.llm.model,
            "available_models": [],
            "chunk_config": {
                "size": self.chunk_size,
                "overlap": self.chunk_overlap,
            },
        }

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"<escape>.*?$", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\\u003c.*?\\u003e", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        return cleaned.strip()
