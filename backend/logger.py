import json
import os
from datetime import datetime
from pathlib import Path


class ConversationLogger:
    """Handles logging of questions, answers, and timing metrics"""

    def __init__(self, log_dir: str = "../logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"chatbot_log_{datetime.now().strftime('%Y%m%d')}.json"

    def log_interaction(
        self,
        question: str,
        answer: str,
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        retrieved_chunks: list = None,
        model: str = "default",
    ):
        """
        Log a single interaction to JSON file

        Args:
            question: User's question
            answer: AI's generated answer
            retrieval_time: Time taken for RAG retrieval (seconds)
            generation_time: Time taken for LLM generation (seconds)
            total_time: Total time (seconds)
            retrieved_chunks: List of retrieved document chunks
            model: Model name used
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "timings": {
                "retrieval_seconds": round(retrieval_time, 4),
                "generation_seconds": round(generation_time, 4),
                "total_seconds": round(total_time, 4),
            },
            "chunks_retrieved": len(retrieved_chunks) if retrieved_chunks else 0,
            "model": model,
        }

        logs = []
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []

        logs.append(log_entry)

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    def log_file_upload(self, filename: str, num_chunks: int, embedding_time: float):
        """Log file upload and embedding process"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "file_upload",
            "filename": filename,
            "num_chunks": num_chunks,
            "embedding_time_seconds": round(embedding_time, 4),
        }

        logs = []
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []

        logs.append(log_entry)

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
