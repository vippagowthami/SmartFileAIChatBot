"""Smart File AI Chatbot Backend Modules"""

from .llm import OllamaLLM
from .db import VectorDatabase
from .rag import RAGPipeline
from .logger import ConversationLogger
from .file_processor import FileProcessor

__version__ = "1.0.0"
__author__ = "Smart File AI"
__all__ = [
    "OllamaLLM",
    "VectorDatabase",
    "RAGPipeline",
    "ConversationLogger",
    "FileProcessor",
]
