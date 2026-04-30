import os
import re
from pathlib import Path


class FileProcessor:
    """Handles file upload and text extraction"""

    SUPPORTED_FORMATS = {".txt", ".pdf", ".doc", ".docx"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    @staticmethod
    def extract_text(file_path: str) -> str:
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".txt":
            return FileProcessor._extract_txt(file_path)
        elif file_ext == ".pdf":
            return FileProcessor._extract_pdf(file_path)
        elif file_ext in {".doc", ".docx"}:
            return FileProcessor._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @staticmethod
    def _extract_txt(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        try:
            import PyPDF2
            text = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n\n".join(text)
        except ImportError:
            raise ImportError("PyPDF2 required. Run: pip install PyPDF2")
        except Exception as e:
            raise Exception(f"Failed to extract PDF: {e}")

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise ImportError("python-docx required. Run: pip install python-docx")
        except Exception as e:
            raise Exception(f"Failed to extract DOCX: {e}")

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
        """
        Split text into chunks respecting paragraph and sentence boundaries.
        Uses large chunks (default 1500 chars) for better context preservation.
        """
        if not text.strip():
            return []

        if len(text) <= chunk_size:
            return [text.strip()]

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size

            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Prefer paragraph boundary
            break_pos = text.rfind("\n\n", start + chunk_size // 2, end)

            if break_pos == -1:
                # Prefer sentence boundary
                for punct in (". ", "! ", "? ", "\n"):
                    bp = text.rfind(punct, start + chunk_size // 2, end)
                    if bp != -1:
                        break_pos = bp + len(punct)
                        break
                else:
                    # Hard cut
                    break_pos = end
            else:
                break_pos += 2  # skip the "\n\n"

            chunk = text[start:break_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Slide forward, keeping overlap
            start = max(start + 1, break_pos - overlap)

        return chunks

    @staticmethod
    def validate_file(file_path: str) -> bool:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in FileProcessor.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}. Supported: {FileProcessor.SUPPORTED_FORMATS}")
        if path.stat().st_size > FileProcessor.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {path.stat().st_size} bytes. Max: {FileProcessor.MAX_FILE_SIZE}")
        return True

    @staticmethod
    def clean_text(text: str) -> str:
        # Collapse excessive whitespace but preserve paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
        return text.strip()
