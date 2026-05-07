"""
Language detection and translation service for multilingual support.
Provides automatic language detection, language preference persistence,
and optional translation fallback.
"""

import time
from typing import Literal
from functools import lru_cache

try:
    from langdetect import detect, detect_langs, LangDetectException
except ImportError:
    detect = None
    detect_langs = None
    LangDetectException = Exception

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None


# Supported languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
}


class LanguageDetectionError(Exception):
    """Raised when language detection fails"""
    pass


class TranslationError(Exception):
    """Raised when translation fails"""
    pass


class LanguageService:
    """Service for language detection, preference persistence, and translation"""
    
    def __init__(self):
        """Initialize language service with caching"""
        self._detection_cache = {}  # Cache detected languages by text hash
        self._translation_cache = {}  # Cache translations
        
    def get_supported_languages(self) -> dict:
        """Returns the dictionary of supported languages"""
        return SUPPORTED_LANGUAGES
        
    def detect_language(self, text: str, fallback: str = "en") -> dict:
        """
        Detect language of given text.
        
        Args:
            text: Text to detect language for
            fallback: Fallback language code if detection fails
            
        Returns:
            Dictionary with keys:
                - code: ISO 639-1 language code (e.g., 'en', 'es')
                - name: Human-readable language name
                - confidence: Confidence score (0-1)
                - is_supported: Whether we support translation for this language
        """
        if not text or not text.strip():
            return {"code": fallback, "name": SUPPORTED_LANGUAGES.get(fallback, "Unknown"), 
                    "confidence": 0.0, "is_supported": fallback in SUPPORTED_LANGUAGES}
        
        # Check cache
        text_hash = hash(text[:100])  # Hash first 100 chars
        if text_hash in self._detection_cache:
            return self._detection_cache[text_hash]
        
        result = {
            "code": fallback,
            "name": SUPPORTED_LANGUAGES.get(fallback, "Unknown"),
            "confidence": 0.0,
            "is_supported": fallback in SUPPORTED_LANGUAGES,
        }
        
        if not detect:
            # langdetect not available, return fallback
            return result
        
        try:
            # Detect language - returns string like 'en'
            detected_code = detect(text)
            
            # Get confidence scores for more detailed info
            try:
                langs_probs = detect_langs(text)
                confidence = max([p.prob for p in langs_probs], default=0.0)
            except:
                confidence = 0.95  # High confidence if single detection succeeded
            
            result = {
                "code": detected_code.lower(),
                "name": SUPPORTED_LANGUAGES.get(detected_code.lower(), detected_code),
                "confidence": confidence,
                "is_supported": detected_code.lower() in SUPPORTED_LANGUAGES,
            }
        except LangDetectException:
            # Detection failed, use fallback
            pass
        except Exception as e:
            print(f"[LanguageService] Unexpected error in detect_language: {e}")
        
        # Cache result
        self._detection_cache[text_hash] = result
        return result
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> dict:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (default 'auto' for auto-detect)
            target_lang: Target language code (default 'en' for English)
            
        Returns:
            Dictionary with keys:
                - original: Original text
                - translated: Translated text
                - source_lang: Source language code
                - target_lang: Target language code
                - success: Whether translation succeeded
                - error: Error message if translation failed
        """
        result = {
            "original": text,
            "translated": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "success": True,
            "error": None,
        }
        
        if not text or not text.strip():
            return result
        
        if source_lang == target_lang:
            return result
        
        if not GoogleTranslator:
            result["success"] = False
            result["error"] = "Translation library not installed. Install 'deep-translator' for translation support."
            return result
        
        # Check cache
        cache_key = f"{hash(text[:100])}_{source_lang}_{target_lang}"
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]
        
        try:
            if source_lang == "auto":
                # Auto-detect source language
                detected = self.detect_language(text)
                source_lang = detected["code"]
            
            # Translate
            translator = GoogleTranslator(source_language=source_lang, target_language=target_lang)
            translated_text = translator.translate(text)
            
            result["translated"] = translated_text
            result["source_lang"] = source_lang
            result["success"] = True
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["translated"] = text  # Return original if translation fails
            print(f"[LanguageService] Translation error: {e}")
        
        # Cache result
        self._translation_cache[cache_key] = result
        return result
    
    def get_supported_languages(self) -> dict:
        """Get all supported languages"""
        return SUPPORTED_LANGUAGES.copy()
    
    def is_language_supported(self, lang_code: str) -> bool:
        """Check if language is supported for translation"""
        return lang_code.lower() in SUPPORTED_LANGUAGES
    
    def inject_language_directive(
        self,
        system_prompt: str,
        detected_lang: str,
        detected_lang_name: str,
        user_language_preference: str | None = None,
    ) -> str:
        """
        Inject language directives into system prompt to guide LLM response language.
        
        Args:
            system_prompt: Original system prompt
            detected_lang: Detected language code from user input
            detected_lang_name: Human-readable detected language name
            user_language_preference: User's language preference (if different from detected)
            
        Returns:
            Updated system prompt with language directives
        """
        if user_language_preference and user_language_preference != detected_lang:
            directive = (
                f"USER LANGUAGE: The user is communicating in {detected_lang_name} ({detected_lang}), "
                f"but prefers responses in {SUPPORTED_LANGUAGES.get(user_language_preference, user_language_preference)} ({user_language_preference}). "
                f"RESPOND IN {user_language_preference.upper()} ONLY.\n\n"
            )
        else:
            directive = (
                f"USER LANGUAGE: The user is communicating in {detected_lang_name} ({detected_lang}). "
                f"RESPOND IN {detected_lang.upper()} ONLY.\n\n"
            )
        
        # Insert directive at the beginning of system prompt
        return directive + system_prompt
    
    def clear_cache(self):
        """Clear language detection and translation caches"""
        self._detection_cache.clear()
        self._translation_cache.clear()


# Global instance
_language_service = None


def get_language_service() -> LanguageService:
    """Get or create global language service instance"""
    global _language_service
    if _language_service is None:
        _language_service = LanguageService()
    return _language_service
