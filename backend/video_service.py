"""
Service for converting text-based lessons into structured video generation JSON.
"""

import json
import requests
from typing import List, Dict, Any

class VideoScriptService:
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = f"{ollama_url}/api/generate"
        self.model = model

    def generate_script(self, lesson_text: str, language: str = "en") -> Dict[str, Any]:
        """
        Converts a lesson script into a structured video generation JSON.
        """
        prompt = (
            "You are a professional video script writer and AI visual engineer.\n"
            "Your task is to convert the following LESSON TEXT into a structured VIDEO GENERATION JSON format.\n\n"
            "JSON SCHEMA:\n"
            "{\n"
            "  \"title\": \"String\",\n"
            "  \"language\": \"String (ISO 639-1)\",\n"
            "  \"scenes\": [\n"
            "    {\n"
            "      \"id\": Number,\n"
            "      \"visual_prompt\": \"Detailed description for an AI image/video model (e.g., Midjourney/Sora style)\",\n"
            "      \"narration\": \"The exact text to be spoken in this scene\",\n"
            "      \"on_screen_text\": \"Keywords to display on screen (optional)\",\n"
            "      \"duration_estimate\": Number (in seconds)\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"LESSON TEXT:\n{lesson_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Break the lesson into 4-8 logical scenes.\n"
            "2. Visual prompts should be cinematic, descriptive, and consistent.\n"
            "3. Narration MUST be in the requested language.\n"
            "4. RETURN ONLY VALID JSON. No extra text.\n"
            "5. The output should be ready for a video generation pipeline."
        )

        def _call_ollama(use_json_mode: bool):
            json_payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9
                }
            }
            if use_json_mode:
                json_payload["format"] = "json"
            
            return requests.post(self.ollama_url, json=json_payload, timeout=90)

        try:
            print(f"[VideoScriptService] Calling Ollama at {self.ollama_url} with model {self.model}")
            
            # Attempt 1: With JSON mode
            response = _call_ollama(True)
            
            # If 400 or something, maybe JSON mode isn't supported, try Attempt 2
            if response.status_code != 200:
                print(f"[VideoScriptService] JSON mode failed ({response.status_code}), trying without it...")
                response = _call_ollama(False)

            if response.status_code != 200:
                error_msg = f"Ollama returned {response.status_code}: {response.text}"
                print(f"[VideoScriptService Error] {error_msg}")
                return {"error": error_msg, "scenes": []}

            raw_response = response.json().get("response", "").strip()
            
            # Clean up potential markdown blocks if LLM ignores "format: json"
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0].strip()
            
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError as je:
                print(f"[VideoScriptService Error] JSON Decode Failed: {je}")
                print(f"[VideoScriptService] Raw Output: {raw_response[:500]}...")
                return {"error": f"Invalid JSON format: {str(je)}", "scenes": []}

        except Exception as e:
            print(f"[VideoScriptService Exception] {e}")
            return {
                "error": str(e),
                "title": "Failed to generate script",
                "scenes": []
            }

# Factory function instead of a restrictive singleton
def get_video_service(ollama_url: str = "http://127.0.0.1:11434", model: str = "llama3"):
    return VideoScriptService(ollama_url, model)
