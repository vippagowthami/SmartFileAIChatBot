import re
import time
import os

import requests

try:
    from .text_vectorizer import tokenize, vectorize_text
except Exception:
    from text_vectorizer import tokenize, vectorize_text


class OllamaLLM:
    """Interface for Ollama local LLM"""

    def __init__(
        self,
        model: str = "llama3",
        embedding_model: str = "all-minilm",
        base_url: str = "http://localhost:11434",
        use_ollama_embeddings: bool | None = None,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url
        if use_ollama_embeddings is None:
            self.use_ollama_embeddings = (
                os.getenv("USE_OLLAMA_EMBEDDINGS", "true").lower() == "true"
            )
        else:
            self.use_ollama_embeddings = use_ollama_embeddings
        self.generate_endpoint = f"{base_url}/api/generate"
        self.embed_endpoint = f"{base_url}/api/embed"
        self.legacy_embeddings_endpoint = f"{base_url}/api/embeddings"

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def get_embedding(self, text: str) -> tuple[list[float], float]:
        """Generate a single embedding."""
        embeddings, elapsed = self.get_embeddings([text])
        return embeddings[0] if embeddings else vectorize_text(text), elapsed

    def get_embeddings(self, texts: list[str]) -> tuple[list[list[float]], float]:
        """Generate embeddings for one or more texts."""
        start = time.time()
        if not texts:
            return [], 0.0

        if not self.use_ollama_embeddings:
            return [vectorize_text(t) for t in texts], time.time() - start

        payload = {
            "model": self.embedding_model,
            "input": texts if len(texts) > 1 else texts[0],
        }

        try:
            response = requests.post(
                self.embed_endpoint,
                json=payload,
                timeout=60,  # generous timeout for large batches
            )

            if response.status_code == 404:
                # Older Ollama: use legacy endpoint one-by-one
                if len(texts) == 1:
                    legacy_payload = {"model": self.embedding_model, "prompt": texts[0]}
                    lr = requests.post(
                        self.legacy_embeddings_endpoint,
                        json=legacy_payload,
                        timeout=30,
                    )
                    lr.raise_for_status()
                    return [lr.json().get("embedding", vectorize_text(texts[0]))], time.time() - start
                else:
                    result_embeddings = []
                    total_elapsed = 0.0
                    for t in texts:
                        emb, el = self.get_embedding(t)
                        result_embeddings.append(emb)
                        total_elapsed += el
                    return result_embeddings, total_elapsed

            response.raise_for_status()
            result = response.json()

            if result.get("embeddings"):
                return result["embeddings"], time.time() - start
            elif result.get("embedding"):
                return [result["embedding"]], time.time() - start

            raise ValueError("Empty embeddings from Ollama")

        except Exception:
            # Fallback to local hash-based embeddings
            return [vectorize_text(t) for t in texts], time.time() - start

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.5,
        *,
        intent: str | None = None,
        verbosity: str | None = None,
    ) -> tuple[str, float]:
        """Generate a response. Uses RAG context when available."""
        start = time.time()
        if context.strip():
            answer = self._generate_document_answer(prompt, context, temperature, intent=intent, verbosity=verbosity)
        else:
            answer = self._generate_general_answer(prompt, temperature, intent=intent, verbosity=verbosity)
        return answer, time.time() - start

    # ------------------------------------------------------------------ #
    # Document-based answer
    # ------------------------------------------------------------------ #
    def _generate_document_answer(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.1,
        *,
        intent: str | None = None,
        verbosity: str | None = None,
    ) -> str:
        intent = (intent or "").strip().lower() or "explanation"
        verbosity = (verbosity or "").strip().lower() or "medium"

        length_hint = {
            "concise": "Answer in 3–6 lines. Be direct.",
            "medium": "Give a clear, structured answer. Keep it moderately detailed.",
            "detailed": "Give a detailed breakdown with sections and examples where relevant.",
        }.get(verbosity, "Give a clear, structured answer.")

        system_message = (
            "You are Smart File AI Chat, a local assistant answering questions ONLY from provided documents.\n\n"
            "STRICT RULES (MANDATORY):\n"
            "1) ANSWER ONLY FROM DOCUMENT CONTEXT. Do NOT use external knowledge or hallucinate.\n"
            "2) If the question is answered in the documents, extract the answer verbatim and add citation: (src: filename)\n"
            "3) If documents do NOT contain the answer, respond with: 'The uploaded documents do not contain information about this topic.'\n"
            "4) NEVER guess, infer, or add information not explicitly in the documents.\n"
            "5) Format: Clear bullet points or short paragraphs. Include citations for all claims.\n"
            f"Intent: {intent}. {length_hint}\n\n"
            "Remember: If unsure, say so. Never fabricate."
        )

        full_prompt = (
            f"{system_message}\n\n"
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"QUESTION: {prompt}\n\n"
            f"ANSWER (from documents only):"
        )

        try:
            response = requests.post(
                self.generate_endpoint,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                            "temperature": temperature,
                            "top_p": 0.9,
                        "num_predict": 2000,
                        },
                },
                timeout=120,
            )
            response.raise_for_status()
            generated = response.json().get("response", "").strip()
            generated = self._clean_response(generated)

            # If the model produced a refusal-like string, prefer a safe document snippet instead
            lower_g = (generated or "").lower()
            refusal_like = self._is_refusal(generated) or bool(re.search(r"\bi cannot answer\b|\bi am unable to\b|\bi cannot provide\b", lower_g))

            if generated and len(generated) > 10 and not self._is_refusal(generated):
                sources = self._extract_sources_from_context(context)

                # If we have document sources but the generated text looks short
                # or does not end cleanly, append a safe document snippet so
                # the user always receives a complete answer.
                if sources and (not generated.endswith(('.', '!', '?')) or len(generated) < 300 or re.search(r"\w-$", generated)):
                    fallback_snippet = self._fallback_document_answer(prompt, context)
                    combined = generated.rstrip() + "...\n\n" + fallback_snippet
                    return self._enforce_rag_response_format(combined, sources)

                return self._enforce_rag_response_format(generated, sources)
        except Exception:
            pass

        fallback = self._fallback_document_answer(prompt, context)
        sources = self._extract_sources_from_context(context)
        # Fallback might be a general answer; only enforce strict RAG format when we actually have sources.
        return self._enforce_rag_response_format(fallback, sources) if sources else fallback

    # ------------------------------------------------------------------ #
    # General answer (no documents)
    # ------------------------------------------------------------------ #
    def _generate_general_answer(
        self,
        prompt: str,
        temperature: float = 0.5,
        *,
        intent: str | None = None,
        verbosity: str | None = None,
    ) -> str:
        # Handle greetings with a fast static response
        if self._is_greeting_prompt(prompt):
            return self._greeting_response(prompt)

        intent = (intent or "").strip().lower() or "explanation"
        verbosity = (verbosity or "").strip().lower() or "medium"
        length_hint = {
            "concise": "Keep it short (3–6 lines).",
            "medium": "Moderate length. Use bullets only if it helps clarity.",
            "detailed": "Detailed explanation with a clear breakdown and an example if useful.",
        }.get(verbosity, "Moderate length.")

        system_message = (
            "You are Smart File AI Chat.\n"
            "Write like an advanced conversational AI (ChatGPT/Claude style): natural, helpful, and non-robotic.\n"
            "Always answer the user's question directly (no deflection).\n"
            "If the question asks for a factual person/place/date/definition, answer the exact fact first and then add one short sentence of context.\n"
            "Do NOT mention limitations/capabilities.\n"
            f"Intent: {intent}. {length_hint}"
        )

        full_prompt = (
            f"{system_message}\n\n"
            f"QUESTION: {prompt}\n\n"
            f"ANSWER:"
        )

        try:
            def _call_ollama(prompt_text: str, temp: float, max_tokens: int) -> str:
                r = requests.post(
                    self.generate_endpoint,
                    json={
                        "model": self.model,
                        "prompt": prompt_text,
                        "stream": False,
                        "options": {
                            "temperature": temp,
                            "top_p": 0.9,
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=120,
                )
                r.raise_for_status()
                return self._clean_response(r.json().get("response", "").strip())

            token_budget = 380 if verbosity == "concise" else (650 if verbosity == "detailed" else 520)
            generated = _call_ollama(full_prompt, temperature, token_budget)

            if generated and len(generated) > 15 and not self._is_refusal(generated):
                if not self._is_low_quality_answer(generated, prompt):
                    return generated

            # Retry once with a stricter instruction set if the answer is low-quality.
            strict_system = (
                "You are Smart File AI Chat.\n"
                "You MUST answer the user's question directly.\n"
                "Do NOT ask follow-up questions for simple 'what is' / 'define' questions.\n"
                "Do NOT say you are limited, refuse, or mention capabilities.\n"
                "Write a natural, human-like answer (not a rigid template).\n"
                "Avoid placeholder text like 'concise description', 'key points', or generic filler.\n"
                "If the question is asking for a factual answer, give the exact fact first instead of a template.\n"
                "If the question asks for a definition, begin with a one-sentence definition and then add 2-4 concrete facts or examples.\n"
                f"Intent: {intent}. Length: {verbosity}.\n"
                "Be concise but complete for the requested length."
            )
            strict_prompt = f"{strict_system}\n\nQUESTION: {prompt}\n\nANSWER:"
            regenerated = _call_ollama(strict_prompt, min(0.2, temperature), token_budget + 400)

            if regenerated and len(regenerated) > 15 and not self._is_refusal(regenerated):
                if not self._is_low_quality_answer(regenerated, prompt):
                    return regenerated
        except Exception:
            pass

        return self._fallback_general_answer(prompt)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _is_greeting_prompt(self, prompt: str) -> bool:
        # More robust greeting detection for short, casual inputs.
        s = (prompt or "").strip()
        if not s:
            return True

        # If the prompt is long, don't treat it as a greeting.
        if len(s) > 60:
            return False

        # Normalize repeated letters and non-alpha chars
        normalized = re.sub(r"([a-z])\1+", r"\1", s.lower())
        normalized = re.sub(r"[^a-z\s]", " ", normalized).strip()
        tokens = [t for t in normalized.split() if t]

        greetings = {
            "hi", "hello", "hey", "helo", "hy", "hlo",
            "morning", "afternoon", "evening",
            "thanks", "thank", "thankyou", "thx", "ty",
            "bye", "goodbye", "farewell",
        }

        casual = {"yaar", "buddy", "mate", "bruh", "dude"}

        if not tokens:
            return False

        # If all tokens are greetings or casual words, treat as greeting
        if all(t in greetings or t in casual for t in tokens):
            return True

        # Short messages where the first token is a greeting are also greetings
        if len(tokens) <= 3 and tokens[0] in greetings:
            return True

        return False

    def _greeting_response(self, prompt: str) -> str:
        lower = (prompt or "").strip().lower()
        if any(t in lower for t in ("thanks", "thank you", "thx", "ty")):
            return "You’re welcome — glad to help! Anything else you want to ask?"
        if any(t in lower for t in ("bye", "goodbye", "farewell", "see you", "see ya")):
            return "Goodbye! Come back anytime if you need more help."
        if "morning" in lower:
            return "Good morning! What can I help you with today?"
        if "afternoon" in lower:
            return "Good afternoon! What can I help you with today?"
        if "evening" in lower:
            return "Good evening! What can I help you with today?"

        # Casual greetings
        if any(c in lower for c in ("yaar", "buddy", "mate", "dude", "bruh")):
            return "Hey! I’m here — how can I help you today?"

        # Default friendly greeting
        return "Hi there! I’m Smart File AI — ask me a question or upload a document and I’ll help." 

    def _clean_response(self, text: str) -> str:
        """Removes trailing refusal appendages generated by over-aligned models."""
        refusal_markers = [
            "i cannot fulfill this request",
            "my current capabilities are limited",
            "i cannot access",
            "i cannot summarize",
            "i am unable to",
            "i apologize, but i cannot",
            "i am sorry, but i cannot",
            "i'm sorry, but i cannot",
        ]
        
        cleaned = text or ""

        # Strip "function call"/tool artifacts some models emit
        # Examples seen:
        #   ң{ ... }<end_function_call>ң{ ... }<end_function_call>
        #   <end_function_call>
        cleaned = re.sub(r"<\s*end_function_call\s*>", "", cleaned, flags=re.IGNORECASE)
        # Remove Cyrillic-prefixed JSON blobs even when concatenated without newlines
        cleaned = re.sub(r"[\u0400-\u04FF]\s*\{[\s\S]*?\}", "", cleaned)
        # Remove obvious tool/metadata JSON blobs (FROM/LICENSE/DESCRIPTION) if still present
        cleaned = re.sub(r"\{\s*\"FROM\"\s*:\s*\"[^\"]+\"[\s\S]*?\}", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        for marker in refusal_markers:
            idx = cleaned.lower().rfind(marker)
            if idx > 0: # Only strip if it's APPENDED (not at the very start)
                cleaned = cleaned[:idx].strip()
                
        # Strip weird model artifacts like <escape>ط that sometimes appear before appending refusals
        cleaned = re.sub(r'<escape>[^\s]*\s*$', '', cleaned, flags=re.IGNORECASE).strip()
        
        # Also clean trailing special characters that might be left over
        cleaned = re.sub(r'[\r\n]+$', '', cleaned).strip()
        return cleaned

    def _is_low_quality_answer(self, answer: str, prompt: str) -> bool:
        """
        Detects common "non-answers" (vague meta text, asking user to specify)
        so we can replace them with a real answer.
        """
        a = (answer or "").strip().lower()
        p = (prompt or "").strip().lower()

        # Model asks to "specify which aspect" instead of answering
        bad_markers = [
            "could you specify which aspect",
            "could you specify which part",
            "please specify which aspect",
            "which aspect would you like",
            "i would be happy to provide",
            "my capabilities are limited",
            "i cannot provide a detailed explanation",
            "i cannot answer questions about programming",
            "is an important concept with several key aspects",
            "concise description of what",
            "the primary purpose, common use cases",
            "a short real-world example illustrating",
            "typical scenarios or problems where",
        ]
        if any(m in a for m in bad_markers):
            return True

        # For simple definition questions, refusing to define is low quality.
        simple_q = (
            p.startswith("what is ")
            or p.startswith("define ")
            or p.startswith("definition of ")
            or p.startswith("what are ")
        )
        if simple_q and ("?" in a and len(a) < 220):
            # short response ending in a question is usually a deflection
            return True

        return False

    def _is_refusal(self, text: str) -> bool:
        """Only flag genuine outright refusals (response starts with a refusal phrase)."""
        lower = text.lower().strip()
        
        # We want to be very selective here. Only catch if the AI COMPLETELY refuses.
        # If it says "I'm sorry, I can't find that in the doc, but here is what I know...", that's NOT a refusal.
        starts = [
            "i cannot assist with this",
            "i cannot help with this",
            "i am unable to assist",
            "this request cannot be fulfilled",
            "i cannot fulfill this request",
            "my current capabilities are limited",
            "i cannot provide a detailed explanation",
            "i cannot answer questions",
            "i'm sorry, but i cannot assist",
            "i am a smart file ai assistant",
            "i cannot answer questions about programming concepts",
            "my capabilities are limited to assisting",
        ]

        if any(lower.startswith(s) for s in starts):
            return True

        # Hard refusal phrases that are almost always refusals
        hard_refusals = [
            "i apologize, but i cannot",
            "i cannot answer",
            "i'm sorry, i cannot",
            "i am sorry, i cannot",
            "i am sorry, but i cannot",
            "i'm sorry, but i cannot",
            "sorry, i cannot assist",
        ]
        
        # Only count these as refusals if they are the ENTIRETY or majority of the response
        if any(s in lower for s in hard_refusals) and len(lower) < 150:
            return True
            
        return False

    # ------------------------------------------------------------------ #
    # Fallbacks (used when Ollama is down / times out)
    # ------------------------------------------------------------------ #
    def _fallback_document_answer(self, prompt: str, context: str) -> str:
        """Extract a snippet from context as a basic answer."""
        prompt_tokens = set(tokenize(prompt))
        best_text = ""
        best_score = -1
        best_source = ""

        current_source = ""
        current_lines: list[str] = []

        for line in context.splitlines():
            if line.startswith("[Source"):
                if current_lines:
                    text_block = " ".join(current_lines).strip()
                    score = len(prompt_tokens.intersection(tokenize(text_block)))
                    if score > best_score:
                        best_score = score
                        best_text = text_block
                        best_source = current_source
                    current_lines = []
                current_source = line.strip()
            elif line.strip():
                current_lines.append(line.strip())

        if current_lines:
            text_block = " ".join(current_lines).strip()
            score = len(prompt_tokens.intersection(tokenize(text_block)))
            if score > best_score:
                best_score = score
                best_text = text_block
                best_source = current_source

        if best_text and best_score > 1:
            snippet = best_text[:1000].strip()
            label = best_source if best_source else "the document"
            return f"Based on {label}:\n\n{snippet}"

        return self._fallback_general_answer(prompt)

    def _fallback_general_answer(self, prompt: str) -> str:
        """Last-resort answer when Ollama is completely unavailable."""
        cleaned = re.sub(r"\s+", " ", prompt.strip()).rstrip("?")
        lower = cleaned.lower()

        # Flexible check for greetings
        temp_norm = re.sub(r"([a-z])\1+", r"\1", lower)
        temp_norm = re.sub(r"[^a-z\s]", "", temp_norm).strip()
        if temp_norm in {"hi", "hello", "hey", "helo", "hy", "hlo", "thanks", "thank you", "thx", "ty"}:
            return "Hello! How can I help you today?"

        if "machine learning" in lower or "deep learning" in lower:
            return (
                "Machine Learning (ML) is a branch of AI where systems learn from data to make predictions or decisions. "
                "Deep Learning is a subset using multi-layer neural networks to learn complex patterns. "
                "Common applications include image recognition, NLP, and recommendation systems."
            )

        if "artificial intelligence" in lower or " ai " in lower or lower.endswith(" ai"):
            return (
                "Artificial Intelligence (AI) is the simulation of human intelligence in machines. "
                "It covers machine learning, natural language processing, computer vision, robotics, and more. "
                "AI systems can reason, learn, and adapt to solve complex problems."
            )

        if "python" in lower:
            return (
                "Python is a versatile, high-level programming language known for its readable syntax. "
                "It is widely used in web development, data science, machine learning, automation, and scripting. "
                "Its rich ecosystem of libraries (NumPy, Pandas, TensorFlow, Django) makes it a top choice for developers."
            )

        if "prime minister of india" in lower or re.fullmatch(r"who is (the )?(pm|prime minister) of india\??", lower):
            return "The Prime Minister of India is Narendra Modi."

        if "capital of india" in lower:
            return "The capital of India is New Delhi."

        if "president of india" in lower:
            return "The President of India is Droupadi Murmu."

        # High-quality deterministic mini-answers for common programming topics
        if "java" == lower.strip() or lower.startswith("what is java") or lower.startswith("define java"):
            return (
                "Java is a general-purpose, object-oriented programming language and runtime platform.\n\n"
                "- What it’s used for: backend services, Android apps, enterprise systems, and large-scale applications.\n"
                "- Key idea: write once, run anywhere (Java code runs on the JVM).\n"
                "- Why people use it: strong ecosystem, good performance, and lots of libraries/frameworks.\n"
                "- Example: many banking and enterprise systems are built in Java.\n"
            )

        if "oop" in lower or "oops" in lower or "object oriented" in lower:
            return (
                "OOP (Object-Oriented Programming) is a way to design programs using “objects” that bundle data + behavior.\n\n"
                "- Encapsulation: keep data + methods together; hide internal details.\n"
                "- Abstraction: expose only what’s necessary; hide complexity.\n"
                "- Inheritance: create new classes from existing ones (reuse/extend behavior).\n"
                "- Polymorphism: same interface, different implementations (e.g., method overriding).\n"
                "- Bonus ideas: classes/objects, interfaces, composition.\n"
            )

        if lower.startswith("what is ") or lower.startswith("define ") or lower.startswith("definition of "):
            # Extract the term after "what is"/"define"/"definition of"
            term = cleaned
            for prefix in ("what is ", "define ", "definition of "):
                if lower.startswith(prefix):
                    term = cleaned[len(prefix):].strip()
                    break
            term = term.strip().strip(".")
            if not term:
                return (
                    "Tell me the word/term you want defined, and I’ll define it with a short example."
                )
            return (
                f"{term.capitalize()} is a topic that is typically defined by what it is, how it works, and where it is used.\n\n"
                f"- Definition: a short, direct explanation of {term}.\n"
                f"- Why it matters: the practical reason people use or study {term}.\n"
                f"- Example: one concrete example that makes {term} easier to understand.\n"
            )

        # Always provide a useful general answer even without documents.
        return (
            f"Here’s a clear general explanation of '{cleaned}'.\n\n"
            "- Definition: the core meaning or idea.\n"
            "- How it works: the main mechanism or concept.\n"
            "- Example: one practical example.\n"
            "- Takeaway: why it matters or when to use it.\n\n"
            "If you want, I can also give a beginner-friendly version, a comparison, or a step-by-step explanation."
        )

    def list_available_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # Response formatting enforcement (RAG policy)
    # ------------------------------------------------------------------ #
    def _extract_sources_from_context(self, context: str) -> list[str]:
        """
        Extracts `Source X: filename` labels from the assembled RAG context.
        The context is produced by `RAGPipeline` in the form:
          [Source 1: myfile.pdf]
          <chunk>
        """
        sources: list[str] = []
        for line in context.splitlines():
            m = re.match(r"^\[Source\s+\d+:\s*(.+?)\]\s*$", line.strip())
            if m:
                src = m.group(1).strip()
                if src and src not in sources:
                    sources.append(src)
        return sources

    def _enforce_rag_response_format(self, text: str, sources: list[str]) -> str:
        """
        Enforces the response format rules when we have retrieved document context.
        - Adds [From Files] section if missing
        - Ensures citations are present as (src: filename[, filename...]) in the file-based section
        - If the model indicates insufficiency, ensures the exact required sentence is present
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        src_suffix = ""
        if sources:
            src_suffix = f"(src: {', '.join(sources)})"

        # If the model already produced structured sections, only patch citations + required sentence.
        has_from = "[from files]" in cleaned.lower()
        has_general = "[general answer]" in cleaned.lower()
        says_insufficient = "the uploaded files do not fully contain the answer." in cleaned.lower()

        if has_from or has_general:
            out = cleaned

            # If it contains a General Answer section, ensure the required insufficiency sentence appears first.
            if has_general and not says_insufficient:
                out = "The uploaded files do not fully contain the answer.\n\n" + out

            # Ensure a source citation exists somewhere in the From Files section.
            if sources:
                # If any (src: ...) exists already, don't duplicate.
                if "(src:" not in out.lower():
                    out = out.rstrip() + f"\n\n{src_suffix}"
            return out

        # Unstructured response on the RAG path: wrap it as a file-based answer and add citations.
        if sources:
            return f"[From Files]\n{cleaned}\n\n{src_suffix}"

        # If no sources, return as-is (should be rare on RAG path).
        return cleaned
