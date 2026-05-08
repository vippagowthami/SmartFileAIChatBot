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
                timeout=30,  # Reduced timeout for better responsiveness
            )

            if response.status_code == 404:
                # Older Ollama: use legacy endpoint one-by-one
                if len(texts) == 1:
                    legacy_payload = {"model": self.embedding_model, "prompt": texts[0]}
                    lr = requests.post(
                        self.legacy_embeddings_endpoint,
                        json=legacy_payload,
                        timeout=15,  # Reduced timeout
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
        memory_context: str = "",
        teaching_style: str = "friendly",
        teaching_guidance: str = "",
        detected_language: str = "en",
        language_preference: str | None = None,
        model_override: str | None = None,
    ) -> tuple[str, float]:
        """Generate a response. Uses RAG context when available."""
        start = time.time()
        if context.strip():
            answer = self._generate_document_answer(
                prompt,
                context,
                temperature,
                intent=intent,
                verbosity=verbosity,
                memory_context=memory_context,
                teaching_style=teaching_style,
                teaching_guidance=teaching_guidance,
                detected_language=detected_language,
                language_preference=language_preference,
                model_override=model_override,
            )
        else:
            answer = self._generate_general_answer(
                prompt,
                temperature,
                intent=intent,
                verbosity=verbosity,
                memory_context=memory_context,
                teaching_style=teaching_style,
                teaching_guidance=teaching_guidance,
                detected_language=detected_language,
                language_preference=language_preference,
                model_override=model_override,
            )
        return answer, time.time() - start

    # ------------------------------------------------------------------ #
    # Document-based answer
    # ------------------------------------------------------------------ #
    def _generate_document_answer(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.2,
        *,
        intent: str | None = None,
        verbosity: str | None = None,
        memory_context: str = "",
        teaching_style: str = "friendly",
        teaching_guidance: str = "",
        detected_language: str = "en",
        language_preference: str | None = None,
        model_override: str | None = None,
    ) -> str:
        """Generation specifically for when we HAVE document context."""
        
        # Determine depth based on intent and verbosity
        depth_instruction = "Provide a concise, direct answer."
        if verbosity == "detailed" or intent in {"deep_dive", "explanation"}:
            depth_instruction = (
                "Provide a direct answer first, then a clear step-by-step explanation using plain headings and bullet points."
            )
        elif verbosity == "medium" or intent in {"list", "comparison"}:
            depth_instruction = "Provide a balanced, informative answer with relevant details."

        # Build language directive
        response_lang = language_preference or detected_language or "en"
        lang_map = {
            "en": "English",
            "hi": "Hindi",
            "te": "Telugu",
        }
        lang_name = lang_map.get(response_lang, response_lang)
        language_directive = f"RESPOND IN {lang_name.upper()} ({response_lang.upper()}) ONLY.\n"

        system_message = (
            "You are Smart File AI. Provide direct answers from DOCUMENT CONTEXT only.\n"
            f"{language_directive}"
            f"STYLE: {teaching_style}\n"
            f"{depth_instruction}\n"
            "RULES: 1) Answer first. 2) Cite [Source N]. 3) No hallucination. 4) Be concise."
        )

        prompt_payload = (
            f"{system_message}\n\n"
            f"STUDENT MEMORY:\n{memory_context or 'No stored memory available.'}\n\n"
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {prompt}\n\n"
            f"EXPERT ANSWER (Source-Based Only):"
        )

        try:
            response = requests.post(
                self.generate_endpoint,
                json={
                    "model": model_override or self.model,
                    "prompt": prompt_payload,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                            "num_predict": 512,
                    },
                },
                timeout=30,  # Fast response timeout
            )
            if response.status_code != 200:
                print(f"[Ollama Error] {response.status_code}: {response.text}")
            
            response.raise_for_status()
            generated = response.json().get("response", "").strip()
            generated = self._clean_response(generated)

            if generated and len(generated) > 10 and not self._is_refusal(generated):
                sources = self._extract_sources_from_context(context)
                if sources:
                    return self._enforce_rag_response_format(generated, sources)
                return generated
        except Exception as e:
            print(f"[Ollama Exception] {e}")
            pass

        return self._fallback_document_answer(prompt, context)

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
        memory_context: str = "",
        teaching_style: str = "friendly",
        teaching_guidance: str = "",
        detected_language: str = "en",
        language_preference: str | None = None,
        model_override: str | None = None,
    ) -> str:
        # Handle greetings with a fast static response
        if self._is_greeting_prompt(prompt):
            return self._greeting_response(prompt, memory_context=memory_context)

        # Build language directive
        response_lang = language_preference or detected_language or "en"
        lang_map = {
            "en": "English",
            "hi": "Hindi",
            "te": "Telugu",
        }
        lang_name = lang_map.get(response_lang, response_lang)
        language_directive = f"RESPOND IN {lang_name.upper()} ({response_lang.upper()}) ONLY.\n"

        # Determine depth for general answers
        depth_instruction = ""
        if verbosity == "detailed" or intent in {"deep_dive", "explanation"}:
            depth_instruction = (
                "Use a clear, step-by-step explanation in natural language with concise headings."
            )

        # Clear directive to avoid conversational confusion
        prompt_payload = (
            "Instruction: Directly answer the user's question clearly and accurately.\n\n"
            f"{language_directive}"
            f"STYLE: {depth_instruction}\n"
            "DO NOT expose internal memory, scoring, or reasoning traces in the output.\n"
            f"TEACHING STYLE: {teaching_style}\n"
            f"TEACHING GUIDANCE:\n{teaching_guidance or 'Use the teaching style naturally and adapt to the student.'}\n\n"
            f"STUDENT MEMORY:\n{memory_context or 'No stored memory available.'}\n\n"
            f"Question: {prompt}\n"
            "Answer:"
        )

        try:
            def _call_ollama(text: str, temp: float) -> str:
                r = requests.post(
                    self.generate_endpoint,
                    json={
                        "model": model_override or self.model,
                        "prompt": text,
                        "stream": False,
                        "options": {
                            "temperature": temp,
                            "top_p": 0.9,
                            "num_predict": 256,
                        },
                    },
                    timeout=30,
                )
                r.raise_for_status()
                return self._clean_response(r.json().get("response", "").strip())

            generated = _call_ollama(prompt_payload, temperature)

            if generated and len(generated) > 10 and not self._is_refusal(generated):
                return generated
        except Exception as e:
            print(f"[Ollama General Exception] {e}")
            pass

        return self._fallback_general_answer(prompt)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _is_greeting_prompt(self, prompt: str) -> bool:
        s = (prompt or "").strip().lower()
        if not s:
            return True
        if len(s) > 50:
            return False
        
        greetings = {"hi", "hello", "hey", "morning", "afternoon", "evening", "thanks", "thank", "thankyou", "thx", "ty", "bye", "goodbye"}
        tokens = s.split()
        if tokens and tokens[0] in greetings:
            return True

        casual = {"yaar", "buddy", "mate", "bruh", "dude"}
        acknowledgments = {"yes", "yeah", "yep", "ok", "okay", "k", "sure", "alright", "fine", "cool", "no", "nah", "nope"}

        if not tokens:
            return False

        # If all tokens are greetings, acknowledgments, or casual words, treat as a conversational prompt.
        if all(t in greetings or t in casual or t in acknowledgments for t in tokens):
            return True

        # Short messages where the first token is a greeting are also greetings
        if len(tokens) <= 3 and tokens[0] in greetings:
            return True

        if len(tokens) <= 3 and tokens[0] in acknowledgments:
            return True

        return False

    def _greeting_response_with_memory(self, prompt: str, memory_context: str = "") -> str:
        lower = (prompt or "").strip().lower()
        name = self._extract_name_from_memory(memory_context)
        goal = self._extract_goal_from_memory(memory_context)
        salutation = f"Hello {name}!" if name else "Hello!"

        if any(t in lower for t in ("thanks", "thank you", "thx", "ty")):
            return f"{salutation} You're most welcome. I'm glad I could help."
        if any(t in lower for t in ("bye", "goodbye", "farewell", "see you", "see ya")):
            return "Goodbye! Feel free to return anytime."
        if any(t in lower for t in ("yes", "yeah", "yep", "ok", "okay", "k", "sure", "alright", "fine", "cool")):
            return f"{salutation} Understood. Tell me what you'd like to continue with."
        if "morning" in lower:
            return f"Good morning{f' {name}' if name else ''}! How can I help today?"
        if "afternoon" in lower:
            return f"Good afternoon{f' {name}' if name else ''}! What would you like to work on?"
        if "evening" in lower:
            return f"Good evening{f' {name}' if name else ''}! What can I help you with?"

        # Casual greetings
        if any(c in lower for c in ("yaar", "buddy", "mate", "dude", "bruh")):
            return f"{salutation} I'm here and ready to help. What's on your mind?"

        # Default friendly greeting
        if goal:
            return f"{salutation} Great to see you again. Last time you wanted to {goal}. Shall we continue from there?"
        return f"{salutation} Great to see you again. How can I help you today?"

    def _greeting_response(self, prompt: str, memory_context: str = "") -> str:
        return self._greeting_response_with_memory(prompt, memory_context=memory_context)

    def _extract_name_from_memory(self, memory_context: str) -> str | None:
        text = memory_context or ""
        m = re.search(r"(?im)^\s*-\s*name\s*:\s*([A-Za-z][A-Za-z\-']{1,40})\s*$", text)
        if not m:
            return None
        return m.group(1).strip().title()

    def _extract_goal_from_memory(self, memory_context: str) -> str | None:
        text = memory_context or ""
        m = re.search(r"(?im)^\s*-\s*goals\s*:\s*(.+)$", text)
        if not m:
            return None
        raw = m.group(1).strip()
        return raw.split(",")[0].strip() if raw else None

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

        # Strip blackboard/internal reasoning markers if emitted by the model.
        cleaned = re.sub(r"\[/?BLACKBOARD_STEP[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?im)^\s*(student memory|student profile|conversation state|relevant past memory)\s*:\s*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*(internal reasoning|scratchpad|analysis)\s*:\s*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*(internal reasoning|scratchpad|analysis|debug|trace)\s*:\s*.*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*-\s*(follow-up detected|current entities|active entities from prior chats)\s*:.*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*memory\s+\d+\s*\[[^\]]*\]\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?im)\b(score|weak_signal|retrieved documents|timings?)\b\s*[:=].*$", "", cleaned)
        cleaned = re.sub(r"(?im)\b(chain of thought|chain-of-thought|cot|reasoning trace|thinking:)\b.*$", "", cleaned)

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
            "here’s a clear general explanation",
            "here is a clear general explanation",
            "definition: the core meaning or idea",
            "how it works: the main mechanism or concept",
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
            snippet = best_text[:1200].strip()
            label = best_source if best_source else "the document"
            return f"According to {label}:\n\n{snippet}"

        return self._fallback_general_answer(prompt)

    def _fallback_general_answer(self, prompt: str) -> str:
        """Last-resort answer when Ollama is completely unavailable."""
        cleaned = re.sub(r"\s+", " ", prompt.strip()).rstrip("?")
        lower = cleaned.lower()

        # Flexible check for greetings
        temp_norm = re.sub(r"([a-z])\1+", r"\1", lower)
        temp_norm = re.sub(r"[^a-z\s]", "", temp_norm).strip()
        if temp_norm in {"hi", "hello", "hey", "helo", "hy", "hlo", "thanks", "thank you", "thx", "ty", "yes", "yeah", "yep", "ok", "okay", "k", "sure", "alright", "fine", "cool", "no", "nah", "nope"}:
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

        if any(phrase in lower for phrase in ("your name", "who are you", "what is your name")):
            return "I’m Smart File AI. I help with questions and uploaded documents."

        if lower in {"what", "why", "how", "uhm", "uhmm", "um", "hmm", "y", "k", "ok", "okay", "yes", "no", "sure", "alright"}:
            return "Could you rephrase that or add a little more detail? I can answer general questions, explain topics, and help with documents."

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
            term = cleaned
            for prefix in ("what is ", "define ", "definition of "):
                if lower.startswith(prefix):
                    term = cleaned[len(prefix):].strip()
                    break
            term = term.strip().strip("?").strip(".")
            
            # If we have a common term, give a real mini-definition instead of a template
            if term == "corba":
                return (
                    "CORBA (Common Object Request Broker Architecture) is a standard designed to facilitate communication between systems on different platforms.\n\n"
                    "- Key Function: It allows applications to call methods on objects across a network as if they were local.\n"
                    "- Language Neutral: It supports multiple programming languages via IDL (Interface Definition Language).\n"
                    "- Middleware: It acts as an intermediary (Object Request Broker) that handles data marshalling and network details."
                )

            if not term:
                return "I'm not sure which term you'd like me to define. Could you please specify?"

            return (
                f"I'm sorry, I don't have a specific expert definition for '{term}' in my local knowledge bank right now. "
                f"However, '{term}' is generally a technical topic that involves specific architectural or programming principles. "
                "If you upload a document about it, I can provide a much more precise answer."
            )

        # Always provide a useful general answer even without documents.
        if len(cleaned.split()) <= 3:
            return (
                f"I understand you're asking about '{cleaned}'. Could you provide a bit more detail? "
                "I'm here to explain concepts, compare ideas, or analyze your uploaded documents."
            )

        return (
            f"I can help with '{cleaned}', but I need a bit more context to answer it well. "
            "Try adding a topic, goal, or example, and I’ll respond directly."
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
        Enforces the response format rules for RAG context.
        - Ensures citations are present.
        - Removes robotic section headers.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        # Check if the model already included any form of citation
        has_citations = False
        # Look for [Source 1], [Source 2], etc. or [filename]
        if re.search(r"\[Source \d+\]|\[.+?\]", cleaned):
            has_citations = True
        
        # If no citations were found, append them professionally at the bottom
        if not has_citations and sources:
            source_labels = [f"[Source {i+1}: {src}]" for i, src in enumerate(sources)]
            return f"{cleaned}\n\nSources: {', '.join(source_labels)}"

        return cleaned
