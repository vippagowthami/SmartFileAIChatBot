import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from .db import VectorDatabase
    from .llm import OllamaLLM
    from .persistence import JsonPersistence
    from .text_vectorizer import tokenize
except Exception:
    from db import VectorDatabase
    from llm import OllamaLLM
    from persistence import JsonPersistence
    from text_vectorizer import tokenize


_STOPWORDS = {
    "a", "an", "and", "are", "as", "ask", "about", "be", "by", "can", "do", "does", "for",
    "from", "give", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please", "show",
    "tell", "that", "the", "this", "to", "what", "when", "where", "which", "why", "with", "you",
    "your", "my", "we", "they", "their", "our", "who", "whom", "would", "could", "should",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_key(value: str | None, fallback: str = "default-student") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip())
    return cleaned or fallback


class StudentMemoryStore:
    """Persistent student memory backed by JSON profiles and a semantic vector store."""

    def __init__(
        self,
        llm: OllamaLLM,
        db_path: str,
        profile_path: str = "./data/student_profiles.json",
        collection_name: str = "student_memory",
    ):
        self.llm = llm
        self.db_path = Path(db_path)
        self.profile_store = JsonPersistence(profile_path, {"students": {}})
        self.memory_db = VectorDatabase(db_path=str(self.db_path), collection_name=collection_name)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recall_context(
        self,
        student_id: str | None,
        query: str,
        session_id: str | None = None,
        max_results: int = 3,
        is_follow_up: bool = False,
        follow_up_kind: str = "none",
    ) -> dict:
        student_key = self._student_key(student_id, session_id)
        profile = self._get_profile(student_key)
        query_text = (query or "").strip()
        teaching_profile = self.resolve_teaching_profile(
            student_id=student_key,
            session_id=session_id,
            question=query_text,
        )

        matches: list[dict] = []
        query_entities = self._extract_entities(query_text)
        if query_text:
            query_embedding, _ = self.llm.get_embedding(query_text)
            result = self.memory_db.search(
                query_embedding=query_embedding,
                num_results=max_results + 3,
                where={"student_id": student_key},
            )
            raw = result.get("results", [])
            # Re-rank semantic results using profile signals (weak topics, recency, weak_signal)
            matches = self._rerank_matches(
                raw,
                profile,
                query_text=query_text,
                query_entities=query_entities,
                is_follow_up=is_follow_up,
            )[:max_results]

        profile_summary = self._build_profile_summary(profile)
        conversation_state = self._build_conversation_state(
            profile,
            query_entities=query_entities,
            is_follow_up=is_follow_up,
            follow_up_kind=follow_up_kind,
        )
        recent_memory = self._build_recent_memory(profile, session_id=session_id)
        semantic_memory = self._build_semantic_memory(matches)

        context_parts = [part for part in [profile_summary, conversation_state, recent_memory, semantic_memory] if part]
        return {
            "student_id": student_key,
            "profile": profile,
            "matches": matches,
            "teaching_profile": teaching_profile,
            "context": "\n\n".join(context_parts).strip(),
        }

    def set_teaching_style_preference(
        self,
        student_id: str | None,
        teaching_style_preference: str,
        session_id: str | None = None,
    ) -> dict:
        student_key = self._student_key(student_id, session_id)
        profile = self._get_profile(student_key)
        normalized = self._normalize_teaching_style(teaching_style_preference)
        profile["teaching_style_preference"] = normalized
        profile["teaching_style_updated_at"] = _utc_now()
        self._save_profile(student_key, profile)
        return {"student_id": student_key, "teaching_style_preference": normalized}

    def set_language_preference(
        self,
        student_id: str | None,
        language_code: str,
        session_id: str | None = None,
    ) -> dict:
        """
        Set explicit language preference for student.
        
        Args:
            student_id: Student identifier
            language_code: ISO 639-1 language code (e.g., 'en', 'es', 'fr')
            session_id: Session identifier
            
        Returns:
            Dictionary with student_id and updated language preference
        """
        student_key = self._student_key(student_id, session_id)
        profile = self._get_profile(student_key)
        normalized = language_code.lower().strip()
        profile["language_preference"] = normalized
        profile["preferred_language_updated_at"] = _utc_now()
        self._save_profile(student_key, profile)
        return {
            "student_id": student_key,
            "language_preference": normalized,
            "updated_at": profile["preferred_language_updated_at"],
        }

    def get_language_preference(
        self,
        student_id: str | None,
        detected_language: str = "en",
        session_id: str | None = None,
    ) -> str:
        """
        Get effective language preference (explicit preference or detected language).
        
        Returns:
            ISO 639-1 language code
        """
        student_key = self._student_key(student_id, session_id)
        profile = self._get_profile(student_key)
        
        # Return explicit preference if set
        if profile.get("language_preference"):
            return profile["language_preference"]
        
        # Otherwise return detected language or default to English
        return (detected_language or "en").lower().strip()

    def resolve_teaching_profile(
        self,
        student_id: str | None,
        session_id: str | None = None,
        question: str | None = None,
        intent: str | None = None,
    ) -> dict:
        student_key = self._student_key(student_id, session_id)
        profile = self._get_profile(student_key)
        preference = self._normalize_teaching_style(profile.get("teaching_style_preference"))
        performance_state, performance_score = self._determine_performance_state(profile)
        teaching_style = self._select_teaching_style(
            preference,
            performance_state,
            question=question,
            intent=intent,
        )
        guidance = self._build_teaching_guidance(
            teaching_style=teaching_style,
            preference=preference,
            performance_state=performance_state,
            performance_score=performance_score,
            profile=profile,
            question=question,
            intent=intent,
        )
        resolved = {
            "student_id": student_key,
            "teaching_style_preference": preference,
            "teaching_style": teaching_style,
            "performance_state": performance_state,
            "performance_score": round(performance_score, 3),
            "guidance": guidance,
        }
        profile["performance_state"] = performance_state
        profile["performance_score"] = round(performance_score, 3)
        profile["teaching_style_resolved"] = teaching_style
        profile["teaching_style_resolution"] = resolved
        profile["teaching_style_updated_at"] = _utc_now()
        self._save_profile(student_key, profile)
        return resolved

    def record_interaction(
        self,
        *,
        student_id: str | None,
        session_id: str | None,
        question: str,
        answer: str,
        normalized_question: str | None = None,
        intent: str | None = None,
        verbosity: str | None = None,
        use_rag: bool = True,
        retrieved_documents: int = 0,
        retrieved_sources: list | None = None,
        timings: dict | None = None,
        detected_language: str | None = None,
        is_follow_up: bool = False,
    ) -> dict:
        student_key = self._student_key(student_id, session_id)
        profile = self._get_profile(student_key)
        timestamp = _utc_now()
        topics = self._extract_topics(normalized_question or question, intent)
        entities = self._extract_entities(normalized_question or question)
        profile_signals = self._extract_profile_signals(normalized_question or question)
        personal = self._extract_personal_details(question)
        primary_topic = topics[0] if topics else (intent or "general")
        weak_signal = self._is_weak_signal(answer, use_rag=use_rag, retrieved_documents=retrieved_documents)

        profile["student_id"] = student_key
        profile.setdefault("created_at", timestamp)
        profile["updated_at"] = timestamp
        profile["session_ids"] = self._append_unique(profile.get("session_ids", []), session_id)
        profile["interaction_count"] = int(profile.get("interaction_count", 0)) + 1
        profile["last_seen_at"] = timestamp
        profile["last_question"] = question
        profile["last_answer_preview"] = self._trim_text(answer, 220)
        profile["last_intent"] = intent or "other"
        profile["last_verbosity"] = verbosity or "medium"
        profile["last_follow_up"] = bool(is_follow_up)
        profile["recent_topics"] = self._append_unique(profile.get("recent_topics", []), primary_topic, limit=10)

        if personal.get("preferred_name"):
            profile["preferred_name"] = personal["preferred_name"]
            profile["preferred_name_updated_at"] = timestamp
        for fact in personal.get("facts", []):
            profile["personal_facts"] = self._append_unique(profile.get("personal_facts", []), fact, limit=12)

        # Track stable learner profile signals across sessions.
        if profile_signals.get("learning_level"):
            profile["learning_level"] = profile_signals["learning_level"]
            profile["learning_level_updated_at"] = timestamp
        for goal in profile_signals.get("goals", []):
            profile["goals"] = self._append_unique(profile.get("goals", []), goal, limit=8)
        for pref in profile_signals.get("preferences", []):
            profile["preferences"] = self._append_unique(profile.get("preferences", []), pref, limit=10)

        # Track entities for context continuity in follow-up questions.
        entity_stats = profile.get("entity_stats", {}) or {}
        for entity in entities:
            stats = entity_stats.get(entity, {})
            stats["count"] = int(stats.get("count", 0)) + 1
            stats["last_seen_at"] = timestamp
            entity_stats[entity] = stats
        profile["entity_stats"] = entity_stats
        profile["active_entities"] = self._build_active_entities(entity_stats)

        topic_stats = profile.get("topic_stats", {}) or {}
        for topic in topics or [primary_topic]:
            stats = topic_stats.get(topic, {})
            stats["asked_count"] = int(stats.get("asked_count", 0)) + 1
            stats["weakness_score"] = round(float(stats.get("weakness_score", 0.0)) + (1.0 if weak_signal else -0.1), 3)
            stats["last_seen_at"] = timestamp
            stats["last_session_id"] = session_id
            stats["examples"] = self._append_unique(stats.get("examples", []), self._trim_text(question, 120), limit=4)
            topic_stats[topic] = stats
        profile["topic_stats"] = topic_stats
        profile["weak_topics"] = self._build_weak_topics(topic_stats)
        profile["performance_state"], profile["performance_score"] = self._determine_performance_state(profile)

        # Track detected language
        if detected_language:
            detected_lang_lower = detected_language.lower().strip()
            detected_langs = profile.get("detected_languages", {}) or {}
            detected_langs[detected_lang_lower] = detected_langs.get(detected_lang_lower, 0) + 1
            profile["detected_languages"] = detected_langs

        profile.setdefault("recent_interactions", [])
        profile["recent_interactions"] = self._append_recent_interaction(
            profile["recent_interactions"],
            {
                "timestamp": timestamp,
                "session_id": session_id,
                "question": question,
                "answer_preview": self._trim_text(answer, 220),
                "topic": primary_topic,
                "intent": intent or "other",
                "weak_signal": weak_signal,
                "entities": entities,
            },
            limit=12,
        )

        self._save_profile(student_key, profile)
        memory_text = self._build_memory_text(
            question=question,
            answer=answer,
            topic=primary_topic,
            topics=topics,
            intent=intent,
            verbosity=verbosity,
            weak_signal=weak_signal,
            use_rag=use_rag,
            retrieved_documents=retrieved_documents,
            retrieved_sources=retrieved_sources or [],
            timings=timings or {},
        )
        self._store_vector_memory(student_key, session_id, memory_text, metadata={
            "student_id": student_key,
            "session_id": session_id or "",
            "topic": primary_topic,
            "intent": intent or "other",
            "weak_signal": weak_signal,
            "entities": "|".join(entities),
            "created_at": timestamp,
        })
        return {"student_id": student_key, "topic": primary_topic, "weak_signal": weak_signal}

    def get_profile(self, student_id: str | None, session_id: str | None = None) -> dict:
        student_key = self._student_key(student_id, session_id)
        return self._get_profile(student_key)

    def get_stats(self) -> dict:
        profiles = self.profile_store.read().get("students", {}) or {}
        return {
            "students": len(profiles),
            "memory_chunks": self.memory_db.document_count(),
            "storage_path": str(self.db_path),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _student_key(self, student_id: str | None, session_id: str | None = None) -> str:
        return _safe_key(student_id or session_id)

    def _get_profile(self, student_key: str) -> dict:
        data = self.profile_store.read()
        students = data.setdefault("students", {})
        profile = students.get(student_key)
        if not profile:
            profile = {
                "student_id": student_key,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "interaction_count": 0,
                "session_ids": [],
                "topic_stats": {},
                "weak_topics": [],
                "recent_topics": [],
                "recent_interactions": [],
                "language_preference": None,  # User's explicit language preference (ISO 639-1 code)
                "detected_languages": {},  # Map of detected languages and their frequency
                "preferred_language_updated_at": None,
                "learning_level": None,
                "learning_level_updated_at": None,
                "goals": [],
                "preferences": [],
                "preferred_name": None,
                "preferred_name_updated_at": None,
                "personal_facts": [],
                "entity_stats": {},
                "active_entities": [],
            }
            students[student_key] = profile
            self.profile_store.write(data)
        else:
            # Migrate old profiles that don't have language fields
            if "language_preference" not in profile:
                profile["language_preference"] = None
            if "detected_languages" not in profile:
                profile["detected_languages"] = {}
            if "preferred_language_updated_at" not in profile:
                profile["preferred_language_updated_at"] = None
            if "learning_level" not in profile:
                profile["learning_level"] = None
            if "learning_level_updated_at" not in profile:
                profile["learning_level_updated_at"] = None
            if "goals" not in profile:
                profile["goals"] = []
            if "preferences" not in profile:
                profile["preferences"] = []
            if "preferred_name" not in profile:
                profile["preferred_name"] = None
            if "preferred_name_updated_at" not in profile:
                profile["preferred_name_updated_at"] = None
            if "personal_facts" not in profile:
                profile["personal_facts"] = []
            if "entity_stats" not in profile:
                profile["entity_stats"] = {}
            if "active_entities" not in profile:
                profile["active_entities"] = []
        return profile

    def _normalize_teaching_style(self, teaching_style: str | None) -> str:
        normalized = (teaching_style or "friendly").strip().lower()
        aliases = {
            "strict": "strict",
            "firm": "strict",
            "tough": "strict",
            "friendly": "friendly",
            "supportive": "friendly",
            "kind": "friendly",
            "story": "storytelling",
            "storytelling": "storytelling",
            "narrative": "storytelling",
            "example": "storytelling",
        }
        return aliases.get(normalized, "friendly")

    def _determine_performance_state(self, profile: dict) -> tuple[str, float]:
        weak_topics = profile.get("weak_topics", []) or []
        interaction_count = int(profile.get("interaction_count", 0) or 0)
        if not weak_topics:
            if interaction_count < 3:
                return "new", 0.0
            return "steady", 0.2

        top_score = float(weak_topics[0].get("score", 0.0) or 0.0)
        weak_count = len(weak_topics)
        performance_score = min(1.0, (top_score / 4.0) + (weak_count * 0.08))

        if top_score >= 2.5 or weak_count >= 4:
            return "needs_support", performance_score
        if top_score >= 1.1 or weak_count >= 2:
            return "mixed", performance_score
        return "strong", max(0.0, performance_score - 0.15)

    def _select_teaching_style(
        self,
        preference: str,
        performance_state: str,
        *,
        question: str | None = None,
        intent: str | None = None,
    ) -> str:
        preference = self._normalize_teaching_style(preference)

        if performance_state == "needs_support":
            if preference == "strict":
                return "friendly"
            return preference

        if preference == "friendly" and performance_state == "strong":
            q = (question or "").lower()
            if intent in {"explanation", "deep_dive"} or any(token in q for token in ["explain", "how", "why", "summarize"]):
                return "storytelling"

        if preference == "storytelling" and performance_state in {"new", "mixed"}:
            return "storytelling"

        return preference

    def _build_teaching_guidance(
        self,
        *,
        teaching_style: str,
        preference: str,
        performance_state: str,
        performance_score: float,
        profile: dict,
        question: str | None = None,
        intent: str | None = None,
    ) -> str:
        weak_topics = profile.get("weak_topics", []) or []
        top_weak = ", ".join(item["topic"] for item in weak_topics[:3]) or "none"
        style_instructions = {
            "strict": (
                "Teach with precision, brevity, and rigor. State the answer directly, correct mistakes clearly, "
                "and add a short checkpoint or mini-test when useful."
            ),
            "friendly": (
                "Teach with warmth and encouragement. Keep the explanation easy to follow, avoid intimidation, "
                "and reassure the student while still being accurate."
            ),
            "storytelling": (
                "Teach through short stories, analogies, and concrete examples first, then connect them back to the formal concept."
            ),
        }
        performance_instructions = {
            "needs_support": (
                "The student is currently struggling. Slow down, use smaller steps, simplify terminology, and add one extra example."
            ),
            "mixed": (
                "The student is making partial progress. Balance clarity with challenge and briefly reinforce weak spots."
            ),
            "strong": (
                "The student appears to be doing well. You may increase challenge slightly and move faster through basics."
            ),
            "new": (
                "This is an early profile. Build trust, explain terms gently, and avoid assuming prior knowledge."
            ),
            "steady": (
                "The student is reasonably stable. Keep the pacing moderate and adapt to the question complexity."
            ),
        }

        guidance_lines = [
            f"Teaching style: {teaching_style}",
            f"Student preference: {preference}",
            f"Performance state: {performance_state} (score={round(performance_score, 3)})",
            f"Known weak topics: {top_weak}",
            f"Style guidance: {style_instructions.get(teaching_style, style_instructions['friendly'])}",
            f"Performance guidance: {performance_instructions.get(performance_state, performance_instructions['steady'])}",
        ]
        if question:
            guidance_lines.append(f"Current question: {self._trim_text(question, 160)}")
        if intent:
            guidance_lines.append(f"Current intent: {intent}")
        return "\n".join(guidance_lines)

    def _rerank_matches(
        self,
        matches: list[dict],
        profile: dict,
        *,
        query_text: str,
        query_entities: list[str],
        is_follow_up: bool,
    ) -> list[dict]:
        """Re-rank semantic matches by combining vector similarity (distance) with
        profile-aware boosts: weak topics, weak_signal flags, and recency.

        The VectorDatabase returns a "distance" where lower is better. We convert
        it to a base score (1 - distance) and apply additive boosts.
        """
        if not matches:
            return []

        now = datetime.now(timezone.utc)
        weak_topics = {item.get("topic") for item in (profile.get("weak_topics") or [])}
        goal_tokens = set(tokenize(" ".join(profile.get("goals", []) or [])))

        reranked: list[dict] = []
        for m in matches:
            meta = (m.get("metadata") or {})
            dist = float(m.get("distance", 1.0) or 1.0)
            base_score = max(0.0, 1.0 - dist)

            # Boost if memory topic matches known weak topics
            topic = (meta.get("topic") or "").strip()
            boost = 0.0
            if topic and topic in weak_topics:
                boost += 0.25

            # Boost if memory was marked as a weak signal
            if meta.get("weak_signal"):
                boost += 0.18

            # Boost memories that overlap with detected entities.
            memory_entities = str(meta.get("entities") or "")
            memory_entity_set = {part.strip() for part in memory_entities.split("|") if part.strip()}
            entity_overlap = len(memory_entity_set.intersection(set(query_entities)))
            if entity_overlap:
                boost += min(0.24, 0.08 * entity_overlap)

            # Boost memories aligned with student goals.
            if goal_tokens:
                memory_tokens = set(tokenize((m.get("text") or "")[:320]))
                overlap = len(goal_tokens.intersection(memory_tokens))
                if overlap:
                    boost += min(0.15, 0.04 * overlap)

            # For follow-up turns, prioritize very recent memories.
            if is_follow_up:
                boost += 0.08

            # Small recency boost (favor memories within last 7 days)
            created = None
            try:
                created = meta.get("created_at")
                if created:
                    created_dt = datetime.fromisoformat(created)
                    # ensure tz-aware
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    age_seconds = (now - created_dt).total_seconds()
                    week = 60 * 60 * 24 * 7
                    recency_factor = max(0.0, 1.0 - (age_seconds / week))
                    boost += 0.12 * recency_factor
            except Exception:
                pass

            score = min(1.0, base_score + boost)
            m_copy = dict(m)
            m_copy["score"] = round(score, 4)
            reranked.append(m_copy)

        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked

    def _save_profile(self, student_key: str, profile: dict) -> None:
        data = self.profile_store.read()
        students = data.setdefault("students", {})
        students[student_key] = profile
        self.profile_store.write(data)

    def _build_profile_summary(self, profile: dict) -> str:
        weak_topics = profile.get("weak_topics", []) or []
        recent_topics = profile.get("recent_topics", []) or []
        topic_count = len(profile.get("topic_stats", {}) or {})
        lines = [
            "STUDENT PROFILE:",
            f"- Total interactions: {profile.get('interaction_count', 0)}",
            f"- Active sessions: {len(profile.get('session_ids', []) or [])}",
            f"- Topics tracked: {topic_count}",
        ]
        if profile.get("preferred_name"):
            lines.append(f"- Name: {profile.get('preferred_name')}")
        if recent_topics:
            lines.append(f"- Recent topics: {', '.join(recent_topics[-5:])}")
        if weak_topics:
            weak_preview = ", ".join(item['topic'] for item in weak_topics[:4])
            lines.append(f"- Weak topics: {weak_preview}")
        if profile.get("learning_level"):
            lines.append(f"- Learning level: {profile.get('learning_level')}")
        if profile.get("goals"):
            lines.append(f"- Goals: {', '.join((profile.get('goals') or [])[:4])}")
        if profile.get("preferences"):
            lines.append(f"- Preferences: {', '.join((profile.get('preferences') or [])[:4])}")
        if profile.get("personal_facts"):
            lines.append(f"- Personal details: {', '.join((profile.get('personal_facts') or [])[:4])}")
        last_question = profile.get("last_question")
        if last_question:
            lines.append(f"- Last question: {self._trim_text(last_question, 140)}")
        return "\n".join(lines)

    def _build_conversation_state(
        self,
        profile: dict,
        *,
        query_entities: list[str],
        is_follow_up: bool,
        follow_up_kind: str,
    ) -> str:
        active_entities = profile.get("active_entities", []) or []
        if not active_entities and not is_follow_up:
            return ""

        lines = ["CONVERSATION STATE:"]
        if is_follow_up:
            lines.append(f"- Follow-up detected: yes ({follow_up_kind})")
        else:
            lines.append("- Follow-up detected: no")
        if query_entities:
            lines.append(f"- Current entities: {', '.join(query_entities[:6])}")
        if active_entities:
            lines.append(f"- Active entities from prior chats: {', '.join(active_entities[:8])}")
        return "\n".join(lines)

    def _build_recent_memory(self, profile: dict, session_id: str | None = None) -> str:
        recent = profile.get("recent_interactions", []) or []
        if session_id:
            filtered = [item for item in recent if item.get("session_id") == session_id]
            if filtered:
                recent = filtered
        recent = recent[-4:]
        if not recent:
            return ""

        lines = ["RECENT INTERACTIONS:"]
        for item in recent:
            lines.append(
                f"- {item.get('timestamp', '')}: Q={self._trim_text(item.get('question', ''), 100)} | A={self._trim_text(item.get('answer_preview', ''), 100)}"
            )
        return "\n".join(lines)

    def _build_semantic_memory(self, matches: list[dict]) -> str:
        if not matches:
            return ""

        lines = ["RELEVANT PAST MEMORY:"]
        for index, match in enumerate(matches, 1):
            meta = match.get("metadata", {}) or {}
            text = self._trim_text(match.get("text", ""), 220)
            topic = meta.get("topic") or "general"
            lines.append(f"- Memory {index} [{topic}]: {text}")
        return "\n".join(lines)

    def _build_memory_text(
        self,
        *,
        question: str,
        answer: str,
        topic: str,
        topics: list[str],
        intent: str | None,
        verbosity: str | None,
        weak_signal: bool,
        use_rag: bool,
        retrieved_documents: int,
        retrieved_sources: list,
        timings: dict,
    ) -> str:
        source_names = [str(src.get("source", "")) for src in retrieved_sources if isinstance(src, dict)]
        topic_text = ", ".join(dict.fromkeys(topics)) if topics else topic
        return (
            f"Student question: {question}\n"
            f"Assistant answer: {self._trim_text(answer, 600)}\n"
            f"Topic: {topic_text}\n"
            f"Primary topic: {topic}\n"
            f"Intent: {intent or 'other'}\n"
            f"Verbosity: {verbosity or 'medium'}\n"
            f"Used RAG: {use_rag} | Retrieved documents: {retrieved_documents} | Weak signal: {weak_signal}\n"
            f"Retrieved sources: {', '.join(source_names) if source_names else 'none'}\n"
            f"Timings: {timings or {}}"
        )

    def _store_vector_memory(self, student_key: str, session_id: str | None, text: str, metadata: dict) -> None:
        if not text.strip():
            return
        embedding, _ = self.llm.get_embedding(text)
        self.memory_db.add_documents(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())],
        )

    def _extract_topics(self, text: str, intent: str | None) -> list[str]:
        tokens = [token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 2]
        if not tokens:
            return [intent] if intent else ["general"]

        topics: list[str] = []
        for token in tokens:
            if token not in topics:
                topics.append(token)
            if len(topics) >= 3:
                break

        if len(topics) >= 2:
            topics.insert(0, " ".join(topics[:2]))

        return list(dict.fromkeys([topic for topic in topics if topic]))

    def _extract_entities(self, text: str) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        # Quoted phrases and title-like tokens are strong entity candidates.
        quoted = re.findall(r"['\"]([^'\"]{2,40})['\"]", raw)
        titled = re.findall(r"\b[A-Z][a-zA-Z0-9_+#-]{1,30}\b", raw)
        tokens = [t for t in tokenize(raw) if t not in _STOPWORDS and len(t) >= 4]

        entities: list[str] = []
        for item in quoted + titled + tokens:
            cleaned = re.sub(r"\s+", " ", str(item)).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in _STOPWORDS:
                continue
            if lowered not in [e.lower() for e in entities]:
                entities.append(cleaned)
            if len(entities) >= 8:
                break
        return entities

    def _extract_profile_signals(self, text: str) -> dict:
        q = (text or "").strip().lower()
        if not q:
            return {"learning_level": None, "goals": [], "preferences": []}

        learning_level = None
        if any(k in q for k in ["beginner", "new to", "starting", "basic level"]):
            learning_level = "beginner"
        elif any(k in q for k in ["intermediate", "mid level"]):
            learning_level = "intermediate"
        elif any(k in q for k in ["advanced", "expert", "deep technical"]):
            learning_level = "advanced"

        goals: list[str] = []
        goal_patterns = [
            r"\bi want to\s+([^\.!\?]{3,120})",
            r"\bmy goal is to\s+([^\.!\?]{3,120})",
            r"\bi need to\s+([^\.!\?]{3,120})",
            r"\bhelp me\s+([^\.!\?]{3,120})",
        ]
        for pat in goal_patterns:
            for match in re.findall(pat, q):
                goals.append(self._trim_text(match.strip(), 80))

        preferences: list[str] = []
        if any(k in q for k in ["short answer", "brief", "concise"]):
            preferences.append("prefers concise answers")
        if any(k in q for k in ["detailed", "step by step", "in detail"]):
            preferences.append("prefers detailed explanations")
        if any(k in q for k in ["example", "real world"]):
            preferences.append("prefers examples")

        return {
            "learning_level": learning_level,
            "goals": list(dict.fromkeys(goals))[:3],
            "preferences": list(dict.fromkeys(preferences))[:3],
        }

    def _build_active_entities(self, entity_stats: dict) -> list[str]:
        ranked: list[tuple[str, int, str]] = []
        for entity, stats in (entity_stats or {}).items():
            ranked.append((
                entity,
                int(stats.get("count", 0) or 0),
                str(stats.get("last_seen_at", "")),
            ))
        ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return [item[0] for item in ranked[:12]]

    def _extract_personal_details(self, text: str) -> dict:
        raw = (text or "").strip()
        lowered = raw.lower()
        result = {"preferred_name": None, "facts": []}

        patterns = [
            r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,29})",
            r"\bi am\s+([A-Za-z][A-Za-z\-']{1,29})",
            r"\bcall me\s+([A-Za-z][A-Za-z\-']{1,29})",
        ]
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                bad = {"fine", "okay", "ok", "ready", "learning"}
                if candidate.lower() not in bad:
                    result["preferred_name"] = candidate.title()
                    break

        fact_patterns = [
            r"\bi live in\s+([^\.!\?]{2,80})",
            r"\bi study\s+([^\.!\?]{2,80})",
            r"\bi like\s+([^\.!\?]{2,80})",
            r"\bi prefer\s+([^\.!\?]{2,80})",
        ]
        for pat in fact_patterns:
            for match in re.findall(pat, lowered):
                cleaned = self._trim_text(match.strip(), 70)
                if cleaned:
                    result["facts"].append(cleaned)

        result["facts"] = list(dict.fromkeys(result["facts"]))[:3]
        return result

    def _build_weak_topics(self, topic_stats: dict) -> list[dict]:
        ranked = []
        for topic, stats in topic_stats.items():
            score = float(stats.get("weakness_score", 0.0))
            if score <= 0:
                continue
            ranked.append(
                {
                    "topic": topic,
                    "score": round(score, 3),
                    "asked_count": int(stats.get("asked_count", 0)),
                    "last_seen_at": stats.get("last_seen_at"),
                }
            )
        ranked.sort(key=lambda item: (item["score"], item["asked_count"]), reverse=True)
        return ranked[:10]

    def _is_weak_signal(self, answer: str, *, use_rag: bool, retrieved_documents: int) -> bool:
        if use_rag and retrieved_documents == 0:
            return True
        lowered = (answer or "").lower()
        weak_markers = (
            "i'm sorry",
            "provided documents do not contain",
            "don't know",
            "do not know",
            "not sure",
            "i am not sure",
            "cannot answer",
            "unable to",
            "no information",
        )
        return any(marker in lowered for marker in weak_markers)

    def _append_unique(self, items: list, value, limit: int = 5) -> list:
        values = [item for item in items if item not in (None, "")]
        if value and value not in values:
            values.append(value)
        return values[-limit:]

    def _append_recent_interaction(self, items: list[dict], value: dict, limit: int = 12) -> list[dict]:
        values = [item for item in items if isinstance(item, dict)]
        values.append(value)
        return values[-limit:]

    def _trim_text(self, text: str, max_length: int) -> str:
        value = re.sub(r"\s+", " ", (text or "")).strip()
        if len(value) <= max_length:
            return value
        return value[: max_length - 1].rstrip() + "…"
