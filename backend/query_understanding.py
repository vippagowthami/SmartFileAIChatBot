import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryUnderstanding:
    improved_query: str
    intent: str  # definition | explanation | comparison | list | deep_dive | other
    verbosity: str  # concise | medium | detailed


_MULTISPACE = re.compile(r"\s+")


def _basic_normalize(text: str) -> str:
    t = (text or "").strip()
    t = _MULTISPACE.sub(" ", t)
    return t


def normalize_query(user_input: str) -> str:
    """
    Heuristic-only normalization (fast, offline).
    We keep it conservative: fix whitespace, common SMS typos, and expand a few common abbreviations.
    """
    q = _basic_normalize(user_input)
    if not q:
        return q

    lower = q.lower()

    # Common quick typos / shorthand
    replacements = {
        r"\bwat\b": "what",
        r"\bwht\b": "what",
        r"\bwhats\b": "what is",
        r"\bpls\b": "please",
        r"\bplz\b": "please",
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\boops\b": "oop",
        r"\bml\b": "machine learning",
        r"\bnlp\b": "natural language processing",
        r"\bai\b": "artificial intelligence",
    }
    normalized = lower
    for pat, rep in replacements.items():
        normalized = re.sub(pat, rep, normalized, flags=re.IGNORECASE)

    # If the input is a casual greeting/thank-you/bye, do not turn it into a definition.
    casual_tokens = {"hi", "hello", "hey", "heyy", "hiya", "thanks", "thank", "thankyou", "thx", "ty", "bye", "goodbye", "yaar", "buddy", "mate", "dude", "bruh"}
    tokens = [t for t in re.split(r"\s+", normalized) if t]
    if tokens and all(t in casual_tokens for t in tokens):
        # Return normalized but don't wrap it into a 'what is' question
        return normalized

    # Expand "corba adv" / "corba advantages"
    normalized = re.sub(r"\bcorba\s+adv\b", "advantages of corba", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\badv\s+of\s+corba\b", "advantages of corba", normalized, flags=re.IGNORECASE)

    # If the user gives a short topic term, turn it into a definition question.
    # Keep acknowledgments and ultra-short fillers like "yes", "ok", "k", "what", "y", or "uhmm" from becoming fake definitions.
    short_non_topics = {
        "y", "k", "ok", "okay", "yes", "no", "nah", "nope", "sure", "fine", "cool", "alright",
        "what", "why", "how", "uhm", "uhmm", "um", "hmm", "hm", "idk", "okayyy", "okayy",
    }
    if re.fullmatch(r"[a-z0-9][a-z0-9 \-_/]{0,40}", normalized) and len(normalized.split()) <= 3:
        tokens = normalized.split()
        if len(tokens) == 1 and (len(normalized) <= 2 or normalized in short_non_topics):
            return q
        if normalized in short_non_topics:
            return q
        if not normalized.startswith(("what ", "define ", "explain ", "why ", "how ")):
            normalized = f"what is {normalized}?"

    # Restore capitalization lightly (first letter) while keeping user text
    improved = normalized.strip()
    if improved and improved[0].isalpha():
        improved = improved[0].upper() + improved[1:]
    return improved


def detect_intent(query: str) -> str:
    q = _basic_normalize(query).lower()
    if not q:
        return "other"

    if any(k in q for k in [" vs ", "versus", "difference between", "compare "]):
        return "comparison"
    if q.startswith(("define ", "definition of ", "what is ", "what are ")):
        return "definition"
    if any(k in q for k in ["list ", "types of", "examples of", "advantages", "disadvantages", "pros", "cons"]):
        return "list"
    if any(k in q for k in ["how does", "how do", "why does", "why do", "architecture", "internals", "deep dive", "under the hood", "implement"]):
        return "deep_dive"
    if any(k in q for k in ["explain", "why", "how", "describe", "overview"]):
        return "explanation"
    return "other"


def choose_verbosity(query: str, intent: str) -> str:
    q = _basic_normalize(query).lower()
    # Explicit user hints
    if any(k in q for k in ["in detail", "detailed", "deep", "step by step", "elaborate"]):
        return "detailed"
    if any(k in q for k in ["brief", "short", "quick", "in short", "tldr"]):
        return "concise"

    # Default by intent
    if intent in {"definition"}:
        return "concise"
    if intent in {"comparison", "list", "explanation"}:
        return "medium"
    if intent in {"deep_dive"}:
        return "detailed"
    return "medium"


def understand_query(user_input: str) -> QueryUnderstanding:
    improved = normalize_query(user_input)
    intent = detect_intent(improved)
    verbosity = choose_verbosity(improved, intent)
    return QueryUnderstanding(improved_query=improved, intent=intent, verbosity=verbosity)

