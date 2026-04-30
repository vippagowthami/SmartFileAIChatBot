import hashlib
import math
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def vectorize_text(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)

    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(digest, 16) % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]

    return vector


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    limit = min(len(vector_a), len(vector_b))
    if limit == 0:
        return 0.0

    dot_product = sum(vector_a[index] * vector_b[index] for index in range(limit))
    magnitude_a = math.sqrt(sum(vector_a[index] * vector_a[index] for index in range(limit)))
    magnitude_b = math.sqrt(sum(vector_b[index] * vector_b[index] for index in range(limit)))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)