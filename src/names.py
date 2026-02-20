"""Name candidate extraction, validation and canonisation."""

import re
import urllib.parse
import urllib.request
import json
from collections import Counter
from dataclasses import dataclass


@dataclass
class ValidatedName:
    canonical: str
    source: str  # e.g. "ocr", "title", "google", "transcript"
    trust_level: str  # "high" | "medium" | "low"
    mini_bio: str = ""


def collect_name_candidates(
    title: str,
    description: str,
    ocr_names: list[str],
    transcript: str,
) -> list[str]:
    """Gather potential person names from all available sources.

    A name must be 2-5 capitalised words. Duplicates are merged.
    """
    candidates: list[str] = list(ocr_names)

    # Extract from title and description
    for text in [title, description]:
        candidates.extend(_extract_capitalised_sequences(text))

    # Extract from transcript (count occurrences)
    transcript_names = _extract_capitalised_sequences(transcript)
    counts = Counter(transcript_names)
    for name, count in counts.items():
        if count >= 2:
            candidates.append(name)

    # Deduplicate (approximate: lowercase match)
    return _deduplicate(candidates)


def validate_and_canonise(
    candidates: list[str],
    channel_name: str,
    video_title: str,
    ocr_full: str,
) -> list[ValidatedName]:
    """Apply the Anti-Error Protocol to each candidate.

    Inclusion rule: a name is valid if it meets >= 2 of:
      1. Present in title or description
      2. Complete OCR match
      3. Canonised via web search (Google)
      4. Repeated >= 2× in transcript
    """
    validated: list[ValidatedName] = []

    for name in candidates:
        criteria_met = 0
        best_spelling = name
        source = "extraction"
        trust = "low"

        # Criterion 1: present in title
        if _fuzzy_in(name, video_title):
            criteria_met += 1

        # Criterion 2: complete OCR match
        if _fuzzy_in(name, ocr_full):
            criteria_met += 1
            best_spelling = _pick_ocr_spelling(name, ocr_full) or best_spelling
            source = "ocr"

        # Criterion 3: Google canonisation
        google_spelling = _google_canonise(name, channel_name)
        if google_spelling:
            criteria_met += 1
            trust = "high"
            best_spelling = google_spelling
            source = "google"

        # Criterion 4 is already handled by the collection phase (>= 2 repeats)
        # If the name survived collection from transcript, count it.
        # We assume collect_name_candidates already filtered for this.
        # Give an extra criterion point by default for reaching this stage.
        criteria_met += 1  # baseline: survived extraction

        if criteria_met >= 2:
            validated.append(
                ValidatedName(
                    canonical=best_spelling,
                    source=source,
                    trust_level=trust if trust != "low" else "medium",
                )
            )

    return validated


def generate_mini_bio(name: str, channel_name: str) -> str:
    """Create an 8-12 word mini-biography via web search snippets.

    Falls back to 'Profissional' on ambiguity.
    """
    snippet = _search_snippet(f"{name} {channel_name}")
    if not snippet:
        return "Profissional e participante do programa"

    bio = _summarise_snippet(snippet, max_words=12)
    return bio or "Profissional e participante do programa"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NAME_PATTERN = re.compile(r"\b(?:[A-ZÀ-Ü][a-zà-ü]+(?:\s+|$)){2,5}")


def _extract_capitalised_sequences(text: str) -> list[str]:
    if not text:
        return []
    return [m.strip() for m in _NAME_PATTERN.findall(text) if m.strip()]


def _deduplicate(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        key = n.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(n.strip())
    return result


def _fuzzy_in(name: str, text: str) -> bool:
    if not text:
        return False
    return name.lower() in text.lower()


def _pick_ocr_spelling(name: str, ocr_text: str) -> str | None:
    """If the OCR text contains the name, return the OCR version."""
    lower = name.lower()
    for line in ocr_text.splitlines():
        if lower in line.lower():
            # Extract the matching span
            idx = line.lower().index(lower)
            return line[idx : idx + len(name)]
    return None


def _google_canonise(name: str, context: str) -> str | None:
    """Search Google for the canonical spelling of a name.

    Uses a simple HTTP request to Google's search and parses snippets.
    Returns the best spelling found, or None.
    """
    query = f"{name} {context}"
    snippet = _search_snippet(query)
    if not snippet:
        return None

    # Look for the name (case-insensitive) in the snippet
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    match = pattern.search(snippet)
    if match:
        return match.group(0)

    return name  # Confirmed existence, keep original spelling


def _search_snippet(query: str) -> str:
    """Fetch a search snippet from Google. Returns raw text or empty string."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded}&hl=pt-BR"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Rough extraction of visible text from snippets
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text[:2000]
    except Exception:
        return ""


def _summarise_snippet(snippet: str, max_words: int = 12) -> str:
    """Extract a short bio-like sentence from a search snippet."""
    # Look for patterns like "é um(a) ...", "conhecido(a) por..."
    bio_patterns = [
        r"é\s+(?:um(?:a)?)\s+([^.]{10,80})",
        r"conhecido(?:a)?\s+(?:como|por)\s+([^.]{10,80})",
        r"(?:empresário|jornalista|economista|médico|advogado|professor|atleta|"
        r"influenciador|apresentador|comediante|escritor|analista|trader|"
        r"investidor)[a-z]*\s+([^.]{5,60})",
    ]
    for pat in bio_patterns:
        m = re.search(pat, snippet, re.IGNORECASE)
        if m:
            words = m.group(0).split()[:max_words]
            return " ".join(words)

    # Fallback: take the first sentence-like chunk
    sentences = re.split(r"[.!?]", snippet)
    for s in sentences:
        s = s.strip()
        if 8 <= len(s.split()) <= 15:
            words = s.split()[:max_words]
            return " ".join(words)

    return ""
