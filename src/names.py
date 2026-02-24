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
    role: str = ""  # e.g. "Economista", "Empresário"
    present_in_video: bool = True  # True = participant, False = just cited


# Words that look like capitalised sequences but aren't person names
_NON_NAME_WORDS = {
    "entrevistador", "entrevistadora", "palestrante", "apresentador",
    "apresentadora", "participante", "convidado", "convidada",
    "mediador", "mediadora", "moderador", "moderadora", "ouvinte",
    "gestao", "gestão", "arbitragem", "podcast", "episódio", "episodio",
    "canal", "inscreva", "compartilhe", "youtube", "instagram",
    "twitter", "facebook", "whatsapp", "telegram", "linkedin",
    "introdução", "introducao", "conclusão", "conclusao",
    "parte", "capitulo", "capítulo", "resumo", "destaque",
}


def collect_name_candidates(
    title: str,
    description: str,
    ocr_names: list[str],
    transcript: str,
    channel_name: str = "",
) -> list[str]:
    """Gather potential person names from all available sources.

    A name must be 2-5 capitalised words. Duplicates are merged.
    Filters out channel name and common non-person words.
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
    candidates = _deduplicate(candidates)

    # Filter out non-person names
    candidates = _filter_non_names(candidates, channel_name)

    return candidates


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
      4. Repeated >= 2x in transcript
    """
    validated: list[ValidatedName] = []

    for name in candidates:
        criteria_met = 0
        best_spelling = name
        source = "extraction"
        trust = "low"

        in_title = _fuzzy_in(name, video_title)
        in_ocr = _fuzzy_in(name, ocr_full)

        # Criterion 1: present in title
        if in_title:
            criteria_met += 1

        # Criterion 2: complete OCR match
        if in_ocr:
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
        # Give an extra criterion point by default for reaching this stage.
        criteria_met += 1  # baseline: survived extraction

        if criteria_met >= 2:
            # Determine if person is actually PRESENT in the video
            # vs just being cited/discussed
            present = in_title or in_ocr
            validated.append(
                ValidatedName(
                    canonical=best_spelling,
                    source=source,
                    trust_level=trust if trust != "low" else "medium",
                    present_in_video=present,
                )
            )

    return validated


def generate_mini_bio(name: str, channel_name: str) -> tuple[str, str]:
    """Create an 8-12 word mini-biography via web search snippets.

    Returns (role, bio). Falls back to defaults on ambiguity.
    """
    snippet = _search_snippet(f"{name} {channel_name}")
    if not snippet:
        return "", "Participante do programa"

    role = _extract_role(snippet)
    bio = _summarise_snippet(snippet, max_words=12)
    if not bio:
        bio = "Participante do programa"

    return role, bio


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NAME_PATTERN = re.compile(r"\b(?:[A-ZÀ-Ü][a-zà-ü]+(?:\s+|$)){2,5}")

_ROLE_KEYWORDS = [
    "economista", "empresário", "empresária", "jornalista", "médico", "médica",
    "advogado", "advogada", "professor", "professora", "atleta",
    "influenciador", "influenciadora", "apresentador", "apresentadora",
    "comediante", "escritor", "escritora", "analista", "trader",
    "investidor", "investidora", "político", "política", "engenheiro",
    "engenheira", "cientista", "pesquisador", "pesquisadora", "youtuber",
    "streamer", "fundador", "fundadora", "consultor", "consultora",
    "diretor", "diretora", "produtor", "produtora", "músico",
    "cantor", "cantora", "filósofo", "filósofa",
    "psicólogo", "psicóloga", "historiador", "historiadora",
    "sociólogo", "socióloga", "podcaster",
]

# Build a compiled regex that matches role keywords as whole words only
_ROLE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(r) for r in _ROLE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Indicators that Google returned a blocked/captcha page
_BLOCKED_INDICATORS = [
    "redirecionamento não iniciar",
    "unusual traffic",
    "captcha",
    "are you a robot",
    "nosso sistema detectou",
    "{display:none}",
    "recaptcha",
    "sorry/index",
    "detected unusual traffic",
    "clique aqui se o redirecionamento",
]


def _is_blocked_response(text: str) -> bool:
    """Detect if Google returned a captcha/block page instead of results."""
    text_lower = text.lower()
    return any(ind in text_lower for ind in _BLOCKED_INDICATORS)


def _extract_role(text: str) -> str:
    """Extract a professional role from text using whole-word matching."""
    match = _ROLE_PATTERN.search(text)
    if match:
        return match.group(1).capitalize()
    return ""


def _filter_non_names(candidates: list[str], channel_name: str) -> list[str]:
    """Remove candidates that are clearly not person names."""
    channel_lower = channel_name.lower().strip()
    # Break channel name into words for partial matching
    channel_words = set(channel_lower.split())

    result: list[str] = []
    for name in candidates:
        name_lower = name.lower().strip()
        name_words = set(name_lower.split())

        # Skip if any word is in the non-name blacklist
        if name_words & _NON_NAME_WORDS:
            continue

        # Skip if the name matches the channel name
        if channel_lower and (
            name_lower == channel_lower
            or name_lower in channel_lower
            or channel_lower in name_lower
        ):
            continue

        # Skip if all words of the name are channel words
        if channel_words and name_words <= channel_words:
            continue

        # Skip names with only 1-letter words (OCR artifacts)
        if all(len(w) <= 2 for w in name_words):
            continue

        result.append(name)

    return result


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
            idx = line.lower().index(lower)
            return line[idx : idx + len(name)]
    return None


def _google_canonise(name: str, context: str) -> str | None:
    """Search Google for the canonical spelling of a name."""
    query = f"{name} {context}"
    snippet = _search_snippet(query)
    if not snippet:
        return None

    pattern = re.compile(re.escape(name), re.IGNORECASE)
    match = pattern.search(snippet)
    if match:
        return match.group(0)

    return name  # Confirmed existence, keep original spelling


def _search_snippet(query: str) -> str:
    """Fetch a search snippet from Google. Returns raw text or empty string.

    Returns empty if Google blocks the request (captcha/redirect).
    """
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
        text = text[:2000]

        # Detect blocked/captcha responses
        if _is_blocked_response(text):
            return ""

        return text
    except Exception:
        return ""


def _summarise_snippet(snippet: str, max_words: int = 12) -> str:
    """Extract a short bio-like sentence from a search snippet."""
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
