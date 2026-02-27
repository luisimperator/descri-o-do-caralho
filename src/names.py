"""Name candidate extraction, validation and canonisation."""

import logging
import os
import re
import urllib.parse
import urllib.request
import json
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    description: str = "",
) -> list[ValidatedName]:
    """Apply the Anti-Error Protocol to each candidate.

    Inclusion rule: a name is valid if it meets >= 2 of:
      1. Present in title or description
      2. Complete OCR match
      3. Canonised via web search (Google)
      4. Repeated >= 2x in transcript

    present_in_video is True if the name appears in title, description,
    or OCR (thumbnail). Names only from transcript are considered "cited".
    """
    validated: list[ValidatedName] = []

    for name in candidates:
        criteria_met = 0
        best_spelling = name
        source = "extraction"
        trust = "low"

        in_title = _fuzzy_in(name, video_title)
        in_ocr = _fuzzy_in(name, ocr_full)
        in_description = _fuzzy_in(name, description)

        # Criterion 1: present in title or description
        if in_title or in_description:
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
            # Present if in title, description, or thumbnail (OCR).
            # Names ONLY from transcript = cited, not present.
            present = in_title or in_ocr or in_description
            validated.append(
                ValidatedName(
                    canonical=best_spelling,
                    source=source,
                    trust_level=trust if trust != "low" else "medium",
                    present_in_video=present,
                )
            )

    return validated


def extract_role_from_description(name: str, description: str) -> tuple[str, str]:
    """Try to extract a participant's role from the video description.

    Looks for patterns like:
      - "João Manoel, Advogado"
      - "João Manoel - Advogado e Árbitro"
      - "Advogado João Manoel"
      - "João Manoel (Advogado)"
      - "João Manoel | Advogado"

    Returns (role, bio) or ("", "") if nothing found.
    """
    if not description or not name:
        return "", ""

    first_name = name.split()[0]
    last_name = name.split()[-1] if len(name.split()) > 1 else ""
    name_esc = re.escape(name)
    first_esc = re.escape(first_name)

    # Build role alternatives
    role_alts = "|".join(re.escape(r) for r in _ROLE_KEYWORDS)
    role_re = r"(?:" + role_alts + r")[a-zà-ü]*"

    patterns = [
        # "Name, Role context" / "Name - Role context" / "Name | Role"
        name_esc + r"\s*[,\-–—\|:]\s*(" + role_re + r"(?:\s+[^.\n,]{2,40})?)",
        # "Name (Role context)"
        name_esc + r"\s*\(\s*(" + role_re + r"[^)]{0,40})\)",
        # "Role Name" (e.g. "Advogado João Manoel")
        r"(" + role_re + r")\s+" + name_esc,
        # "Role em/de/... Name" (e.g. "especialista em mediação Ana Rocha")
        r"(" + role_re + r"(?:\s+(?:em|de|do|da|no|na|dos|das|para)\s+[a-zà-üA-ZÀ-Ü]+){0,3})\s+" + name_esc,
        # First name variants: "João, Advogado"
        first_esc + r"\s*[,\-–—\|:]\s*(" + role_re + r"(?:\s+[^.\n,]{2,30})?)",
        # "Role FirstName"
        r"(" + role_re + r")\s+" + first_esc,
        # "Role em/de/... FirstName"
        r"(" + role_re + r"(?:\s+(?:em|de|do|da|no|na|dos|das|para)\s+[a-zà-üA-ZÀ-Ü]+){0,3})\s+" + first_esc,
    ]

    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m:
            match_text = m.group(1).strip()
            role = _extract_role(match_text) or match_text.split()[0].capitalize()
            bio = match_text if len(match_text.split()) > 1 else ""
            words = bio.split()[:12]
            bio = " ".join(words) if words else ""
            logger.info("Role from description for '%s': role=%s, bio=%s",
                        name, role, bio)
            return role, bio

    return "", ""


def generate_mini_bio(
    name: str,
    channel_name: str,
    description: str = "",
) -> tuple[str, str]:
    """Create an 8-12 word mini-biography via Google search.

    Uses Google Custom Search JSON API (GOOGLE_CSE_ID + GOOGLE_API_KEY).
    Falls back to DuckDuckGo and description parsing if not configured.

    Returns (role, bio). Falls back to defaults on failure.
    """
    # 1. Google Custom Search API (official, works from servers)
    snippet = _search_google_api(f"{name} {channel_name}")
    if snippet:
        role = _extract_role(snippet)
        bio = _summarise_snippet(snippet, max_words=12)
        if role or bio:
            logger.info("Google API bio for '%s': role=%s", name, role)
            return role, bio or "Participante do programa"

    # 2. DuckDuckGo web search (fallback, no API key)
    snippet = _search_duckduckgo(f"{name} {channel_name}")
    if snippet:
        role = _extract_role(snippet)
        bio = _summarise_snippet(snippet, max_words=12)
        if role or bio:
            logger.info("DuckDuckGo bio for '%s': role=%s", name, role)
            return role, bio or "Participante do programa"

    # 3. Extract from video description (offline)
    if description:
        role, bio = extract_role_from_description(name, description)
        if role:
            return role, bio or "Participante do programa"

    return "", "Participante do programa"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NAME_PATTERN = re.compile(r"\b(?:[A-ZÀ-Ü][a-zà-ü]+(?:\s+|$)){2,5}")

_ROLE_KEYWORDS = [
    # Jurídico / Arbitragem
    "advogado", "advogada", "advogada criminalista", "advogado criminalista",
    "árbitro", "árbitra", "arbitralista",
    "mediador", "mediadora",
    "juiz", "juíza", "juiz federal", "juiz de direito",
    "desembargador", "desembargadora",
    "procurador", "procuradora",
    "promotor", "promotora",
    "magistrado", "magistrada",
    "defensor", "defensora", "defensor público", "defensora pública",
    "jurista", "constitucionalista",
    "delegado", "delegada",
    "notário", "tabelião", "tabeliã",
    "perito", "perita",
    # Acadêmico
    "professor", "professora", "prof.",
    "doutor", "doutora", "dr.", "dra.",
    "mestre", "especialista",
    "pesquisador", "pesquisadora",
    "cientista", "reitor", "reitora",
    # Negócios / Economia
    "economista", "empresário", "empresária",
    "consultor", "consultora",
    "diretor", "diretora",
    "fundador", "fundadora",
    "CEO", "CFO", "COO", "CTO",
    "sócio", "sócia", "sócio-fundador", "sócia-fundadora",
    "gestor", "gestora",
    "investidor", "investidora",
    "analista", "trader",
    "contador", "contadora",
    "administrador", "administradora",
    "empreendedor", "empreendedora",
    # Mídia / Entretenimento
    "jornalista", "apresentador", "apresentadora",
    "influenciador", "influenciadora",
    "comediante", "escritor", "escritora",
    "produtor", "produtora",
    "músico", "cantor", "cantora",
    "youtuber", "streamer", "podcaster",
    # Saúde
    "médico", "médica", "psicólogo", "psicóloga",
    "psiquiatra", "terapeuta", "nutricionista",
    # Engenharia / TI
    "engenheiro", "engenheira",
    "arquiteto", "arquiteta",
    "programador", "programadora", "desenvolvedor", "desenvolvedora",
    # Outros
    "atleta", "político", "política",
    "filósofo", "filósofa",
    "historiador", "historiadora",
    "sociólogo", "socióloga",
    "militar", "coronel", "general",
]

# Build a compiled regex that matches role keywords as whole words only
_ROLE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(r) for r in _ROLE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _extract_role(text: str) -> str:
    """Extract a professional role from text using whole-word matching."""
    match = _ROLE_PATTERN.search(text)
    if match:
        role = match.group(1).strip()
        # Capitalize first letter, keep rest (handles "CEO", "Dr.", etc.)
        if role[0].islower():
            role = role.capitalize()
        return role
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
    snippet = _search_google_api(f"{name} {context}")
    if not snippet:
        snippet = _search_duckduckgo(f"{name} {context}")
    if not snippet:
        return None

    pattern = re.compile(re.escape(name), re.IGNORECASE)
    match = pattern.search(snippet)
    if match:
        return match.group(0)

    return name  # Confirmed existence, keep original spelling


def _search_google_api(query: str) -> str:
    """Search Google via Custom Search JSON API.

    Requires two env vars:
      - GOOGLE_API_KEY: API key with Custom Search API enabled
      - GOOGLE_CSE_ID:  Programmable Search Engine ID (cx)

    Free tier: 100 queries/day. Works from any server, no captcha.

    Setup:
      1. console.cloud.google.com → enable "Custom Search API"
      2. programmablesearchengine.google.com → create engine
         (add sites like wikipedia.org, linkedin.com, etc.)
      3. Copy the Search engine ID → set as GOOGLE_CSE_ID in Railway
      4. GOOGLE_API_KEY falls back to YOUTUBE_API_KEY automatically
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        # Fall back to YOUTUBE_API_KEY — same Google Cloud project
        api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    cse_id = os.environ.get("GOOGLE_CSE_ID", "").strip()

    if not api_key:
        logger.warning("Google Custom Search: sem API key (GOOGLE_API_KEY nem YOUTUBE_API_KEY)")
        return ""
    if not cse_id:
        logger.warning("Google Custom Search: GOOGLE_CSE_ID não configurado")
        return ""

    logger.info("Google API: buscando '%s' (key=%s... cx=%s...)",
                query, api_key[:8], cse_id[:8])

    params = urllib.parse.urlencode({
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": "5",
        "lr": "lang_pt",
    })
    url = f"https://www.googleapis.com/customsearch/v1?{params}"

    req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items = data.get("items", [])
        if not items:
            logger.debug("Google API: sem resultados para '%s'", query)
            return ""

        # Combine titles and snippets from results
        texts = []
        for item in items:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title:
                texts.append(title)
            if snippet:
                texts.append(snippet)

        result = " ".join(texts)
        logger.info("Google API OK para '%s': %d chars de %d resultados",
                     query, len(result), len(items))
        return result[:3000]

    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        logger.warning("Google API falhou para '%s': HTTP %d — %s",
                        query, exc.code, error_body[:150])
        return ""
    except Exception as exc:
        logger.warning("Google API falhou para '%s': %s", query, exc)
        return ""


def _search_duckduckgo(query: str) -> str:
    """Search DuckDuckGo HTML endpoint. Works from datacenters."""
    url = "https://html.duckduckgo.com/html/"
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract result snippets (class="result__snippet")
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL
        )
        if not snippets:
            # Fallback: try extracting from result bodies
            snippets = re.findall(
                r'class="result__body"[^>]*>(.*?)</div>', html, re.DOTALL
            )

        if not snippets:
            logger.debug("DuckDuckGo: nenhum snippet para '%s'", query)
            return ""

        # Clean HTML tags and join snippets
        texts = []
        for s in snippets[:5]:
            clean = re.sub(r"<[^>]+>", " ", s)
            clean = re.sub(r"\s+", " ", clean).strip()
            if clean:
                texts.append(clean)

        result = " ".join(texts)
        logger.info("DuckDuckGo OK para '%s': %d chars", query, len(result))
        return result[:2000]

    except Exception as exc:
        logger.debug("DuckDuckGo falhou para '%s': %s", query, exc)
        return ""




def _summarise_snippet(snippet: str, max_words: int = 12) -> str:
    """Extract a short bio-like sentence from a search snippet."""
    # Build role alternatives from the keywords list
    role_alts = "|".join(re.escape(r) for r in _ROLE_KEYWORDS)

    bio_patterns = [
        # "é advogado/árbitro/etc"
        r"é\s+(?:um(?:a)?\s+)?(?:" + role_alts + r")[a-zà-ü]*(?:\s+[^.]{3,60})?",
        # "é um(a) profissional"
        r"é\s+(?:um(?:a)?)\s+([^.]{10,80})",
        # "atua como advogado"
        r"atua(?:ndo)?\s+(?:como|na|no|em)\s+[^.]{5,60}",
        # "conhecido(a) como/por"
        r"conhecido(?:a)?\s+(?:como|por)\s+([^.]{10,80})",
        # "especialista em / especializado em"
        r"especialista\s+em\s+[^.]{5,60}",
        r"especializado(?:a)?\s+em\s+[^.]{5,60}",
        # Role keyword followed by context
        r"(?:" + role_alts + r")[a-zà-ü]*\s+(?:e\s+)?(?:especialista|especializado)?[^.]{3,60}",
    ]
    for pat in bio_patterns:
        m = re.search(pat, snippet, re.IGNORECASE)
        if m:
            text = m.group(0).strip()
            words = text.split()[:max_words]
            return " ".join(words)

    # Fallback: take the first sentence-like chunk
    sentences = re.split(r"[.!?]", snippet)
    for s in sentences:
        s = s.strip()
        if 5 <= len(s.split()) <= 15:
            words = s.split()[:max_words]
            return " ".join(words)

    return ""
