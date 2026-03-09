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

# Track search errors for diagnostics
_search_errors: list[str] = []

# Cache search results to avoid duplicate API calls
_search_cache: dict[str, str] = {}

# Flag to skip Google API after first fatal error (403, etc.)
_google_api_disabled = False


def get_search_errors() -> list[str]:
    """Return and clear recent search errors for diagnostics."""
    errors = list(_search_errors)
    _search_errors.clear()
    _search_cache.clear()
    return errors


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
    Handles Brazilian composite names (e.g. "João Manoel Lima Junior").
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

    # Merge split names: "João Manoel" + "Lima Junior" → "João Manoel Lima Junior"
    # Check all source texts for the combined form
    all_texts = "\n".join([title, description, "\n".join(ocr_names), transcript])
    candidates = _merge_split_names(candidates, all_texts)

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
    """Create participant info: cargo/empresa + mini bio.

    Returns (role, bio) where:
      - role = "Cargo, Empresa" (e.g. "Sócio, FGBS Advocacia")
      - bio = mini description (e.g. "Advogada com experiência em compliance")

    Falls back to defaults on failure.
    """
    # Use cached unified search (same call used by canonise)
    snippet = _web_search_all(name, channel_name)
    if snippet:
        position = _extract_position_company(snippet, name)
        bio = _summarise_snippet(snippet, max_words=12)
        if position or bio:
            role = position or _extract_role(snippet) or "Participante"
            # Avoid duplicating position in bio
            if bio and role and _texts_overlap(bio, role):
                bio = _summarise_snippet_excluding(snippet, role, max_words=12)
            logger.info("Web bio for '%s': role=%s, bio=%s", name, role, bio[:50] if bio else "")
            return role, bio

    # Fallback: extract from video description (offline, no network)
    if description:
        role, bio = extract_role_from_description(name, description)
        if role:
            return role, bio

    return "Participante", ""


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


def _extract_position_company(snippet: str, name: str) -> str:
    """Extract position + company from search snippet.

    Looks for LinkedIn-style patterns:
      - "Fulano - Sócio na Empresa X"
      - "Fulano | Consultora Jurídica at FGBS"
      - "Title · Company Name"
      - "Role em/na/no Empresa"
      - "Sócio-fundador da Empresa X"

    Returns e.g. "Consultora Jurídica, FGBS Advocacia" or "".
    """
    if not snippet:
        return ""

    first_name = name.split()[0] if name else ""
    first_esc = re.escape(first_name) if first_name else ""

    # LinkedIn title patterns from Google snippets
    # e.g. "Fernanda Barjud, CCA IBGC, MCIArb - Consultora Jurídica"
    # e.g. "Lima Junior | Advogado at Escritório X"
    name_esc = re.escape(name)
    linkedin_patterns = [
        # "Name, Certs - Title" (LinkedIn: "Fernanda Barjud, CCA IBGC, MCIArb - Consultora Jurídica")
        # Grab the part AFTER the last dash/pipe before period
        name_esc + r"[^.\n]*?\s*[-–—]\s*([A-ZÀ-Ü][^.\n]{3,50})",
        # "Name - Title" or "Name | Title" (stop at period or LinkedIn)
        name_esc + r"\s*[-–—\|]\s*([^.\n]+?)(?:\.|$|\s*[·\|]\s*LinkedIn|\s*[-–—]\s*LinkedIn)",
        # "Name · Company | LinkedIn" (Google snippet format)
        name_esc + r"\s*·\s*([^.\n]+?)(?:\.|$|\s*[-–—]\s*LinkedIn)",
        # FirstName ... - Title (stop at period)
        first_esc + r"[^.\n]{0,40}?\s*[-–—]\s*([A-ZÀ-Ü][^.\n]{5,50})" if first_esc else None,
    ]

    for pat in linkedin_patterns:
        if not pat:
            continue
        m = re.search(pat, snippet, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            raw = _clean_bio_text(raw)
            # Remove "LinkedIn" if it slipped through
            raw = re.sub(r"\bLinkedIn\b", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"[\s,\-–—:;|]+$", "", raw)
            # Normalize English prepositions to Portuguese
            raw = re.sub(r"\bat\b", "no", raw, count=1)
            # Reject YouTube UI junk
            if _is_youtube_ui_junk(raw):
                continue
            # Reject if contains country name as pseudo-company
            # e.g. "Diretora, BRASIL, OAB-SP" — BRASIL is not a company
            if re.search(r"\bBRASIL\b|\bBRAZIL\b|\bPORTUGAL\b", raw, re.IGNORECASE):
                # Strip the country and OAB stuff, keep just the role
                role_only = _extract_role(raw)
                if role_only:
                    raw = role_only
                else:
                    continue
            # Limit to ~8 words max for a clean position title
            words = raw.split()
            if words and len(words) <= 10:
                logger.info("LinkedIn position for '%s': %s", name, raw)
                return raw
            elif words:
                truncated = " ".join(words[:8])
                logger.info("LinkedIn position for '%s' (truncated): %s", name, truncated)
                return truncated

    # Patterns: "role na/no/da/do Company" or "role, Company"
    role_alts = "|".join(re.escape(r) for r in _ROLE_KEYWORDS)
    company_patterns = [
        # "Sócio na Empresa X" / "Advogada no Escritório Y"
        r"(?:" + role_alts + r")[a-zà-ü]*(?:\s*[-–—]\s*|\s+)(?:na|no|da|do|em|at)\s+([A-ZÀ-Ü][^\n,.]{2,40})",
        # "Sócio, Empresa X"
        r"(?:" + role_alts + r")[a-zà-ü]*\s*,\s*([A-ZÀ-Ü][^\n,.]{2,40})",
        # "Sócio-fundador(a) da Empresa"
        r"(?:" + role_alts + r")[a-zà-ü]*(?:-[a-zà-ü]+)?\s+(?:da|do|de|na|no)\s+([A-ZÀ-Ü][^\n,.]{2,40})",
    ]

    for pat in company_patterns:
        m = re.search(pat, snippet, re.IGNORECASE)
        if m:
            company = _clean_bio_text(m.group(1).strip())
            role = _extract_role(snippet)
            if role and company:
                result = f"{role}, {company}"
                logger.info("Position+company for '%s': %s", name, result)
                return result

    # Fallback: just the role keyword
    return ""


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


# Brazilian name suffixes that should NOT be standalone names
_NAME_SUFFIXES = {"junior", "jr", "filho", "neto", "segundo", "terceiro",
                  "sobrinho", "bisneto"}

# Brazilian name connectors (lowercase words within names)
# NOTE: "e" is NOT a connector — it means "and" and separates different people
_NAME_CONNECTORS = {"de", "da", "do", "das", "dos"}


def _merge_split_names(candidates: list[str], source_text: str) -> list[str]:
    """Merge name fragments that are parts of a composite Brazilian name.

    E.g. ["João Manoel", "Lima Junior"] → ["João Manoel Lima Junior"]
    if "João Manoel Lima Junior" appears in any source text.

    Strategy (in order):
    1. Direct concatenation: check if "A B" exists in source text
    2. Connector merge: check if "A de B" exists in source text
    3. Suffix extension: if name ends with Junior/Filho/Neto, scan backwards
    4. Proximity merge: if two candidates appear within 3 words of each other
       in the same sentence, merge them
    """
    if len(candidates) < 2:
        return candidates

    source_lower = source_text.lower()
    merged: list[str] = []
    absorbed: set[int] = set()

    for i, name_a in enumerate(candidates):
        if i in absorbed:
            continue

        best = name_a

        for j, name_b in enumerate(candidates):
            if j <= i or j in absorbed:
                continue

            # Strategy 1: Direct concatenation
            found = False
            for combo in [f"{name_a} {name_b}", f"{name_b} {name_a}"]:
                combo_lower = combo.lower()
                if combo_lower in source_lower:
                    idx = source_lower.index(combo_lower)
                    best = source_text[idx:idx + len(combo)]
                    absorbed.add(j)
                    logger.info("Merged names: '%s' + '%s' → '%s'", name_a, name_b, best)
                    found = True
                    break
            if found:
                continue

            # Strategy 2: Connector merge ("de", "da", "do")
            for conn in _NAME_CONNECTORS:
                for combo in [f"{name_a} {conn} {name_b}", f"{name_b} {conn} {name_a}"]:
                    combo_lower = combo.lower()
                    if combo_lower in source_lower:
                        idx = source_lower.index(combo_lower)
                        best = source_text[idx:idx + len(combo)]
                        absorbed.add(j)
                        logger.info("Merged names with connector: '%s' + '%s' → '%s'",
                                    name_a, name_b, best)
                        break
                if j in absorbed:
                    break
            if j in absorbed:
                continue

            # Strategy 4: Proximity merge — if A and B appear close in same line
            proximity_result = _try_proximity_merge(name_a, name_b, source_text)
            if proximity_result:
                best = proximity_result
                absorbed.add(j)
                logger.info("Proximity merged: '%s' + '%s' → '%s'", name_a, name_b, best)

        # Strategy 3: Suffix extension for standalone suffix names
        last_word = best.split()[-1].lower()
        if last_word in _NAME_SUFFIXES and len(best.split()) <= 2:
            longer = _find_longer_name_in_text(best, source_text)
            if longer and longer.lower() != best.lower():
                best = longer
                logger.info("Extended suffix name: '%s' → '%s'", name_a, best)

        merged.append(best)

    return _deduplicate(merged)


def _try_proximity_merge(name_a: str, name_b: str, source_text: str) -> str | None:
    """Try to merge two names if they appear close together in source text.

    Looks for patterns where name_a and name_b are within 0-2 words
    of each other on the same line (allowing connectors like "de").
    """
    a_lower = name_a.lower()
    b_lower = name_b.lower()

    for line in source_text.splitlines():
        line_lower = line.lower()
        # Check both orders
        for first, second, first_orig, second_orig in [
            (a_lower, b_lower, name_a, name_b),
            (b_lower, a_lower, name_b, name_a),
        ]:
            idx_first = line_lower.find(first)
            if idx_first < 0:
                continue
            idx_second = line_lower.find(second, idx_first + len(first))
            if idx_second < 0:
                continue

            # What's between them?
            between = line[idx_first + len(first):idx_second].strip()
            between_words = between.split()

            # Allow 0 words (adjacent) or 1-2 connector words (de, da, do)
            if len(between_words) == 0:
                # Adjacent: "João Manoel Lima Junior" — already handled by strategy 1
                continue
            if (len(between_words) <= 2 and
                    all(w.lower() in _NAME_CONNECTORS for w in between_words)):
                # Connector: "Maria Clara de Souza Lima"
                full = line[idx_first:idx_second + len(second)]
                words = full.split()
                if 3 <= len(words) <= 7:
                    return full

    return None


def _find_longer_name_in_text(short_name: str, text: str) -> str | None:
    """Find a longer version of a name in source text.

    E.g. given "Lima Junior", finds "João Manoel Lima Junior" in the text.
    Scans backwards from the short name, skipping over non-name words
    to find preceding capitalized words.
    """
    short_lower = short_name.lower()

    # Search line by line for better context
    for line in text.splitlines():
        line_lower = line.lower()
        idx = line_lower.find(short_lower)
        while idx >= 0:
            before = line[:idx].rstrip()
            if before:
                # Grab preceding capitalized words + connectors
                extra_words: list[str] = []
                skipped = 0
                for word in reversed(before.split()):
                    if not word:
                        continue
                    if (word[0].isupper() and
                            re.match(r"^[A-ZÀ-Ü][a-zà-ü]+$", word)):
                        extra_words.insert(0, word)
                        skipped = 0  # reset skip counter
                    elif word.lower() in _NAME_CONNECTORS:
                        extra_words.insert(0, word)
                    else:
                        # Allow skipping 1 non-name word (punctuation, etc.)
                        # but stop after 2 consecutive non-name words
                        skipped += 1
                        if skipped >= 1:
                            break
                    if len(extra_words) >= 4:
                        break

                if extra_words:
                    prefix = " ".join(extra_words)
                    full_name = f"{prefix} {line[idx:idx + len(short_name)]}"
                    words = full_name.split()
                    cap_words = [w for w in words
                                 if w[0].isupper() or w.lower() in _NAME_CONNECTORS]
                    if 3 <= len(words) <= 6 and len(cap_words) >= len(words) - 1:
                        return full_name

            idx = line_lower.find(short_lower, idx + 1)

    return None


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


def _web_search_all(name: str, context: str = "") -> str:
    """Try all search sources in order, with caching.

    Adds channel/topic context to queries to disambiguate common names.
    Validates results to reject wrong-person matches.
    Returns combined snippet text or empty string.
    """
    cache_key = name.lower().strip()
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    # Build a context-enriched query to avoid homonym confusion
    # e.g. "João Manoel advogado arbitragem Canal Arbitragem"
    context_terms = _build_search_context(context)
    query_with_context = f"{name} {context_terms}".strip()
    query_basic = f"{name} {context}".strip()

    snippet = ""

    # 1. Google Custom Search — prioritize LinkedIn results
    snippet = _search_google_api(f"{name} linkedin {context_terms}")
    if not snippet:
        snippet = _search_google_api(query_with_context)

    # 2. Wikipedia PT (free, reliable)
    if not snippet:
        snippet = _search_wikipedia(f"{name} {context_terms}")

    # 3. DuckDuckGo Instant Answer API (free, structured)
    if not snippet:
        snippet = _search_duckduckgo_api(query_with_context)

    # 4. DuckDuckGo HTML scraping (last resort)
    if not snippet:
        snippet = _search_duckduckgo(query_with_context)

    # Validate: reject results that clearly describe a different person
    if snippet and not _snippet_matches_context(snippet, name, context):
        logger.warning(
            "Busca para '%s' rejeitada — resultado não combina com contexto '%s': %s",
            name, context, snippet[:120],
        )
        snippet = ""

    _search_cache[cache_key] = snippet
    return snippet


# Words that indicate the search found the WRONG person
_WRONG_PERSON_INDICATORS = {
    "sacerdote", "padre", "bispo", "cardeal", "papa",
    "cantor", "cantora", "músico", "música", "banda", "álbum", "disco",
    "ator", "atriz", "filme", "novela", "telenovela", "série",
    "jogador", "jogadora", "futebol", "seleção", "gol", "campeonato",
    "classificando", "quartas de final", "semifinal", "final",
    "enfrentar", "boca juniors", "libertadores", "copa do mundo",
    "imperador", "rei", "rainha", "príncipe", "princesa",
    "santo", "santa", "beato", "beata", "mártir",
    "compositor", "cineasta",
    "prefeito", "prefeita", "governador", "governadora",
    "vereador", "vereadora", "deputado", "deputada",
}

# Words that confirm the person matches legal/arbitration context
_RIGHT_PERSON_INDICATORS = {
    "advogado", "advogada", "árbitro", "árbitra", "arbitragem",
    "direito", "jurídico", "jurídica", "juiz", "juíza",
    "mediador", "mediadora", "mediação",
    "procurador", "procuradora", "promotor", "promotora",
    "tribunal", "vara", "câmara", "oab", "escritório",
    "contrato", "litígio", "disputa", "cláusula",
    "professor", "professora", "mestre", "doutor", "doutora",
    "consultor", "consultora", "gestor", "gestora",
    "engenheiro", "engenheira", "empresário", "empresária",
    "especialista", "perito", "perita",
}


def _snippet_matches_context(snippet: str, name: str, context: str) -> bool:
    """Check if a search snippet actually describes the right person.

    Rejects results that clearly describe a different person (e.g.,
    a historical military figure instead of a lawyer) and YouTube UI junk.
    """
    text_lower = snippet.lower()
    context_lower = context.lower()

    # Reject YouTube UI garbage
    if _is_youtube_ui_junk(snippet):
        return False

    # Check for wrong-person indicators
    wrong_count = sum(1 for w in _WRONG_PERSON_INDICATORS if w in text_lower)
    right_count = sum(1 for w in _RIGHT_PERSON_INDICATORS if w in text_lower)

    # Also check if channel name context words appear
    context_words = set(context_lower.split())
    context_match = any(w in text_lower for w in context_words if len(w) > 3)

    # If snippet has wrong indicators and NO right indicators → reject
    if wrong_count > 0 and right_count == 0 and not context_match:
        return False

    # If snippet has more wrong than right indicators → reject
    if wrong_count > right_count and not context_match:
        return False

    return True


def _build_search_context(channel_name: str) -> str:
    """Build search context terms from channel name.

    For channels about arbitration/law, adds relevant legal terms
    to help disambiguation.
    """
    channel_lower = channel_name.lower()

    # Detect channel topic and add relevant search terms
    legal_keywords = ["arbitragem", "direito", "jurídic", "advogad", "mediação",
                      "tribunal", "legal", "law", "arbitrat"]
    is_legal = any(kw in channel_lower for kw in legal_keywords)

    if is_legal:
        return f"advogado arbitragem direito {channel_name}"

    # For other channels, just use the channel name
    return channel_name


def _google_canonise(name: str, context: str) -> str | None:
    """Search web for the canonical spelling of a name."""
    snippet = _web_search_all(name, context)
    if not snippet:
        return None

    pattern = re.compile(re.escape(name), re.IGNORECASE)
    match = pattern.search(snippet)
    if match:
        return match.group(0)

    return name  # Confirmed existence, keep original spelling


def _search_wikipedia(name: str) -> str:
    """Search Portuguese Wikipedia API. Free, no key, works from any server."""
    # Try PT Wikipedia first, then EN
    for lang in ("pt", "en"):
        try:
            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": name,
                "format": "json",
                "srlimit": "3",
                "srprop": "snippet",
                "utf8": "1",
            })
            url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "DescricaoArbitragem/1.0 (bot; educational)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = data.get("query", {}).get("search", [])
            if not results:
                continue

            # Combine snippets, clean HTML
            texts = []
            for r in results:
                snippet = r.get("snippet", "")
                clean = re.sub(r"<[^>]+>", " ", snippet)
                clean = re.sub(r"\s+", " ", clean).strip()
                if clean:
                    texts.append(clean)

            if texts:
                result = " ".join(texts)
                logger.info("Wikipedia (%s) OK para '%s': %d chars de %d results",
                            lang, name, len(result), len(results))
                return result[:2000]

        except Exception as exc:
            logger.debug("Wikipedia (%s) falhou para '%s': %s", lang, name, exc)

    return ""


def _search_duckduckgo_api(name: str) -> str:
    """Search DuckDuckGo Instant Answer API. Free, no key, structured data."""
    try:
        params = urllib.parse.urlencode({
            "q": name,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DescricaoArbitragem/1.0 (bot; educational)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        texts = []
        # Abstract (main result)
        abstract = data.get("Abstract", "")
        if abstract:
            texts.append(abstract)
        # AbstractText (plain text)
        abstract_text = data.get("AbstractText", "")
        if abstract_text and abstract_text != abstract:
            texts.append(abstract_text)
        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            text = topic.get("Text", "")
            if text:
                texts.append(text)

        if texts:
            result = " ".join(texts)
            logger.info("DDG API OK para '%s': %d chars", name, len(result))
            return result[:2000]

        logger.debug("DDG API: sem resultados para '%s'", name)
        return ""

    except Exception as exc:
        logger.debug("DDG API falhou para '%s': %s", name, exc)
        return ""


def _search_google_api(query: str) -> str:
    """Search Google via Custom Search JSON API.

    Requires two env vars:
      - GOOGLE_API_KEY: API key with Custom Search API enabled
      - GOOGLE_CSE_ID:  Programmable Search Engine ID (cx)

    Free tier: 100 queries/day. Works from any server, no captcha.
    Automatically disables after first 403 to avoid wasting time.
    """
    global _google_api_disabled

    if _google_api_disabled:
        return ""

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        # Fall back to YOUTUBE_API_KEY — same Google Cloud project
        api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    cse_id = os.environ.get("GOOGLE_CSE_ID", "").strip()

    if not api_key or not cse_id:
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
        total = data.get("searchInformation", {}).get("totalResults", "?")
        if not items:
            msg = (f"Google Search: 0 resultados (totalResults={total}) "
                   f"— verifique se o CSE tem 'Search the entire web' habilitado")
            logger.warning("Google API: 0 resultados para '%s' (total=%s)", query, total)
            if msg not in _search_errors:
                _search_errors.append(msg)
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
        err_msg = f"Google Search HTTP {exc.code}"
        if exc.code == 403:
            _google_api_disabled = True
            err_msg = (f"Google Search 403 — Custom Search API não habilitada "
                       f"neste projeto. Habilite em console.cloud.google.com → "
                       f"APIs → Custom Search JSON API. "
                       f"Detalhe: {error_body[:150]}")
        elif exc.code == 400:
            err_msg = f"Google Search 400 — verifique GOOGLE_CSE_ID: {error_body[:100]}"
        else:
            err_msg = f"Google Search HTTP {exc.code}: {error_body[:100]}"
        logger.warning("Google API falhou para '%s': %s", query, err_msg)
        _search_errors.append(err_msg)
        return ""
    except Exception as exc:
        err_msg = f"Google Search erro: {exc}"
        logger.warning("Google API falhou para '%s': %s", query, err_msg)
        _search_errors.append(err_msg)
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

        # Try multiple patterns — DDG changes HTML structure
        snippets = []
        for pattern in [
            r'class="result__snippet"[^>]*>(.*?)</a>',
            r'class="result__snippet"[^>]*>(.*?)</span>',
            r'class="result__snippet"[^>]*>(.*?)</div>',
            r'class="result__body"[^>]*>(.*?)</div>',
            # Lite version pattern
            r'class="result-snippet"[^>]*>(.*?)</[a-z]',
        ]:
            snippets = re.findall(pattern, html, re.DOTALL)
            if snippets:
                break

        if not snippets:
            # Last resort: extract any text between result markers
            snippets = re.findall(r'snippet["\s][^>]*>(.*?)</', html, re.DOTALL)

        if not snippets:
            logger.warning("DuckDuckGo: nenhum snippet para '%s' (html=%d bytes)",
                           query, len(html))
            return ""

        # Clean HTML tags and join snippets
        texts = []
        for s in snippets[:5]:
            clean = re.sub(r"<[^>]+>", " ", s)
            clean = re.sub(r"&[a-zA-Z]+;", " ", clean)  # HTML entities
            clean = re.sub(r"\s+", " ", clean).strip()
            if clean and len(clean) > 10:
                texts.append(clean)

        result = " ".join(texts)
        logger.info("DuckDuckGo OK para '%s': %d chars", query, len(result))
        return result[:2000]

    except Exception as exc:
        logger.warning("DuckDuckGo falhou para '%s': %s", query, exc)
        return ""




def _texts_overlap(a: str, b: str) -> bool:
    """Check if two texts significantly overlap (>50% of words in common)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    common = words_a & words_b
    smaller = min(len(words_a), len(words_b))
    return len(common) / smaller > 0.5


def _summarise_snippet_excluding(snippet: str, exclude: str, max_words: int = 12) -> str:
    """Extract bio from snippet, skipping text that overlaps with 'exclude'.

    Used to avoid duplicating the position/company in the bio.
    Prefers descriptive sentences with verbs/adjectives over name/title fragments.
    """
    # Split into sentences
    sentences = re.split(r"[.!?]", snippet)

    # Score and pick the best descriptive sentence
    best = ""
    best_score = -1
    for s in sentences:
        s = _clean_bio_text(s.strip())
        words = s.split()
        if len(words) < 3 or len(words) > 20:
            continue
        # Skip if it overlaps with exclude text
        if _texts_overlap(s, exclude):
            continue
        # Score: prefer sentences with descriptive words
        score = 0
        s_lower = s.lower()
        for indicator in ["experiência", "atuação", "especialista", "atua",
                          "profissional", "formado", "formada", "graduado",
                          "membro", "sólida", "ampla", "larga"]:
            if indicator in s_lower:
                score += 2
        # Prefer sentences that contain role keywords
        if _ROLE_PATTERN.search(s):
            score += 1
        # Penalize sentences that look like names/titles
        if re.match(r"^[A-ZÀ-Ü][a-zà-ü]+ [A-ZÀ-Ü]", s):
            score -= 2
        if score > best_score:
            best_score = score
            best = " ".join(words[:max_words])

    return best


_YOUTUBE_UI_JUNK = [
    "seja notificado",
    "notificado a cada",
    "a cada atualização",
    "inscreva-se",
    "inscreva se",
    "se inscreva",
    "ative o sininho",
    "ative as notificações",
    "clique no sininho",
    "curtir e compartilhar",
    "deixe seu like",
    "deixe o like",
    "link na descrição",
    "link na bio",
    "sigam no instagram",
    "siga no instagram",
    "siga nas redes",
    "redes sociais",
    "nos acompanhe",
    "compartilhe o vídeo",
    "compartilhe este vídeo",
    "subscribe",
    "notification bell",
    "turn on notifications",
    "like and subscribe",
    "mostrar menos",
    "mostrar mais",
    "show less",
    "show more",
    "leia mais",
    "saiba mais",
]


def _is_youtube_ui_junk(text: str) -> bool:
    """Check if text is YouTube UI/CTA garbage rather than real content."""
    text_lower = text.lower().strip()
    return any(junk in text_lower for junk in _YOUTUBE_UI_JUNK)


def _clean_bio_text(text: str) -> str:
    """Clean garbage from bio text: emojis, @handles, URLs, years, etc."""
    # Remove emojis (Unicode emoji ranges)
    text = re.sub(
        r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FE0F"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
        r"\U0000200D\U0000FE0F]+", " ", text
    )
    # Remove @handles
    text = re.sub(r"@\w+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove standalone years like (1991) or "1834–1923"
    text = re.sub(r"\(\d{4}\)", "", text)
    text = re.sub(r"\b\d{4}\s*[–—-]\s*\d{4}\b", "", text)
    # Remove ALL_CAPS noise words (but keep company names/certifications)
    # Only remove if they look like random noise, not acronyms
    # Keep: FGBS, IBGC, CCA, MCIArb, OAB, etc.
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing punctuation junk
    text = re.sub(r"^[\s,\-–—:;|]+", "", text)
    text = re.sub(r"[\s,\-–—:;|]+$", "", text)
    return text


def _summarise_snippet(snippet: str, max_words: int = 12) -> str:
    """Extract a short bio-like sentence from a search snippet."""
    # Build role alternatives from the keywords list
    role_alts = "|".join(re.escape(r) for r in _ROLE_KEYWORDS)

    bio_patterns = [
        # "Sou um(a) profissional com..." (LinkedIn about)
        r"[Ss]ou\s+(?:um(?:a)?\s+)?(?:profissional|" + role_alts + r")[a-zà-ü]*\s+(?:com|de|que)[^.]{5,80}",
        # "profissional com sólida/ampla/larga experiência/atuação"
        r"profissional\s+com\s+[^.]{5,80}",
        # "é advogado/árbitro/etc"
        r"é\s+(?:um(?:a)?\s+)?(?:" + role_alts + r")[a-zà-ü]*(?:\s+[^.]{3,60})?",
        # "Role com/e experiência em..."
        r"(?:" + role_alts + r")[a-zà-ü]*\s+(?:com|e)\s+(?:experiência|atuação|especialização)[^.]{3,60}",
        # "é um(a) profissional"
        r"é\s+(?:um(?:a)?)\s+([^.]{10,80})",
        # "atua como advogado" / "atua na área de"
        r"atua(?:ndo)?\s+(?:como|na|no|em)[^.]{5,60}",
        # "especialista em / especializado em"
        r"especialista\s+em\s+[^.]{5,60}",
        r"especializado(?:a)?\s+em\s+[^.]{5,60}",
        # "experiência em/na/no"
        r"experiência\s+(?:em|na|no|de)[^.]{5,60}",
        # Role keyword followed by context
        r"(?:" + role_alts + r")[a-zà-ü]*\s+(?:e\s+)?(?:especialista|especializado)?[^.]{3,60}",
    ]
    for pat in bio_patterns:
        m = re.search(pat, snippet, re.IGNORECASE)
        if m:
            text = _clean_bio_text(m.group(0).strip())
            if _is_youtube_ui_junk(text):
                continue
            words = text.split()[:max_words]
            bio = " ".join(words)
            # Don't end on articles/prepositions
            bio = re.sub(r"\s+(?:e|em|de|do|da|na|no|com|para|a|o|as|os|que|um|uma)\s*$", "", bio)
            if len(bio.split()) >= 2 and not _is_youtube_ui_junk(bio):
                return bio

    # Fallback: take the first clean sentence-like chunk
    sentences = re.split(r"[.!?]", snippet)
    for s in sentences:
        s = _clean_bio_text(s.strip())
        if _is_youtube_ui_junk(s):
            continue
        if 3 <= len(s.split()) <= 15:
            words = s.split()[:max_words]
            result = " ".join(words)
            if not _is_youtube_ui_junk(result):
                return result

    return ""
