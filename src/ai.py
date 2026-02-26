"""Gemini AI integration for research and content generation.

Optimised for the free tier: a SINGLE API call generates all content
(summary, topics, chapters, participant bios) to stay well within
the 15 req/min and 1 500 req/day limits.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Track errors so the pipeline can include them in diagnostics
_recent_errors: list[str] = []

# Models to try in order of preference
_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]

# Cache the working model name so we don't retry 404s every call
_working_model: str | None = None


def gemini_available() -> bool:
    """Check if Gemini API key is configured."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        logger.warning("GEMINI_API_KEY não configurada — IA desabilitada")
    else:
        logger.info("GEMINI_API_KEY configurada (%s...)", key[:8])
    return bool(key)


def get_recent_errors() -> list[str]:
    """Return and clear recent Gemini errors for diagnostics."""
    errors = list(_recent_errors)
    _recent_errors.clear()
    return errors


def generate_all_content(
    title: str,
    description: str,
    transcript: str,
    participant_names: list[str],
    existing_chapters: list[dict],
    duration: int,
    channel_name: str,
) -> dict:
    """Generate ALL AI content in a SINGLE Gemini call.

    Returns a dict with keys:
      - summary: str (1-2 sentences)
      - topics: list[str] (4-6 topics)
      - chapters: list[dict] (with start/title)
      - participants: list[dict] (with name/role/bio)

    Any missing key means that part failed → use heuristic fallback.
    """
    transcript_sample = transcript[:4000] if transcript else ""
    names = ", ".join(participant_names) if participant_names else "os participantes"
    duration_min = duration // 60

    # Build chapters context
    chapters_ctx = ""
    if existing_chapters:
        chapters_ctx = (
            "Capítulos atuais (melhore os títulos, mantenha timestamps):\n"
            + "\n".join(f"{ch['start']}s - {ch['title']}" for ch in existing_chapters)
        )
    elif transcript_sample:
        chapters_ctx = (
            f"Não há capítulos. Crie entre 5 e 10 com timestamps espaçados "
            f"para um vídeo de {duration_min} minutos. "
            f"Primeiro: 0s (Introdução), último: Conclusão."
        )

    # Build participants context
    participants_ctx = ""
    if participant_names:
        participants_ctx = (
            "Para cada participante, descubra o cargo profissional (1-2 palavras) "
            "e uma bio curta (até 15 palavras). "
            "Se não souber, use role='Profissional' e bio='Participante do programa'.\n"
            f"Participantes: {names}"
        )

    prompt = f"""Você é um assistente que gera metadados para vídeos de podcast no YouTube.
Podcast: '{title}'
Canal: '{channel_name}'
Duração: {duration_min} minutos
Descrição original: {description[:500]}
Trecho da transcrição: {transcript_sample}

Gere TUDO abaixo em um ÚNICO JSON:

1. "summary": resumo de 1-2 frases (máx 40 palavras) do conteúdo do episódio.
   Será usado depois de "No episódio de hoje, {names} exploram ..."
   NÃO inclua nomes dos participantes.

2. "topics": array de 4-6 tópicos principais (cada um com 3-8 palavras).

3. "chapters": array de objetos {{"start": segundos, "title": "título descritivo"}}.
   {chapters_ctx}

4. "participants": array de objetos {{"name": "nome", "role": "cargo", "bio": "bio curta"}}.
   {participants_ctx}

Responda APENAS o JSON, sem markdown, sem explicação, sem ```."""

    result = _call_gemini(prompt)
    if not result:
        return {}

    try:
        clean = result.strip()
        # Remove markdown code block if present
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
        if not isinstance(data, dict):
            return {}

        parsed: dict = {}

        # Summary
        if data.get("summary"):
            s = str(data["summary"]).strip().strip('"').strip("'")
            words = s.split()
            if len(words) > 50:
                s = " ".join(words[:50]) + "."
            parsed["summary"] = s

        # Topics
        if isinstance(data.get("topics"), list) and data["topics"]:
            parsed["topics"] = [str(t) for t in data["topics"][:6]]

        # Chapters
        if isinstance(data.get("chapters"), list) and data["chapters"]:
            parsed["chapters"] = [
                {"start": int(ch.get("start", 0)), "title": str(ch.get("title", ""))}
                for ch in data["chapters"]
                if isinstance(ch, dict)
            ]

        # Participants
        if isinstance(data.get("participants"), list) and data["participants"]:
            parsed["participants"] = [
                {
                    "name": str(p.get("name", "")),
                    "role": str(p.get("role", "")),
                    "bio": str(p.get("bio", "Participante do programa")),
                }
                for p in data["participants"]
                if isinstance(p, dict)
            ]

        logger.info("generate_all_content OK: keys=%s", list(parsed.keys()))
        return parsed

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        err = f"Falha ao parsear resposta do Gemini: {exc}"
        logger.error(err)
        _recent_errors.append(err)
        return {}


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BACKOFF = [5, 15, 30]  # seconds to wait on 429


def _call_gemini(prompt: str) -> str:
    """Call Gemini API with retry on rate limits (429).

    Tries multiple models if the primary one returns 404.
    """
    global _working_model

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("_call_gemini: sem API key, pulando")
        return ""

    models_to_try = [_working_model] if _working_model else _MODELS

    for model in models_to_try:
        result = _call_with_retry(prompt, model, api_key)
        if result is not None:
            _working_model = model
            return result
        # result is None means 404 (model not found), try next

    return ""


def _call_with_retry(prompt: str, model: str, api_key: str) -> str | None:
    """Call Gemini with automatic retry on 429 rate limit.

    Returns text, "" on error, or None if 404 (try next model).
    """
    for attempt in range(_MAX_RETRIES + 1):
        result = _call_gemini_single(prompt, model, api_key)

        # None = 404 model not found → bubble up to try next model
        if result is None:
            return None

        # _RATE_LIMITED sentinel: wait and retry
        if result == _RATE_LIMITED:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF[attempt]
                logger.info("Rate limited (429). Aguardando %ds antes de tentar novamente (%d/%d)...",
                            wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait)
                continue
            else:
                logger.error("Rate limit persistente após %d tentativas", _MAX_RETRIES + 1)
                _recent_errors.append(
                    f"Gemini rate limit (429) — tentou {_MAX_RETRIES + 1}x. "
                    f"Aguarde 1 minuto e tente novamente."
                )
                return ""

        # Normal result (could be "" on other errors, or actual text)
        return result

    return ""


# Sentinel value for rate limit
_RATE_LIMITED = "__RATE_LIMITED__"


def _call_gemini_single(prompt: str, model: str, api_key: str) -> str | None:
    """Call a specific Gemini model once.

    Returns:
      - text response on success
      - "" on non-retryable error
      - None if 404 (model not found)
      - _RATE_LIMITED if 429
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )

    body = json.dumps({
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Chamando Gemini (%s) — prompt: %.80s...", model, prompt)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        candidates = data.get("candidates", [])
        if not candidates:
            err = f"Gemini ({model}) retornou sem candidates: {json.dumps(data, ensure_ascii=False)[:300]}"
            logger.error(err)
            _recent_errors.append(err)
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            err = f"Gemini ({model}) candidates sem parts"
            logger.error(err)
            _recent_errors.append(err)
            return ""
        text = parts[0].get("text", "")
        logger.info("Gemini (%s) respondeu OK (%d chars)", model, len(text))
        return text
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass

        err_detail = f"HTTP {exc.code}: {exc.reason}"
        try:
            err_data = json.loads(error_body)
            err_detail = err_data.get("error", {}).get("message", err_detail)
        except Exception:
            if error_body:
                err_detail = f"HTTP {exc.code}: {error_body[:200]}"

        if exc.code == 404:
            logger.info("Modelo %s não encontrado (404), tentando próximo", model)
            return None

        if exc.code == 429:
            logger.warning("Gemini rate limit (429): %s", err_detail)
            return _RATE_LIMITED

        err = f"Gemini ({model}) {err_detail}"
        logger.error(err)
        _recent_errors.append(err)
    except urllib.error.URLError as exc:
        err = f"Gemini ({model}) URLError: {exc.reason}"
        logger.error(err)
        _recent_errors.append(err)
    except Exception as exc:
        err = f"Gemini ({model}) erro: {exc}"
        logger.error(err)
        _recent_errors.append(err)

    return ""
