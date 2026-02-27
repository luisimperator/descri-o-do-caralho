"""AI integration for research and content generation.

Supports FREE providers:
  1. OpenRouter (OPENROUTER_API_KEY) — free models, works from datacenters
  2. Gemini    (GEMINI_API_KEY)      — 15 req/min, 1 500 req/day

A SINGLE API call generates all content (summary, topics, chapters,
participant bios). If one provider fails, falls back to the other.
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


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ai_available() -> bool:
    """Check if any AI provider is configured."""
    openrouter = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    gemini = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    if openrouter:
        logger.info("OPENROUTER_API_KEY configurada")
    if gemini:
        logger.info("GEMINI_API_KEY configurada")
    if not openrouter and not gemini:
        logger.warning("Nenhuma API key de IA configurada — IA desabilitada")
    return openrouter or gemini


# Keep old name as alias so existing imports still work
gemini_available = ai_available


def get_recent_errors() -> list[str]:
    """Return and clear recent AI errors for diagnostics."""
    errors = list(_recent_errors)
    _recent_errors.clear()
    return errors


def get_ai_status() -> dict:
    """Return which AI providers are configured."""
    return {
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "gemini": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    }


# ---------------------------------------------------------------------------
# Main entry point — single call for ALL content
# ---------------------------------------------------------------------------

def generate_all_content(
    title: str,
    description: str,
    transcript: str,
    participant_names: list[str],
    existing_chapters: list[dict],
    duration: int,
    channel_name: str,
) -> dict:
    """Generate ALL AI content in a SINGLE call.

    Tries OpenRouter first (free models, works from servers),
    then Gemini as fallback.

    Returns a dict with keys:
      - summary: str
      - topics: list[str]
      - chapters: list[dict]
      - participants: list[dict]

    Any missing key means that part failed -> use heuristic fallback.
    """
    prompt = _build_prompt(
        title, description, transcript, participant_names,
        existing_chapters, duration, channel_name,
    )

    result_text = ""
    providers = _get_provider_order()

    for provider_fn in providers:
        result_text = provider_fn(prompt)
        if result_text:
            break

    if not result_text:
        return {}

    return _parse_ai_response(result_text)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    title: str,
    description: str,
    transcript: str,
    participant_names: list[str],
    existing_chapters: list[dict],
    duration: int,
    channel_name: str,
) -> str:
    transcript_sample = transcript[:4000] if transcript else ""
    names = ", ".join(participant_names) if participant_names else "os participantes"
    duration_min = duration // 60

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

    participants_ctx = ""
    if participant_names:
        participants_ctx = (
            "Para cada participante, descubra o cargo profissional (1-2 palavras) "
            "e uma bio curta (até 15 palavras). "
            "Se não souber, use role='Profissional' e bio='Participante do programa'.\n"
            f"Participantes: {names}"
        )

    return f"""Você é um assistente que gera metadados para vídeos de podcast no YouTube.
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


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_ai_response(result_text: str) -> dict:
    try:
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
        if not isinstance(data, dict):
            return {}

        parsed: dict = {}

        if data.get("summary"):
            s = str(data["summary"]).strip().strip('"').strip("'")
            words = s.split()
            if len(words) > 50:
                s = " ".join(words[:50]) + "."
            parsed["summary"] = s

        if isinstance(data.get("topics"), list) and data["topics"]:
            parsed["topics"] = [str(t) for t in data["topics"][:6]]

        if isinstance(data.get("chapters"), list) and data["chapters"]:
            parsed["chapters"] = [
                {"start": int(ch.get("start", 0)), "title": str(ch.get("title", ""))}
                for ch in data["chapters"]
                if isinstance(ch, dict)
            ]

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

        logger.info("AI response parsed OK: keys=%s", list(parsed.keys()))
        return parsed

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        err = f"Falha ao parsear resposta da IA: {exc}"
        logger.error(err)
        _recent_errors.append(err)
        return {}


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _get_provider_order() -> list:
    """Return provider call functions in priority order.

    OpenRouter first (free models, works from servers), Gemini second.
    """
    providers = []
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        providers.append(_call_openrouter)
    if os.environ.get("GEMINI_API_KEY", "").strip():
        providers.append(_call_gemini)
    return providers


# ---------------------------------------------------------------------------
# Retry / shared constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BACKOFF = [5, 15, 30]
_RATE_LIMITED = "__RATE_LIMITED__"


# ---------------------------------------------------------------------------
# OpenRouter provider (free models, OpenAI-compatible, works from datacenters)
# ---------------------------------------------------------------------------

# Free models on OpenRouter (suffix :free = no cost)
_OPENROUTER_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]
_openrouter_working_model: str | None = None


def _call_openrouter(prompt: str) -> str:
    """Call OpenRouter API with retry on rate limit."""
    global _openrouter_working_model

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return ""

    models = [_openrouter_working_model] if _openrouter_working_model else _OPENROUTER_MODELS

    for model in models:
        result = _call_openrouter_with_retry(prompt, model, api_key)
        if result is not None:
            _openrouter_working_model = model
            return result

    return ""


def _call_openrouter_with_retry(prompt: str, model: str, api_key: str) -> str | None:
    for attempt in range(_MAX_RETRIES + 1):
        result = _call_openrouter_single(prompt, model, api_key)

        if result is None:
            return None  # model not available, try next

        if result == _RATE_LIMITED:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF[attempt]
                logger.info("OpenRouter rate limited. Aguardando %ds (%d/%d)...",
                            wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait)
                continue
            else:
                _recent_errors.append(
                    f"OpenRouter rate limit — tentou {_MAX_RETRIES + 1}x."
                )
                return ""

        return result

    return ""


def _call_openrouter_single(prompt: str, model: str, api_key: str) -> str | None:
    """Call OpenRouter once. Returns text, '' on error, None if model unavailable, _RATE_LIMITED on 429."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://descricao-arbitragem.up.railway.app",
            "X-Title": "Descricao Arbitragem",
        },
        method="POST",
    )

    logger.info("Chamando OpenRouter (%s)...", model)

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Check for error in response body (OpenRouter sometimes returns 200 with error)
        if data.get("error"):
            err_msg = data["error"].get("message", str(data["error"]))
            err_code = data["error"].get("code", 0)
            if err_code == 404 or "not found" in str(err_msg).lower():
                logger.info("OpenRouter modelo %s indisponível: %s", model, err_msg)
                return None
            if err_code == 429 or "rate" in str(err_msg).lower():
                logger.warning("OpenRouter rate limit: %s", err_msg)
                return _RATE_LIMITED
            err = f"OpenRouter ({model}) erro: {err_msg}"
            logger.error(err)
            _recent_errors.append(err)
            return ""

        choices = data.get("choices", [])
        if not choices:
            err = f"OpenRouter ({model}) retornou sem choices"
            logger.error(err)
            _recent_errors.append(err)
            return ""
        text = choices[0].get("message", {}).get("content", "")
        logger.info("OpenRouter (%s) respondeu OK (%d chars)", model, len(text))
        return text
    except urllib.error.HTTPError as exc:
        error_body = _read_error_body(exc)
        err_detail = _parse_error_detail(exc, error_body)

        if exc.code == 404:
            logger.info("OpenRouter modelo %s não encontrado", model)
            return None
        if exc.code == 429:
            logger.warning("OpenRouter rate limit (429): %s", err_detail)
            return _RATE_LIMITED

        err = f"OpenRouter ({model}) {err_detail}"
        logger.error(err)
        _recent_errors.append(err)
    except Exception as exc:
        err = f"OpenRouter ({model}) erro: {exc}"
        logger.error(err)
        _recent_errors.append(err)

    return ""


# ---------------------------------------------------------------------------
# Gemini provider (direct Google API)
# ---------------------------------------------------------------------------

_GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]
_gemini_working_model: str | None = None


def _call_gemini(prompt: str) -> str:
    """Call Gemini API with retry on rate limit."""
    global _gemini_working_model

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ""

    models = [_gemini_working_model] if _gemini_working_model else _GEMINI_MODELS

    for model in models:
        result = _call_gemini_with_retry(prompt, model, api_key)
        if result is not None:
            _gemini_working_model = model
            return result

    return ""


def _call_gemini_with_retry(prompt: str, model: str, api_key: str) -> str | None:
    for attempt in range(_MAX_RETRIES + 1):
        result = _call_gemini_single(prompt, model, api_key)

        if result is None:
            return None

        if result == _RATE_LIMITED:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF[attempt]
                logger.info("Gemini rate limited. Aguardando %ds (%d/%d)...",
                            wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait)
                continue
            else:
                _recent_errors.append(
                    f"Gemini rate limit (429) — tentou {_MAX_RETRIES + 1}x."
                )
                return ""

        return result

    return ""


def _call_gemini_single(prompt: str, model: str, api_key: str) -> str | None:
    """Call Gemini once. Returns text, '' on error, None on 404, _RATE_LIMITED on 429."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )

    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Chamando Gemini (%s)...", model)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            err = f"Gemini ({model}) retornou sem candidates"
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
        error_body = _read_error_body(exc)
        err_detail = _parse_error_detail(exc, error_body)

        if exc.code == 404:
            logger.info("Gemini modelo %s não encontrado, tentando próximo", model)
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        return ""


def _parse_error_detail(exc: urllib.error.HTTPError, error_body: str) -> str:
    err_detail = f"HTTP {exc.code}: {exc.reason}"
    try:
        err_data = json.loads(error_body)
        msg = err_data.get("error", {}).get("message")
        if msg:
            return msg
    except Exception:
        if error_body:
            return f"HTTP {exc.code}: {error_body[:200]}"
    return err_detail
