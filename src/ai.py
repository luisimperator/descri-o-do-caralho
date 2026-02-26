"""Gemini AI integration for research and content generation."""

import json
import logging
import os
import urllib.error
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)

# Track errors so the pipeline can include them in diagnostics
_recent_errors: list[str] = []

# Models to try in order of preference
_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]


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


def test_gemini() -> dict:
    """Test Gemini API connectivity. Returns status dict."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "GEMINI_API_KEY não configurada"}

    for model in _MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={api_key}"
        )
        body = json.dumps({
            "contents": [{"parts": [{"text": "Responda apenas: OK"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 10},
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return {"ok": True, "model": model, "response": text.strip()}
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            # Try to extract the actual error message
            err_msg = f"HTTP {exc.code}"
            try:
                err_data = json.loads(error_body)
                err_msg = err_data.get("error", {}).get("message", err_msg)
            except Exception:
                err_msg = f"HTTP {exc.code}: {error_body[:150]}"
            # If it's a model-not-found error, try next model
            if exc.code == 404:
                logger.info("Modelo %s não encontrado, tentando próximo...", model)
                continue
            return {"ok": False, "model": model, "error": err_msg}
        except Exception as exc:
            return {"ok": False, "model": model, "error": str(exc)}

    return {"ok": False, "error": "Nenhum modelo Gemini disponível"}


def research_participant(name: str, channel_name: str) -> dict:
    """Research a participant using Gemini to find their role and bio.

    Returns {"role": "...", "bio": "..."}.
    """
    prompt = (
        f"Pesquise sobre '{name}' que participou do canal/podcast '{channel_name}' no YouTube.\n"
        f"Responda APENAS em JSON com este formato exato:\n"
        f'{{"role": "cargo profissional em 1-2 palavras", "bio": "resumo profissional em até 15 palavras"}}\n'
        f"Se não encontrar informações específicas, use:\n"
        f'{{"role": "Profissional", "bio": "Participante do programa {channel_name}"}}\n'
        f"Responda APENAS o JSON, sem markdown, sem explicação."
    )
    result = _call_gemini(prompt)
    if not result:
        return {"role": "", "bio": "Participante do programa"}

    try:
        # Try to parse JSON from response
        clean = result.strip()
        # Remove markdown code block if present
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
        return {
            "role": data.get("role", ""),
            "bio": data.get("bio", "Participante do programa"),
        }
    except (json.JSONDecodeError, KeyError):
        return {"role": "", "bio": "Participante do programa"}


def generate_chapter_titles(
    transcript: str,
    title: str,
    duration: int,
    existing_chapters: list[dict],
) -> list[dict]:
    """Use Gemini to generate meaningful chapter titles from transcript.

    If existing_chapters have timestamps, keeps them and improves titles.
    Otherwise generates chapters with timestamps.
    """
    if not transcript:
        return existing_chapters

    # Truncate transcript to ~4000 chars for the API
    transcript_sample = transcript[:4000]
    duration_min = duration // 60

    if existing_chapters:
        # Improve existing chapter titles
        chapters_text = "\n".join(
            f"{ch['start']}s - {ch['title']}" for ch in existing_chapters
        )
        prompt = (
            f"Este é um podcast chamado '{title}' com {duration_min} minutos.\n"
            f"Capítulos atuais:\n{chapters_text}\n\n"
            f"Trecho da transcrição:\n{transcript_sample}\n\n"
            f"Melhore os títulos dos capítulos para serem mais descritivos. "
            f"Mantenha os mesmos timestamps. "
            f"Responda APENAS em JSON array com este formato:\n"
            f'[{{"start": 0, "title": "Introdução"}}, {{"start": 240, "title": "Título descritivo"}}]\n'
            f"Responda APENAS o JSON array, sem markdown, sem explicação."
        )
    else:
        # Generate chapters from scratch
        prompt = (
            f"Este é um podcast chamado '{title}' com {duration_min} minutos.\n"
            f"Trecho da transcrição:\n{transcript_sample}\n\n"
            f"Crie capítulos para este vídeo com timestamps e títulos descritivos. "
            f"O primeiro capítulo deve ser 0s (Introdução) e o último deve ser a Conclusão. "
            f"Crie entre 5 e 10 capítulos espaçados uniformemente.\n"
            f"Responda APENAS em JSON array com este formato:\n"
            f'[{{"start": 0, "title": "Introdução"}}, {{"start": 240, "title": "Título descritivo"}}]\n'
            f"Responda APENAS o JSON array, sem markdown, sem explicação."
        )

    result = _call_gemini(prompt)
    if not result:
        return existing_chapters

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        chapters = json.loads(clean)
        if isinstance(chapters, list) and chapters:
            return [
                {"start": int(ch.get("start", 0)), "title": ch.get("title", "")}
                for ch in chapters
                if isinstance(ch, dict)
            ]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    return existing_chapters


def generate_summary_ai(
    title: str,
    description: str,
    transcript: str,
    participant_names: list[str],
) -> str:
    """Use Gemini to generate a clean intro summary."""
    transcript_sample = transcript[:2000] if transcript else ""
    names = ", ".join(participant_names) if participant_names else "os participantes"

    prompt = (
        f"Podcast: '{title}'\n"
        f"Participantes: {names}\n"
        f"Descrição original: {description[:500]}\n"
        f"Trecho da transcrição: {transcript_sample}\n\n"
        f"Escreva um resumo de 1-2 frases (máximo 40 palavras) descrevendo "
        f"o conteúdo do episódio. O resumo vai ser usado depois de "
        f"'No episódio de hoje, {names} exploram ...'\n"
        f"Não inclua nomes dos participantes no resumo.\n"
        f"Responda APENAS o texto do resumo, sem aspas, sem explicação."
    )

    result = _call_gemini(prompt)
    if result:
        # Clean up: remove quotes, limit length
        clean = result.strip().strip('"').strip("'")
        words = clean.split()
        if len(words) > 50:
            clean = " ".join(words[:50]) + "."
        return clean

    return ""


def generate_topics_ai(
    title: str,
    description: str,
    transcript: str,
) -> list[str]:
    """Use Gemini to extract main discussion topics."""
    transcript_sample = transcript[:2000] if transcript else ""

    prompt = (
        f"Podcast: '{title}'\n"
        f"Descrição: {description[:500]}\n"
        f"Trecho da transcrição: {transcript_sample}\n\n"
        f"Liste de 4 a 6 tópicos principais abordados neste episódio.\n"
        f"Cada tópico deve ter 3-8 palavras.\n"
        f"Responda APENAS em JSON array de strings:\n"
        f'["Tópico 1", "Tópico 2", "Tópico 3", "Tópico 4"]\n'
        f"Responda APENAS o JSON array, sem markdown, sem explicação."
    )

    result = _call_gemini(prompt)
    if not result:
        return []

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        topics = json.loads(clean)
        if isinstance(topics, list):
            return [str(t) for t in topics[:6]]
    except (json.JSONDecodeError, ValueError):
        pass

    return []


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

# Cache the working model name so we don't retry 404s every call
_working_model: str | None = None


def _call_gemini(prompt: str) -> str:
    """Call Gemini API and return the text response.

    Tries multiple models if the primary one returns 404.
    """
    global _working_model

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("_call_gemini: sem API key, pulando")
        return ""

    # If we already found a working model, use it directly
    models_to_try = [_working_model] if _working_model else _MODELS

    for model in models_to_try:
        result = _call_gemini_single(prompt, model, api_key)
        if result is not None:
            _working_model = model
            return result
        # result is None means 404 (model not found), try next

    return ""


def _call_gemini_single(prompt: str, model: str, api_key: str) -> str | None:
    """Call a specific Gemini model.

    Returns the text response, "" on API error, or None if model not found (404).
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
            "maxOutputTokens": 1024,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
        logger.info("Gemini (%s) respondeu OK (%d chars): %.120s...", model, len(text), text)
        return text
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass

        # Parse error message from JSON response
        err_detail = f"HTTP {exc.code}: {exc.reason}"
        try:
            err_data = json.loads(error_body)
            err_detail = err_data.get("error", {}).get("message", err_detail)
        except Exception:
            if error_body:
                err_detail = f"HTTP {exc.code}: {error_body[:200]}"

        if exc.code == 404:
            logger.info("Modelo %s não encontrado (404), tentando próximo", model)
            return None  # Signal to try next model

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
