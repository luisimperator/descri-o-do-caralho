"""Gemini AI integration for research and content generation."""

import json
import logging
import os
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)


def gemini_available() -> bool:
    """Check if Gemini API key is configured."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        logger.warning("GEMINI_API_KEY não configurada — IA desabilitada")
    else:
        logger.info("GEMINI_API_KEY configurada (%s...)", key[:8])
    return bool(key)


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

def _call_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Call Gemini API and return the text response."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("_call_gemini: sem API key, pulando")
        return ""

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
            logger.error("Gemini retornou sem candidates: %s", json.dumps(data, ensure_ascii=False)[:500])
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            logger.error("Gemini retornou candidates sem parts: %s", json.dumps(candidates[0], ensure_ascii=False)[:500])
            return ""
        text = parts[0].get("text", "")
        logger.info("Gemini respondeu OK (%d chars): %.120s...", len(text), text)
        return text
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error("Gemini HTTP %d: %s — %s", exc.code, exc.reason, error_body)
    except urllib.error.URLError as exc:
        logger.error("Gemini URLError (rede?): %s", exc.reason)
    except Exception as exc:
        logger.error("Gemini erro inesperado: %s", exc)

    return ""
