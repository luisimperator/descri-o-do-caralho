"""Template renderer for the final YouTube description."""

from .content import format_timestamp
from .names import ValidatedName


def render_description(
    title: str,
    main_topic: str,
    ocr_short: str,
    summary: str,
    participants: list[ValidatedName],
    chapters: list[dict],
    keywords: list[str],
    channel_name: str,
    asr_generated: bool = False,
) -> str:
    """Render the final description following the mandatory template.

    Template structure:
        [Título sugerido] | [Tópico Principal]
        OCR: [ocr_text_short]
        No episódio de hoje, [Nomes] exploram [resumo].
        Participantes
        Tópicos Abordados
        Palavras-chave
        Hashtags
    """
    lines: list[str] = []

    # --- Header ---
    lines.append(f"{title} | {main_topic}")
    lines.append("")

    # --- OCR ---
    if ocr_short:
        lines.append(f"OCR: {ocr_short}")
        lines.append("")

    # --- Intro paragraph ---
    names_str = _format_name_list([p.canonical for p in participants])
    lines.append(f"No episódio de hoje, {names_str} exploram {summary}")
    lines.append("")

    # --- Participants ---
    if participants:
        lines.append("Participantes")
        for p in participants:
            bio = p.mini_bio or "Profissional e participante do programa"
            lines.append(f"• {p.canonical} — {bio}")
        lines.append("")

    # --- Chapters ---
    lines.append("Tópicos Abordados:")
    for ch in chapters:
        ts = format_timestamp(ch["start"])
        lines.append(f"{ts} {ch['title']}")
    lines.append("")

    # --- Keywords ---
    if keywords:
        lines.append(f"Palavras-chave: {', '.join(keywords)}")
        lines.append("")

    # --- Hashtags ---
    hashtags = _build_hashtags(channel_name, main_topic, keywords)
    lines.append(hashtags)

    # --- ASR notice ---
    if asr_generated:
        lines.append("")
        lines.append(
            "(Transcrição gerada automaticamente — "
            "pode conter imprecisões.)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_name_list(names: list[str]) -> str:
    """Format a list of names with commas and 'e'."""
    if not names:
        return "os participantes"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " e " + names[-1]


def _build_hashtags(
    channel: str, topic: str, keywords: list[str], max_tags: int = 8
) -> str:
    """Build a hashtag line."""
    tags: list[str] = ["#Podcast"]

    # Channel hashtag
    channel_tag = _to_hashtag(channel)
    if channel_tag and channel_tag not in tags:
        tags.append(channel_tag)

    # Topic hashtag
    topic_tag = _to_hashtag(topic)
    if topic_tag and topic_tag not in tags:
        tags.append(topic_tag)

    # Keyword hashtags
    for kw in keywords:
        tag = _to_hashtag(kw)
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= max_tags:
            break

    return " ".join(tags)


def _to_hashtag(text: str) -> str:
    """Convert text to a #CamelCase hashtag."""
    if not text:
        return ""
    words = text.split()
    camel = "".join(w.capitalize() for w in words if w.isalnum() or w.isalpha())
    if not camel:
        return ""
    return f"#{camel}"
