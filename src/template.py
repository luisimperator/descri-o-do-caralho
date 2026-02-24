"""Template renderer for the final YouTube description."""

from .content import format_timestamp
from .names import ValidatedName


def render_description(
    title: str,
    main_topic: str,
    summary: str,
    participants: list[ValidatedName],
    topics: list[str],
    chapters: list[dict],
    keywords: list[str],
    channel_name: str,
    social_links: dict | None = None,
    asr_generated: bool = False,
) -> str:
    """Render the final description following the mandatory template.

    Only participants with present_in_video=True are shown.
    """
    social_links = social_links or {}
    # Filter to only participants actually present in the video
    present = [p for p in participants if p.present_in_video]

    lines: list[str] = []

    # --- Header ---
    lines.append(f"\U0001f399\ufe0f {title} | {main_topic}")
    lines.append("")

    # --- Intro paragraph ---
    names_str = _format_name_list([p.canonical for p in present])
    lines.append(
        f"No episódio de hoje, {names_str} exploram {summary}"
    )

    # Add "Desde X até Y" sentence if we have enough topics
    if len(topics) >= 2:
        lines.append(
            f"Desde {topics[0].lower()} até {topics[1].lower()}, "
            f"este bate-papo oferece insights valiosos sobre {main_topic.lower()}."
        )
    lines.append("")

    # --- Tópicos Abordados ---
    if topics:
        lines.append("\U0001f50d Tópicos Abordados:")
        for t in topics:
            lines.append(t)
        lines.append("")

    # --- Participantes ---
    if present:
        lines.append("\U0001f464 Participantes")
        for p in present:
            role_part = p.role if p.role else "Participante"
            bio_part = p.mini_bio or "Participante do programa"
            lines.append(f"{p.canonical}, {role_part}, {bio_part}")
        lines.append("")

    # --- Palavras-chave ---
    if keywords:
        kw_display = keywords[:8]
        lines.append(f"\U0001f4cc Palavras-chave: {', '.join(kw_display)}")
        lines.append("")

    # --- CTA "Não perca!" ---
    lines.append(
        f"\u2728 Não perca! Se você está interessado em {main_topic.lower()}, "
        f"este episódio é para você. Assista até o final!"
    )
    lines.append("")

    # --- Capítulos ---
    lines.append("\U0001f516 Capítulos:")
    for ch in chapters:
        ts = format_timestamp(ch["start"])
        lines.append(f"{ts} - {ch['title']}")
    lines.append("")

    # --- Engagement CTA ---
    lines.append(
        "\U0001f44d Gostou do vídeo? Inscreva-se no nosso canal, deixe um like "
        "e compartilhe com seus amigos! Deixe também seu comentário abaixo "
        "sobre o que achou do episódio ou sugestões para os próximos temas."
    )
    lines.append("")

    # --- Social links ---
    lines.append("\U0001f517 Siga-nos nas redes sociais:")
    if social_links.get("instagram"):
        lines.append(f"Instagram: {social_links['instagram']}")
    else:
        lines.append("Instagram: @seuInstagram")
    if social_links.get("twitter"):
        lines.append(f"Twitter: {social_links['twitter']}")
    else:
        lines.append("Twitter: @seuTwitter")
    if social_links.get("facebook"):
        lines.append(f"Facebook: {social_links['facebook']}")
    else:
        lines.append("Facebook: SeuFacebook")
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

    channel_tag = _to_hashtag(channel)
    if channel_tag and channel_tag not in tags:
        tags.append(channel_tag)

    topic_tag = _to_hashtag(topic)
    if topic_tag and topic_tag not in tags:
        tags.append(topic_tag)

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
