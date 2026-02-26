"""Content generation: summaries, chapters, keywords, topics, social links."""

import re
from collections import Counter


def generate_summary(
    title: str,
    description: str,
    transcript: str,
    participant_names: list[str],
    max_words: int = 40,
) -> str:
    """Generate a SHORT summary for the intro paragraph.

    Returns a concise description of the episode content (~30-40 words).
    This is used after "No episódio de hoje, X exploram ..."
    """
    names_str = ", ".join(participant_names) if participant_names else "os participantes"

    # Try to extract a good summary from the description first
    if description:
        summary = _extract_intro_from_description(description, title, max_words)
        if summary:
            return summary

    # Try from transcript
    if transcript:
        summary = _extract_intro_from_text(transcript, title, max_words)
        if summary:
            return summary

    # Fallback: use the topic from title
    topic = _clean_title_for_summary(title)
    return f"temas como {topic}."


def generate_topics(
    title: str,
    description: str,
    transcript: str,
    keywords: list[str],
    max_topics: int = 6,
) -> list[str]:
    """Extract main discussion topics from the video content.

    Returns a list of meaningful topic phrases (not single words).
    """
    topics: list[str] = []

    # Extract topic phrases from the description
    if description:
        desc_topics = _extract_topics_from_description(description)
        for t in desc_topics:
            if t not in topics and len(topics) < max_topics:
                topics.append(t)

    # Extract from title parts (skip the show name — usually first part)
    for sep in ["|", " - ", "–"]:
        if sep in title:
            parts = [p.strip() for p in title.split(sep) if len(p.strip()) > 3]
            # Skip first part (show name), use the rest as topics
            for p in parts[1:]:
                if p not in topics and len(topics) < max_topics:
                    topics.append(p)

    # Extract from transcript hints
    hints = _extract_topic_hints(transcript)
    for h in hints:
        if h not in topics and len(topics) < max_topics:
            topics.append(h)

    # Fill from keyword pairs if still too few
    if len(topics) < 3:
        for kw in keywords:
            topic = kw.capitalize()
            # Skip single short words, prefer meaningful terms
            if len(topic) > 5 and topic not in topics and len(topics) < max_topics:
                topics.append(topic)

    return topics[:max_topics]


def generate_chapters(
    existing_chapters: list[dict],
    transcript: str,
    duration: int,
    max_chapters: int = 25,
) -> list[dict]:
    """Return a list of chapters with start times and titles."""
    if existing_chapters:
        return existing_chapters[:max_chapters]

    if duration <= 0:
        return [{"start": 0, "title": "Introdução"}]

    interval = 240  # 4 minutes
    chapters = [{"start": 0, "title": "Introdução"}]

    topic_hints = _extract_topic_hints(transcript)

    # Generic section names for when there are no topic hints
    generic_titles = [
        "Contexto e cenário atual",
        "Desenvolvimento do tema",
        "Análise e perspectivas",
        "Aprofundamento",
        "Debates e reflexões",
        "Desdobramentos",
        "Perspectivas práticas",
        "Considerações importantes",
        "Visão estratégica",
        "Panorama geral",
    ]

    current = interval
    hint_idx = 0
    generic_idx = 0
    while current < duration and len(chapters) < max_chapters:
        if hint_idx < len(topic_hints):
            title = topic_hints[hint_idx]
            hint_idx += 1
        elif generic_idx < len(generic_titles):
            title = generic_titles[generic_idx]
            generic_idx += 1
        else:
            title = f"Continuação ({len(chapters) + 1})"
        chapters.append({"start": current, "title": title})
        current += interval

    # Add a conclusion chapter near the end
    if duration > 0 and chapters[-1]["start"] < duration - 60:
        chapters.append({"start": duration - 60, "title": "Conclusão"})

    return chapters


def format_timestamp(seconds: int) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def generate_keywords(
    title: str,
    description: str,
    transcript: str,
    ocr_text: str,
    channel_name: str,
    max_keywords: int = 15,
) -> list[str]:
    """Combine terms from transcript, OCR, and metadata into a keyword list."""
    all_text = f"{title} {description} {transcript} {ocr_text}"
    all_text_lower = all_text.lower()

    words = re.findall(r"\b[a-záàâãéèêíïóôõúüç]{4,}\b", all_text_lower)
    counter = Counter(words)

    # Remove stop words, URLs, and garbage
    stop_words = {
        "para", "como", "mais", "está", "isso", "esse", "essa", "esses",
        "essas", "aqui", "aquele", "aquela", "então", "porque", "quando",
        "onde", "qual", "quais", "cada", "todo", "toda", "todos", "todas",
        "muito", "muita", "muitos", "muitas", "outro", "outra", "outros",
        "outras", "mesmo", "mesma", "ainda", "sobre", "pode", "entre",
        "depois", "antes", "agora", "você", "vocês", "nosso", "nossa",
        "dele", "dela", "deles", "delas", "também", "fazer", "falar",
        "coisa", "coisas", "gente", "tinha", "seria", "sido", "sendo",
        "vamos", "ponto", "tipo", "acho", "vezes", "parte", "forma",
        "exemplo", "pessoas", "tempo", "anos", "hoje", "nesse", "nessa",
        "pela", "pelo", "numa", "desse", "dessa", "algo", "assim",
        "bem", "ter", "tem", "são", "uma", "uns", "umas",
        # URLs and web garbage
        "https", "http", "www", "com", "org", "youtube", "youtu",
        "watch", "channel", "playlist", "subscribe",
        # Social media
        "instagram", "twitter", "facebook", "whatsapp", "telegram",
        "linkedin", "tiktok",
        # Generic podcast words
        "canal", "vídeo", "video", "episódio", "episodio", "podcast",
        "inscreva", "compartilhe", "curtir", "like", "link",
        "descrição", "descricao", "comentário", "comentario",
        # Common verbs
        "disse", "falou", "falar", "dizer", "saber", "poder", "querer",
        "estar", "ficar", "olhar", "pensar", "achar", "deixar", "passar",
        "conhecer", "entender", "começar", "chegar", "colocar",
    }
    for sw in stop_words:
        counter.pop(sw, None)

    # Remove channel name words
    for w in channel_name.lower().split():
        if len(w) > 2:
            counter.pop(w, None)

    ocr_words = re.findall(r"\b[a-záàâãéèêíïóôõúüç]{4,}\b", ocr_text.lower())
    for w in ocr_words:
        if w in counter:
            counter[w] += 5

    title_words = re.findall(r"\b[a-záàâãéèêíïóôõúüç]{4,}\b", title.lower())
    for w in title_words:
        if w in counter:
            counter[w] += 10

    top = [word for word, _ in counter.most_common(max_keywords + 5)]
    result = list(dict.fromkeys(top[:max_keywords]))
    return result


def extract_social_links(description: str) -> dict:
    """Extract social media handles/URLs from the video description."""
    if not description:
        return {}

    links: dict[str, str] = {}

    ig = re.search(r"instagram\.com/([a-zA-Z0-9_.]+)", description, re.IGNORECASE)
    if not ig:
        ig = re.search(r"instagram:\s*@([a-zA-Z0-9_.]+)", description, re.IGNORECASE)
    if ig:
        links["instagram"] = f"@{ig.group(1)}"

    tw = re.search(r"(?:twitter|x)\.com/([a-zA-Z0-9_]+)", description, re.IGNORECASE)
    if not tw:
        tw = re.search(r"twitter:\s*@([a-zA-Z0-9_]+)", description, re.IGNORECASE)
    if tw:
        links["twitter"] = f"@{tw.group(1)}"

    fb = re.search(r"facebook\.com/([a-zA-Z0-9.]+)", description, re.IGNORECASE)
    if fb:
        links["facebook"] = fb.group(1)

    return links


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_intro_from_description(description: str, title: str, max_words: int) -> str:
    """Extract a clean intro summary from the video description.

    Looks for the first meaningful sentence about the episode content.
    """
    # Remove URLs from description
    clean = re.sub(r"https?://\S+", "", description)
    # Remove emoji
    clean = re.sub(r"[\U0001f300-\U0001f9ff]", "", clean)

    sentences = re.split(r"[.!?\n]+", clean)
    # Look for a sentence that describes the content (contains topic keywords)
    title_words = set(w.lower() for w in title.split() if len(w) > 3)

    for sent in sentences:
        sent = sent.strip()
        words = sent.split()
        if len(words) < 5 or len(words) > 50:
            continue
        # Skip sentences that are just names or social media
        sent_lower = sent.lower()
        if any(skip in sent_lower for skip in ["inscreva", "siga", "curta", "@", "http"]):
            continue
        # Prefer sentences with title word overlap
        overlap = sum(1 for w in words if w.lower() in title_words)
        if overlap >= 1:
            # Take the sentence, truncated to max_words
            result_words = words[:max_words]
            result = " ".join(result_words)
            if not result.endswith("."):
                result += "."
            return result

    # Fallback: take first reasonable sentence
    for sent in sentences:
        sent = sent.strip()
        words = sent.split()
        if 5 <= len(words) <= 50:
            sent_lower = sent.lower()
            if not any(skip in sent_lower for skip in ["inscreva", "siga", "curta", "@", "http"]):
                result_words = words[:max_words]
                result = " ".join(result_words)
                if not result.endswith("."):
                    result += "."
                return result

    return ""


def _extract_intro_from_text(text: str, title: str, max_words: int) -> str:
    """Extract a summary sentence from transcript text."""
    sentences = re.split(r"[.!?\n]+", text)
    title_words = set(w.lower() for w in title.split() if len(w) > 3)

    for sent in sentences[:20]:  # only check first 20 sentences
        sent = sent.strip()
        words = sent.split()
        if len(words) < 8 or len(words) > 40:
            continue
        overlap = sum(1 for w in words if w.lower() in title_words)
        if overlap >= 1:
            result_words = words[:max_words]
            result = " ".join(result_words)
            if not result.endswith("."):
                result += "."
            return result

    return ""


def _clean_title_for_summary(title: str) -> str:
    """Remove show name from title, keeping the topic."""
    for sep in ["|", " - ", ":", "–"]:
        if sep in title:
            parts = [p.strip() for p in title.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[-1].lower()
    return title.lower()


def _extract_topics_from_description(description: str) -> list[str]:
    """Extract meaningful topic phrases from the description."""
    topics: list[str] = []

    # Remove URLs and emojis
    clean = re.sub(r"https?://\S+", "", description)
    clean = re.sub(r"[\U0001f300-\U0001f9ff]", "", clean)

    # Look for topic-indicating patterns in Portuguese
    patterns = [
        r"(?:sobre|temas?:?)\s+([^,.!?\n]{5,50})",
        r"(?:aborda|discute|explora|analisa)\w*\s+([^,.!?\n]{5,50})",
        r"(?:bate-papo|conversa|entrevista)\s+(?:sobre|com foco em)\s+([^,.!?\n]{5,50})",
        r"(?:práticas? de|estratégias? de|gestão de)\s+([^,.!?\n]{5,40})",
    ]

    for pat in patterns:
        for m in re.finditer(pat, clean, re.IGNORECASE):
            topic = m.group(1).strip().rstrip(",.")
            if topic and len(topic) > 5 and topic not in topics:
                topics.append(topic.capitalize())
            if len(topics) >= 6:
                return topics

    return topics


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 5]


def _extract_topic_hints(transcript: str, max_hints: int = 20) -> list[str]:
    """Extract potential topic phrases from transcript for chapter titles."""
    if not transcript:
        return []

    patterns = [
        r"(?:vamos falar|vamos conversar) (?:sobre|de) ([^,.!?]{5,40})",
        r"(?:o tema|o assunto|o tópico) (?:é|de hoje é) ([^,.!?]{5,40})",
        r"(?:primeiro ponto|segundo ponto|terceiro ponto)[:\s]+([^,.!?]{5,40})",
        r"(?:a questão|o ponto) (?:é|aqui é) ([^,.!?]{5,40})",
    ]

    hints: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, transcript, re.IGNORECASE):
            hint = m.group(1).strip().capitalize()
            if hint and hint not in hints:
                hints.append(hint)
            if len(hints) >= max_hints:
                return hints

    return hints
