"""Content generation: summaries, chapters, keywords, topics, social links."""

import re
from collections import Counter


def generate_summary(
    title: str,
    description: str,
    transcript: str,
    participant_names: list[str],
    max_words: int = 150,
) -> str:
    """Generate a concise summary (max 150 words) of the video content."""
    source_text = f"{title}. {description}. {transcript}"

    sentences = _split_sentences(source_text)
    if not sentences:
        return f"Neste episódio, {', '.join(participant_names) or 'os participantes'} discutem {title}."

    scored = _score_sentences(sentences, title, participant_names)

    summary_parts: list[str] = []
    word_count = 0
    for sentence, _ in scored:
        words = sentence.split()
        if word_count + len(words) > max_words:
            break
        summary_parts.append(sentence)
        word_count += len(words)

    if not summary_parts:
        return f"Neste episódio, {', '.join(participant_names) or 'os participantes'} discutem {title}."

    return " ".join(summary_parts)


def generate_topics(
    title: str,
    description: str,
    transcript: str,
    keywords: list[str],
    max_topics: int = 6,
) -> list[str]:
    """Extract main discussion topics from the video content.

    Returns a list of topic strings (not timestamped).
    """
    topics: list[str] = []

    # Extract from title parts (split by separators)
    for sep in ["|", " - ", ":", "–"]:
        if sep in title:
            parts = [p.strip() for p in title.split(sep) if len(p.strip()) > 3]
            for p in parts:
                if p not in topics and len(topics) < max_topics:
                    topics.append(p)

    # Extract from transcript hints
    hints = _extract_topic_hints(transcript)
    for h in hints:
        if h not in topics and len(topics) < max_topics:
            topics.append(h)

    # Fill from keywords if needed (capitalize and group pairs)
    if len(topics) < 3:
        for kw in keywords:
            topic = kw.capitalize()
            if topic not in topics and len(topics) < max_topics:
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

    current = interval
    hint_idx = 0
    while current < duration and len(chapters) < max_chapters:
        if hint_idx < len(topic_hints):
            title = topic_hints[hint_idx]
            hint_idx += 1
        else:
            title = f"Parte {len(chapters) + 1}"
        chapters.append({"start": current, "title": title})
        current += interval

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
    }
    for sw in stop_words:
        counter.pop(sw, None)

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
    """Extract social media handles/URLs from the video description.

    Returns a dict with keys like 'instagram', 'twitter', 'facebook'.
    """
    if not description:
        return {}

    links: dict[str, str] = {}

    # Instagram — try URL first, then "Instagram: @handle"
    ig = re.search(r"instagram\.com/([a-zA-Z0-9_.]+)", description, re.IGNORECASE)
    if not ig:
        ig = re.search(r"instagram:\s*@([a-zA-Z0-9_.]+)", description, re.IGNORECASE)
    if ig:
        links["instagram"] = f"@{ig.group(1)}"

    # Twitter / X — try URL first, then "Twitter: @handle"
    tw = re.search(r"(?:twitter|x)\.com/([a-zA-Z0-9_]+)", description, re.IGNORECASE)
    if not tw:
        tw = re.search(r"twitter:\s*@([a-zA-Z0-9_]+)", description, re.IGNORECASE)
    if tw:
        links["twitter"] = f"@{tw.group(1)}"

    # Facebook
    fb = re.search(r"facebook\.com/([a-zA-Z0-9.]+)", description, re.IGNORECASE)
    if fb:
        links["facebook"] = fb.group(1)

    return links


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 5]


def _score_sentences(
    sentences: list[str],
    title: str,
    names: list[str],
) -> list[tuple[str, float]]:
    """Score sentences by relevance to the topic."""
    title_words = set(title.lower().split())
    name_words = {n.lower() for n in names}

    scored: list[tuple[str, float]] = []
    for i, sent in enumerate(sentences):
        score = 0.0
        words = set(sent.lower().split())

        overlap = len(words & title_words)
        score += overlap * 2.0

        for n in name_words:
            if n in sent.lower():
                score += 3.0

        score -= i * 0.1

        wc = len(sent.split())
        if 10 <= wc <= 30:
            score += 1.0

        scored.append((sent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


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
