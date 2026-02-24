"""Tests for the content generation module."""

from src.content import (
    format_timestamp,
    generate_chapters,
    generate_keywords,
    generate_summary,
    generate_topics,
    extract_social_links,
)


def test_format_timestamp_minutes():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(600) == "10:00"


def test_format_timestamp_hours():
    assert format_timestamp(3661) == "01:01:01"
    assert format_timestamp(7200) == "02:00:00"


def test_generate_chapters_uses_existing():
    existing = [
        {"start": 0, "title": "Intro"},
        {"start": 120, "title": "Part 1"},
    ]
    result = generate_chapters(existing, "", 600)
    assert result == existing


def test_generate_chapters_auto_segments():
    result = generate_chapters([], "", 900)
    assert result[0]["start"] == 0
    assert result[0]["title"] == "Introdução"
    assert len(result) >= 3  # 900s / 240s = ~3.75


def test_generate_keywords_returns_list():
    kw = generate_keywords(
        title="Investimentos em Ações",
        description="Falamos sobre bolsa de valores",
        transcript="mercado financeiro renda variável investimentos ações",
        ocr_text="Ações Investimentos",
        channel_name="FinançasTV",
    )
    assert isinstance(kw, list)
    assert len(kw) > 0


def test_generate_summary_fallback():
    summary = generate_summary(
        title="Título",
        description="",
        transcript="",
        participant_names=["João Silva"],
    )
    assert "João Silva" in summary


def test_generate_summary_with_content():
    summary = generate_summary(
        title="Mercado Financeiro",
        description="Análise completa do mercado",
        transcript=(
            "Hoje vamos analisar o mercado financeiro brasileiro. "
            "A bolsa subiu 2 por cento esta semana. "
            "Os investidores estão otimistas com os resultados."
        ),
        participant_names=["Ana Costa"],
        max_words=50,
    )
    assert len(summary.split()) <= 55  # allow small margin


def test_generate_topics_from_title():
    topics = generate_topics(
        title="Podcast | Economia | Mercado",
        description="",
        transcript="",
        keywords=["investimento", "bolsa"],
    )
    assert len(topics) >= 2
    assert any("Economia" in t for t in topics)


def test_generate_topics_fills_from_keywords():
    topics = generate_topics(
        title="Podcast Simples",
        description="",
        transcript="",
        keywords=["economia", "mercado", "investimento"],
    )
    assert len(topics) >= 3


def test_extract_social_links_instagram():
    desc = "Siga no Instagram: instagram.com/meucanal"
    links = extract_social_links(desc)
    assert links.get("instagram") == "@meucanal"


def test_extract_social_links_twitter():
    desc = "Twitter: twitter.com/meutwitter"
    links = extract_social_links(desc)
    assert links.get("twitter") == "@meutwitter"


def test_extract_social_links_empty():
    links = extract_social_links("")
    assert links == {}


def test_extract_social_links_facebook():
    desc = "Facebook: facebook.com/meupodcast"
    links = extract_social_links(desc)
    assert links.get("facebook") == "meupodcast"
