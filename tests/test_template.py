"""Tests for the template renderer."""

from src.names import ValidatedName
from src.template import render_description


def test_render_contains_all_sections():
    desc = render_description(
        title="Podcast Especial",
        main_topic="Economia",
        summary="a situação econômica brasileira e perspectivas para 2025.",
        participants=[
            ValidatedName(
                canonical="João Silva",
                source="google",
                trust_level="high",
                mini_bio="Professor da USP e especialista em mercado financeiro",
                role="Economista",
                present_in_video=True,
            ),
        ],
        topics=["Economia brasileira", "Perspectivas para 2025", "Mercado financeiro"],
        chapters=[
            {"start": 0, "title": "Introdução"},
            {"start": 240, "title": "Análise"},
        ],
        keywords=["economia", "brasil", "mercado"],
        channel_name="PodcastBR",
    )

    # Header with emoji
    assert "Podcast Especial | Economia" in desc
    # Participant present
    assert "João Silva" in desc
    assert "Economista" in desc
    # Topics section
    assert "Tópicos Abordados:" in desc
    assert "Economia brasileira" in desc
    # Chapters section
    assert "Capítulos:" in desc
    assert "00:00 - Introdução" in desc
    assert "04:00 - Análise" in desc
    # Keywords
    assert "Palavras-chave:" in desc
    # CTA
    assert "Não perca!" in desc
    # Engagement
    assert "Inscreva-se" in desc
    # Social links
    assert "Siga-nos nas redes sociais:" in desc
    # Hashtags
    assert "#Podcast" in desc


def test_render_filters_cited_participants():
    desc = render_description(
        title="Vídeo Especial",
        main_topic="Geral",
        summary="um tema geral.",
        participants=[
            ValidatedName(
                canonical="Host do Canal",
                source="title",
                trust_level="high",
                mini_bio="Apresentador do programa",
                role="Apresentador",
                present_in_video=True,
            ),
            ValidatedName(
                canonical="Pessoa Citada",
                source="transcript",
                trust_level="medium",
                mini_bio="Político brasileiro",
                role="Político",
                present_in_video=False,
            ),
        ],
        topics=["Política"],
        chapters=[{"start": 0, "title": "Introdução"}],
        keywords=["política"],
        channel_name="Canal",
    )
    # Present participant should appear in Participantes section
    assert "Host do Canal" in desc
    # Cited participant should NOT appear in Participantes section
    assert "Pessoa Citada" not in desc


def test_render_no_participants():
    desc = render_description(
        title="Vídeo Sem Convidados",
        main_topic="Geral",
        summary="um tema geral.",
        participants=[],
        topics=["Geral"],
        chapters=[{"start": 0, "title": "Introdução"}],
        keywords=["geral"],
        channel_name="Canal",
    )
    assert "os participantes" in desc


def test_render_asr_notice():
    desc = render_description(
        title="Teste",
        main_topic="Teste",
        summary="testes automatizados.",
        participants=[],
        topics=[],
        chapters=[{"start": 0, "title": "Início"}],
        keywords=[],
        channel_name="Canal",
        asr_generated=True,
    )
    assert "Transcrição gerada automaticamente" in desc


def test_render_social_links_from_description():
    desc = render_description(
        title="Teste",
        main_topic="Teste",
        summary="um teste.",
        participants=[],
        topics=[],
        chapters=[{"start": 0, "title": "Início"}],
        keywords=[],
        channel_name="Canal",
        social_links={"instagram": "@meucanal", "twitter": "@meutwitter"},
    )
    assert "@meucanal" in desc
    assert "@meutwitter" in desc


def test_render_desde_ate_with_topics():
    desc = render_description(
        title="Podcast",
        main_topic="Economia",
        summary="a economia global.",
        participants=[],
        topics=["Inflação", "Juros", "Câmbio"],
        chapters=[{"start": 0, "title": "Introdução"}],
        keywords=[],
        channel_name="Canal",
    )
    assert "Desde inflação até juros" in desc
    assert "insights valiosos sobre economia" in desc
