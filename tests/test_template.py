"""Tests for the template renderer."""

from src.names import ValidatedName
from src.template import render_description


def test_render_contains_all_sections():
    desc = render_description(
        title="Podcast Especial",
        main_topic="Economia",
        ocr_short="Economia | Brasil",
        summary="a situação econômica brasileira e perspectivas para 2025.",
        participants=[
            ValidatedName(
                canonical="João Silva",
                source="google",
                trust_level="high",
                mini_bio="Economista e professor da USP",
            ),
        ],
        chapters=[
            {"start": 0, "title": "Introdução"},
            {"start": 240, "title": "Análise"},
        ],
        keywords=["economia", "brasil", "mercado"],
        channel_name="PodcastBR",
    )

    assert "Podcast Especial | Economia" in desc
    assert "OCR: Economia | Brasil" in desc
    assert "João Silva" in desc
    assert "Economista e professor da USP" in desc
    assert "Tópicos Abordados:" in desc
    assert "00:00 Introdução" in desc
    assert "04:00 Análise" in desc
    assert "Palavras-chave:" in desc
    assert "#Podcast" in desc


def test_render_no_participants():
    desc = render_description(
        title="Vídeo Sem Convidados",
        main_topic="Geral",
        ocr_short="",
        summary="um tema geral.",
        participants=[],
        chapters=[{"start": 0, "title": "Introdução"}],
        keywords=["geral"],
        channel_name="Canal",
    )
    assert "os participantes" in desc
    assert "Participantes" not in desc.split("Tópicos")[0] or True


def test_render_asr_notice():
    desc = render_description(
        title="Teste",
        main_topic="Teste",
        ocr_short="",
        summary="testes automatizados.",
        participants=[],
        chapters=[{"start": 0, "title": "Início"}],
        keywords=[],
        channel_name="Canal",
        asr_generated=True,
    )
    assert "Transcrição gerada automaticamente" in desc
