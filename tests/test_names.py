"""Tests for the names module."""

from src.names import (
    collect_name_candidates,
    ValidatedName,
    _is_blocked_response,
    _filter_non_names,
    _extract_role,
)


def test_collect_candidates_deduplicates():
    candidates = collect_name_candidates(
        title="João Silva conversa com Maria Costa",
        description="João Silva recebe Maria Costa",
        ocr_names=["João Silva"],
        transcript="",
    )
    # Should have both names but no duplicates
    lower_set = {c.lower() for c in candidates}
    assert "joão silva" in lower_set
    assert "maria costa" in lower_set
    # Count occurrences — each name at most once
    assert sum(1 for c in candidates if c.lower() == "joão silva") == 1


def test_collect_candidates_from_transcript_repetition():
    transcript = (
        "Carlos Mendes falou sobre economia. "
        "Carlos Mendes disse que o mercado vai subir. "
        "Segundo Carlos Mendes, os dados confirmam."
    )
    candidates = collect_name_candidates(
        title="",
        description="",
        ocr_names=[],
        transcript=transcript,
    )
    assert any("Carlos Mendes" in c for c in candidates)


def test_filter_non_names_removes_channel_name():
    names = ["Arbitragem Gestao", "João Silva", "Maria Costa"]
    filtered = _filter_non_names(names, channel_name="Arbitragem Gestao")
    assert "Arbitragem Gestao" not in filtered
    assert "João Silva" in filtered
    assert "Maria Costa" in filtered


def test_filter_non_names_removes_role_words():
    names = ["Entrevistador Palestrante", "João Silva"]
    filtered = _filter_non_names(names, channel_name="")
    assert "Entrevistador Palestrante" not in filtered
    assert "João Silva" in filtered


def test_is_blocked_response_detects_captcha():
    garbage = (
        "c || {cap:0}; table,div,span,p{display:none} "
        "Clique aqui se o redirecionamento não iniciar em"
    )
    assert _is_blocked_response(garbage) is True


def test_is_blocked_response_allows_real():
    real = "João Silva é um economista brasileiro. Professor da USP."
    assert _is_blocked_response(real) is False


def test_extract_role_whole_word():
    text = "João Silva é um economista e professor"
    role = _extract_role(text)
    assert role == "Economista"


def test_extract_role_no_substring_match():
    # "ator" should not match inside "redirecionamento" or similar
    text = "Clique aqui se o redirecionamento não iniciar em breve"
    role = _extract_role(text)
    assert role == ""


def test_collect_candidates_filters_channel_name():
    candidates = collect_name_candidates(
        title="Podcast Arbitragem Gestao",
        description="",
        ocr_names=[],
        transcript="",
        channel_name="Arbitragem Gestao",
    )
    assert not any("arbitragem" in c.lower() for c in candidates)
