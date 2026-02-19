"""Tests for the names module."""

from src.names import collect_name_candidates, ValidatedName


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
