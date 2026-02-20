"""Tests for the OCR module."""

from src.ocr import extract_name_candidates_from_ocr


def test_extract_names_basic():
    text = "João Pedro Silva e Maria Fernanda no podcast"
    names = extract_name_candidates_from_ocr(text)
    assert any("João Pedro Silva" in n for n in names) or any(
        "Maria Fernanda" in n for n in names
    )


def test_extract_names_empty():
    assert extract_name_candidates_from_ocr("") == []


def test_extract_names_no_capitals():
    # Lowercase text should yield no names
    assert extract_name_candidates_from_ocr("sem nomes aqui") == []
