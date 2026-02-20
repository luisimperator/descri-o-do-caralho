"""OCR processing for YouTube thumbnails."""

import re
import subprocess
from pathlib import Path


def extract_text_from_thumbnail(image_path: str) -> dict:
    """Run OCR on a thumbnail image and return full + short text.

    Uses Tesseract OCR as the primary engine.

    Returns:
        dict with keys:
            ocr_text_full  – all recognised text
            ocr_text_short – cleaned/shortened version for display
    """
    if not image_path or not Path(image_path).exists():
        return {"ocr_text_full": "", "ocr_text_short": ""}

    text = _run_tesseract(image_path)
    full = _clean_ocr_text(text)
    short = _shorten(full)
    return {"ocr_text_full": full, "ocr_text_short": short}


def extract_name_candidates_from_ocr(ocr_full: str) -> list[str]:
    """Identify potential person names in OCR output.

    Heuristic: sequences of 2-5 capitalised words.
    """
    if not ocr_full:
        return []

    pattern = r"\b(?:[A-ZÀ-Ü][a-zà-ü]+(?:\s+|$)){2,5}"
    matches = re.findall(pattern, ocr_full)
    return [m.strip() for m in matches if m.strip()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_tesseract(image_path: str) -> str:
    """Execute tesseract and return stdout text."""
    cmd = [
        "tesseract",
        image_path,
        "stdout",
        "-l", "por+eng",
        "--psm", "3",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except FileNotFoundError:
        # Tesseract not installed – return empty gracefully
        return ""
    except subprocess.TimeoutExpired:
        return ""


def _clean_ocr_text(raw: str) -> str:
    """Remove noise from OCR output."""
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", raw)
    # Remove lines that are just symbols / single chars
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and len(ln.strip()) > 1
    ]
    return "\n".join(lines)


def _shorten(full: str, max_chars: int = 120) -> str:
    """Create a compact version of the OCR text."""
    one_line = " | ".join(full.splitlines())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3].rsplit(" ", 1)[0] + "..."
