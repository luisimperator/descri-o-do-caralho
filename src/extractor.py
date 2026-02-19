"""YouTube video data extraction using yt-dlp."""

import json
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoData:
    video_id: str = ""
    title: str = ""
    description: str = ""
    upload_date: str = ""
    channel: str = ""
    channel_url: str = ""
    thumbnail_url: str = ""
    thumbnail_path: str = ""
    duration: int = 0
    chapters: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    transcript: str = ""
    asr_generated: bool = False


def extract_video_data(youtube_url: str, output_dir: str | None = None) -> VideoData:
    """Extract metadata, thumbnail, and transcript from a YouTube video.

    Args:
        youtube_url: Full YouTube URL.
        output_dir: Directory for downloaded assets. Uses a temp dir if None.

    Returns:
        VideoData with all extracted fields populated.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="yt_desc_")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = VideoData()

    # --- Metadata extraction via yt-dlp ---
    meta = _fetch_metadata(youtube_url)
    data.video_id = meta.get("id", "")
    data.title = meta.get("title", "")
    data.description = meta.get("description", "")
    data.upload_date = _format_date(meta.get("upload_date", ""))
    data.channel = meta.get("channel", meta.get("uploader", ""))
    data.channel_url = meta.get("channel_url", "")
    data.thumbnail_url = meta.get("thumbnail", "")
    data.duration = int(meta.get("duration", 0))
    data.tags = meta.get("tags", []) or []

    # Chapters from metadata
    raw_chapters = meta.get("chapters") or []
    data.chapters = [
        {"start": int(ch.get("start_time", 0)), "title": ch.get("title", "")}
        for ch in raw_chapters
    ]

    # --- Thumbnail download ---
    if data.thumbnail_url:
        data.thumbnail_path = _download_thumbnail(
            data.thumbnail_url, out_path / f"{data.video_id}_thumb.jpg"
        )

    # --- Transcript extraction ---
    data.transcript, data.asr_generated = _fetch_transcript(youtube_url, out_path)

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_metadata(url: str) -> dict:
    """Run yt-dlp --dump-json and return parsed metadata."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _download_thumbnail(url: str, dest: Path) -> str:
    """Download thumbnail image to *dest* and return the path as string."""
    try:
        urllib.request.urlretrieve(url, str(dest))
        return str(dest)
    except Exception:
        return ""


def _fetch_transcript(url: str, out_dir: Path) -> tuple[str, bool]:
    """Try to fetch subtitles; fall back to auto-generated (ASR).

    Returns (transcript_text, asr_generated).
    """
    # Try manual subtitles first, then auto-generated
    for write_flag, asr in [("--write-subs", False), ("--write-auto-subs", True)]:
        cmd = [
            "yt-dlp",
            write_flag,
            "--sub-langs", "pt,pt-BR,en",
            "--sub-format", "vtt",
            "--skip-download",
            "--no-warnings",
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            continue

        # Find downloaded subtitle file
        vtt_files = sorted(out_dir.glob("*.vtt"))
        if not vtt_files:
            continue

        text = _parse_vtt(vtt_files[0])
        if text.strip():
            return text, asr

    return "", False


def _parse_vtt(path: Path) -> str:
    """Parse a WebVTT file into plain text, removing timestamps and duplicates."""
    lines: list[str] = []
    prev = ""
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        # Skip headers, timestamps, and blank lines
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("Kind:")
            or line.startswith("Language:")
            or "-->" in line
            or line.isdigit()
        ):
            continue
        # Remove VTT inline tags like <00:00:01.000>
        import re
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and clean != prev:
            lines.append(clean)
            prev = clean
    return " ".join(lines)


def _format_date(raw: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw
