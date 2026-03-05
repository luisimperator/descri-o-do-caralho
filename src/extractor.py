"""YouTube video data extraction.

Uses YouTube Data API v3 (when YOUTUBE_API_KEY is set) with
youtube-transcript-api for transcripts. Falls back to yt-dlp
for local/dev usage.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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
    transcript_segments: list = field(default_factory=list)  # [{"start": seconds, "text": "..."}]
    asr_generated: bool = False


def extract_video_data(youtube_url: str, output_dir: str | None = None) -> VideoData:
    """Extract metadata, thumbnail, and transcript from a YouTube video.

    Uses YouTube Data API v3 if YOUTUBE_API_KEY env var is set,
    otherwise falls back to yt-dlp.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="yt_desc_")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()

    if api_key:
        return _extract_via_api(youtube_url, api_key, out_path)
    else:
        return _extract_via_ytdlp(youtube_url, out_path)


# ---------------------------------------------------------------------------
# YouTube Data API v3 extraction
# ---------------------------------------------------------------------------

def _extract_via_api(youtube_url: str, api_key: str, out_path: Path) -> VideoData:
    """Extract video data using YouTube Data API v3."""
    video_id = _parse_video_id(youtube_url)
    if not video_id:
        raise RuntimeError(f"Não foi possível extrair o video ID de: {youtube_url}")

    data = VideoData(video_id=video_id)

    # Fetch metadata from YouTube Data API
    api_url = (
        f"https://www.googleapis.com/youtube/v3/videos"
        f"?id={video_id}&part=snippet,contentDetails&key={api_key}"
    )
    req = urllib.request.Request(api_url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"YouTube API falhou: {exc}")

    items = result.get("items", [])
    if not items:
        raise RuntimeError(f"Vídeo não encontrado: {video_id}")

    item = items[0]
    snippet = item.get("snippet", {})
    content_details = item.get("contentDetails", {})

    data.title = snippet.get("title", "")
    data.description = snippet.get("description", "")
    data.channel = snippet.get("channelTitle", "")
    data.channel_url = f"https://www.youtube.com/channel/{snippet.get('channelId', '')}"
    data.upload_date = _format_api_date(snippet.get("publishedAt", ""))
    data.tags = snippet.get("tags", []) or []

    # Thumbnail (pick best quality)
    thumbs = snippet.get("thumbnails", {})
    for quality in ["maxres", "high", "medium", "default"]:
        if quality in thumbs:
            data.thumbnail_url = thumbs[quality]["url"]
            break

    # Duration (ISO 8601: PT1H2M3S)
    data.duration = _parse_iso_duration(content_details.get("duration", ""))

    # Chapters from description timestamps
    data.chapters = _parse_chapters_from_description(data.description)

    # Download thumbnail
    if data.thumbnail_url:
        data.thumbnail_path = _download_thumbnail(
            data.thumbnail_url, out_path / f"{data.video_id}_thumb.jpg"
        )

    # Transcript via youtube-transcript-api
    data.transcript, data.asr_generated, data.transcript_segments = _fetch_transcript_api(video_id)

    return data


def _parse_video_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    # youtu.be/VIDEO_ID
    match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    # youtube.com/watch?v=VIDEO_ID
    match = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    # youtube.com/embed/VIDEO_ID
    match = re.search(r"embed/([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    # Might already be a video ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url
    return ""


def _parse_iso_duration(iso: str) -> int:
    """Parse ISO 8601 duration (PT1H2M3S) to seconds."""
    if not iso:
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not match:
        return 0
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = int(match.group(3) or 0)
    return h * 3600 + m * 60 + s


def _format_api_date(iso_date: str) -> str:
    """Convert ISO 8601 date (2024-01-15T10:00:00Z) to YYYY-MM-DD."""
    if not iso_date:
        return ""
    return iso_date[:10]


def _parse_chapters_from_description(description: str) -> list[dict]:
    """Extract chapters from YouTube description timestamps.

    Looks for patterns like:
        00:00 Introdução
        02:15 - Primeiro tema
        1:30:00 Conclusão
    """
    if not description:
        return []

    pattern = r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]?\s*(.+)"
    chapters: list[dict] = []
    for line in description.splitlines():
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            timestamp_str = match.group(1)
            title = match.group(2).strip()
            seconds = _timestamp_to_seconds(timestamp_str)
            chapters.append({"start": seconds, "title": title})

    return chapters


def _timestamp_to_seconds(ts: str) -> int:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def _fetch_transcript_api(video_id: str) -> tuple[str, bool, list[dict]]:
    """Fetch transcript using youtube-transcript-api.

    Tries multiple language combinations and falls back to listing
    available transcripts.

    Returns (transcript_text, asr_generated, segments).
    segments is a list of {"start": seconds, "text": "..."}.
    """
    def _extract_segments(transcript_list) -> tuple[str, list[dict]]:
        segments = []
        texts = []
        for snippet in transcript_list.snippets:
            texts.append(snippet.text)
            segments.append({"start": int(snippet.start), "text": snippet.text})
        return " ".join(texts), segments

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()

        # Try 1: Manual subtitles in preferred languages
        for langs in [["pt", "pt-BR", "en"], ["pt", "en"], ["es", "en"]]:
            try:
                transcript_list = ytt_api.fetch(video_id, languages=langs)
                text, segments = _extract_segments(transcript_list)
                if text.strip():
                    logger.info("Transcrição obtida: %d chars, %d segmentos (idiomas: %s)",
                                len(text), len(segments), langs)
                    return text, False, segments
            except Exception:
                continue

        # Try 2: List all available transcripts and pick the best one
        try:
            transcript_list = ytt_api.list(video_id)
            for t in transcript_list:
                try:
                    fetched = ytt_api.fetch(video_id, languages=[t.language_code])
                    text, segments = _extract_segments(fetched)
                    if text.strip():
                        logger.info("Transcrição obtida: %d chars, %d segmentos (idioma: %s, auto: %s)",
                                    len(text), len(segments), t.language_code, t.is_generated)
                        return text, t.is_generated, segments
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Falha ao listar transcrições para %s: %s", video_id, exc)

        logger.warning("Nenhuma transcrição disponível para %s", video_id)
        return "", False, []
    except ImportError:
        logger.error("youtube-transcript-api não instalado")
        return "", False, []
    except Exception as exc:
        logger.warning("Falha ao obter transcrição para %s: %s", video_id, exc)
        return "", False, []


# ---------------------------------------------------------------------------
# yt-dlp fallback (for local/dev usage)
# ---------------------------------------------------------------------------

def _extract_via_ytdlp(youtube_url: str, out_path: Path) -> VideoData:
    """Extract video data using yt-dlp (fallback when no API key)."""
    data = VideoData()

    meta = _fetch_metadata_ytdlp(youtube_url)
    data.video_id = meta.get("id", "")
    data.title = meta.get("title", "")
    data.description = meta.get("description", "")
    data.upload_date = _format_date_ytdlp(meta.get("upload_date", ""))
    data.channel = meta.get("channel", meta.get("uploader", ""))
    data.channel_url = meta.get("channel_url", "")
    data.thumbnail_url = meta.get("thumbnail", "")
    data.duration = int(meta.get("duration", 0))
    data.tags = meta.get("tags", []) or []

    raw_chapters = meta.get("chapters") or []
    data.chapters = [
        {"start": int(ch.get("start_time", 0)), "title": ch.get("title", "")}
        for ch in raw_chapters
    ]

    if data.thumbnail_url:
        data.thumbnail_path = _download_thumbnail(
            data.thumbnail_url, out_path / f"{data.video_id}_thumb.jpg"
        )

    data.transcript, data.asr_generated = _fetch_transcript_ytdlp(youtube_url, out_path)

    return data


def _fetch_metadata_ytdlp(url: str) -> dict:
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


def _fetch_transcript_ytdlp(url: str, out_dir: Path) -> tuple[str, bool]:
    """Try to fetch subtitles via yt-dlp; fall back to auto-generated."""
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

        vtt_files = sorted(out_dir.glob("*.vtt"))
        if not vtt_files:
            continue

        text = _parse_vtt(vtt_files[0])
        if text.strip():
            return text, asr

    return "", False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _download_thumbnail(url: str, dest: Path) -> str:
    """Download thumbnail image to *dest* and return the path as string."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            dest.write_bytes(resp.read())
        return str(dest)
    except Exception:
        return ""


def _parse_vtt(path: Path) -> str:
    """Parse a WebVTT file into plain text, removing timestamps and duplicates."""
    lines: list[str] = []
    prev = ""
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("Kind:")
            or line.startswith("Language:")
            or "-->" in line
            or line.isdigit()
        ):
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and clean != prev:
            lines.append(clean)
            prev = clean
    return " ".join(lines)


def _format_date_ytdlp(raw: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw
