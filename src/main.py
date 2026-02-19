"""Descrição Arbitragem — YouTube description automation CLI."""

import argparse
import json
import sys
from pathlib import Path

from .extractor import extract_video_data
from .ocr import extract_text_from_thumbnail, extract_name_candidates_from_ocr
from .names import (
    collect_name_candidates,
    validate_and_canonise,
    generate_mini_bio,
)
from .content import generate_summary, generate_chapters, generate_keywords
from .template import render_description


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="descricao-arbitragem",
        description="Gera descrições estruturadas para vídeos do YouTube.",
    )
    parser.add_argument(
        "youtube_url",
        help="URL do vídeo no YouTube",
    )
    parser.add_argument(
        "--output", "-o",
        help="Arquivo de saída (padrão: stdout)",
        default=None,
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Saída em formato JSON em vez de texto",
    )
    parser.add_argument(
        "--work-dir", "-w",
        help="Diretório de trabalho para assets temporários",
        default=None,
    )
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(args.youtube_url, args.work_dir)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        output_text = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        output_text = result["description"]

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Descrição salva em: {args.output}", file=sys.stderr)
    else:
        print(output_text)


def run_pipeline(youtube_url: str, work_dir: str | None = None) -> dict:
    """Execute the full description generation pipeline.

    Returns a dict with all intermediate data and the final description.
    """
    # === Step 1: Extract video data ===
    print("→ Extraindo dados do vídeo...", file=sys.stderr)
    video = extract_video_data(youtube_url, work_dir)

    # === Step 2: OCR on thumbnail ===
    print("→ Processando OCR na thumbnail...", file=sys.stderr)
    ocr_result = extract_text_from_thumbnail(video.thumbnail_path)
    ocr_full = ocr_result["ocr_text_full"]
    ocr_short = ocr_result["ocr_text_short"]

    # === Step 3: Name extraction and validation ===
    print("→ Extraindo e validando nomes...", file=sys.stderr)
    ocr_names = extract_name_candidates_from_ocr(ocr_full)

    candidates = collect_name_candidates(
        title=video.title,
        description=video.description,
        ocr_names=ocr_names,
        transcript=video.transcript,
    )

    validated = validate_and_canonise(
        candidates=candidates,
        channel_name=video.channel,
        video_title=video.title,
        ocr_full=ocr_full,
    )

    # Generate mini-bios
    for person in validated:
        if not person.mini_bio:
            person.mini_bio = generate_mini_bio(person.canonical, video.channel)

    participant_names = [p.canonical for p in validated]

    # === Step 4: Content generation ===
    print("→ Gerando conteúdo...", file=sys.stderr)

    # Main topic (heuristic: first significant phrase from title)
    main_topic = _extract_main_topic(video.title)

    summary = generate_summary(
        title=video.title,
        description=video.description,
        transcript=video.transcript,
        participant_names=participant_names,
    )

    chapters = generate_chapters(
        existing_chapters=video.chapters,
        transcript=video.transcript,
        duration=video.duration,
    )

    keywords = generate_keywords(
        title=video.title,
        description=video.description,
        transcript=video.transcript,
        ocr_text=ocr_full,
        channel_name=video.channel,
    )

    # === Step 5: Render final description ===
    print("→ Renderizando descrição final...", file=sys.stderr)
    description = render_description(
        title=video.title,
        main_topic=main_topic,
        ocr_short=ocr_short,
        summary=summary,
        participants=validated,
        chapters=chapters,
        keywords=keywords,
        channel_name=video.channel,
        asr_generated=video.asr_generated,
    )

    return {
        "video_id": video.video_id,
        "title": video.title,
        "channel": video.channel,
        "upload_date": video.upload_date,
        "duration": video.duration,
        "ocr_text_full": ocr_full,
        "ocr_text_short": ocr_short,
        "participants": [
            {
                "name": p.canonical,
                "source": p.source,
                "trust": p.trust_level,
                "bio": p.mini_bio,
            }
            for p in validated
        ],
        "chapters": chapters,
        "keywords": keywords,
        "summary": summary,
        "main_topic": main_topic,
        "asr_generated": video.asr_generated,
        "description": description,
    }


def _extract_main_topic(title: str) -> str:
    """Extract the main topic from the video title.

    Heuristic: take text after common separators like |, -, :, or use the
    full title if no separator is found.
    """
    for sep in ["|", " - ", ":"]:
        if sep in title:
            parts = title.split(sep)
            # Pick the longer part as the topic
            topic = max(parts, key=lambda p: len(p.strip()))
            return topic.strip()
    return title.strip()


if __name__ == "__main__":
    main()
