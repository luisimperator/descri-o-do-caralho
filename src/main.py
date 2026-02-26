"""Descrição Arbitragem — YouTube description automation CLI."""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from .extractor import extract_video_data
from .ocr import extract_text_from_thumbnail, extract_name_candidates_from_ocr
from .names import (
    collect_name_candidates,
    validate_and_canonise,
    generate_mini_bio,
)
from .content import (
    generate_summary,
    generate_chapters,
    generate_keywords,
    generate_topics,
    extract_social_links,
)
from .template import render_description
from .ai import (
    gemini_available,
    get_recent_errors,
    research_participant,
    generate_chapter_titles,
    generate_summary_ai,
    generate_topics_ai,
)


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

    Uses Gemini AI when GEMINI_API_KEY is set for better participant
    research, chapter titles, summary, and topics.
    """
    use_ai = gemini_available()
    diagnostics = {
        "gemini_configured": use_ai,
        "gemini_used": False,
        "transcript_available": False,
        "chapters_from_video": False,
        "ai_chapters": False,
        "ai_summary": False,
        "ai_topics": False,
        "ai_participants": False,
        "errors": [],
    }
    logger.info("Pipeline: Gemini disponível = %s", use_ai)

    # === Step 1: Extract video data ===
    print("→ Extraindo dados do vídeo...", file=sys.stderr)
    video = extract_video_data(youtube_url, work_dir)

    # === Step 2: OCR on thumbnail ===
    print("→ Processando OCR na thumbnail...", file=sys.stderr)
    ocr_result = extract_text_from_thumbnail(video.thumbnail_path)
    ocr_full = ocr_result["ocr_text_full"]

    # === Step 3: Name extraction and validation ===
    print("→ Extraindo e validando nomes...", file=sys.stderr)
    ocr_names = extract_name_candidates_from_ocr(ocr_full)

    candidates = collect_name_candidates(
        title=video.title,
        description=video.description,
        ocr_names=ocr_names,
        transcript=video.transcript,
        channel_name=video.channel,
    )

    validated = validate_and_canonise(
        candidates=candidates,
        channel_name=video.channel,
        video_title=video.title,
        ocr_full=ocr_full,
        description=video.description,
    )

    # Generate mini-bios and roles
    if use_ai:
        print("→ Pesquisando participantes via Gemini...", file=sys.stderr)
        for person in validated:
            result_ai = research_participant(person.canonical, video.channel)
            person.role = result_ai.get("role", "")
            person.mini_bio = result_ai.get("bio", "Participante do programa")
            if result_ai.get("role"):
                diagnostics["ai_participants"] = True
                diagnostics["gemini_used"] = True
    else:
        for person in validated:
            if not person.mini_bio:
                role, bio = generate_mini_bio(person.canonical, video.channel)
                person.role = role
                person.mini_bio = bio

    diagnostics["transcript_available"] = bool(video.transcript)
    diagnostics["chapters_from_video"] = bool(video.chapters)
    logger.info("Pipeline: transcrição=%d chars, capítulos do vídeo=%d",
                len(video.transcript), len(video.chapters))

    # Only present participants for the description
    present_participants = [p for p in validated if p.present_in_video]
    participant_names = [p.canonical for p in present_participants]

    # === Step 4: Content generation ===
    print("→ Gerando conteúdo...", file=sys.stderr)

    main_topic = _extract_main_topic(video.title)

    keywords = generate_keywords(
        title=video.title,
        description=video.description,
        transcript=video.transcript,
        ocr_text=ocr_full,
        channel_name=video.channel,
    )

    # --- Summary ---
    summary = ""
    if use_ai:
        print("→ Gerando resumo via Gemini...", file=sys.stderr)
        summary = generate_summary_ai(
            title=video.title,
            description=video.description,
            transcript=video.transcript,
            participant_names=participant_names,
        )
        if summary:
            diagnostics["ai_summary"] = True
            diagnostics["gemini_used"] = True
            logger.info("Pipeline: resumo AI OK (%d chars)", len(summary))
        else:
            logger.warning("Pipeline: resumo AI falhou, usando fallback")
            diagnostics["errors"].append("Gemini summary returned empty")
    if not use_ai or not summary:
        summary = generate_summary(
            title=video.title,
            description=video.description,
            transcript=video.transcript,
            participant_names=participant_names,
        )

    # --- Chapters ---
    chapters = generate_chapters(
        existing_chapters=video.chapters,
        transcript=video.transcript,
        duration=video.duration,
    )
    if use_ai:
        if video.transcript:
            print("→ Gerando capítulos via Gemini...", file=sys.stderr)
            ai_chapters = generate_chapter_titles(
                transcript=video.transcript,
                title=video.title,
                duration=video.duration,
                existing_chapters=chapters,
            )
            if ai_chapters:
                chapters = ai_chapters
                diagnostics["ai_chapters"] = True
                diagnostics["gemini_used"] = True
                logger.info("Pipeline: capítulos AI OK (%d capítulos)", len(chapters))
            else:
                logger.warning("Pipeline: capítulos AI falhou, usando fallback")
                diagnostics["errors"].append("Gemini chapters returned empty")
        else:
            logger.warning("Pipeline: sem transcrição, capítulos AI pulados")
            diagnostics["errors"].append("No transcript available for AI chapters")

    # --- Topics ---
    topics = []
    if use_ai:
        print("→ Gerando tópicos via Gemini...", file=sys.stderr)
        topics = generate_topics_ai(
            title=video.title,
            description=video.description,
            transcript=video.transcript,
        )
        if topics:
            diagnostics["ai_topics"] = True
            diagnostics["gemini_used"] = True
            logger.info("Pipeline: tópicos AI OK (%d tópicos)", len(topics))
        else:
            logger.warning("Pipeline: tópicos AI falhou, usando fallback")
            diagnostics["errors"].append("Gemini topics returned empty")
    if not use_ai or not topics:
        topics = generate_topics(
            title=video.title,
            description=video.description,
            transcript=video.transcript,
            keywords=keywords,
        )

    social_links = extract_social_links(video.description)

    # === Step 5: Render final description ===
    print("→ Renderizando descrição final...", file=sys.stderr)
    description = render_description(
        title=video.title,
        main_topic=main_topic,
        summary=summary,
        participants=validated,
        topics=topics,
        chapters=chapters,
        keywords=keywords,
        channel_name=video.channel,
        social_links=social_links,
        asr_generated=video.asr_generated,
    )

    # Collect actual Gemini API errors for diagnostics
    gemini_errors = get_recent_errors()
    if gemini_errors:
        diagnostics["gemini_errors"] = gemini_errors

    logger.info("Pipeline concluído: %s", json.dumps(diagnostics, ensure_ascii=False))

    return {
        "video_id": video.video_id,
        "title": video.title,
        "channel": video.channel,
        "upload_date": video.upload_date,
        "duration": video.duration,
        "participants": [
            {
                "name": p.canonical,
                "source": p.source,
                "trust": p.trust_level,
                "bio": p.mini_bio,
                "role": p.role,
                "present_in_video": p.present_in_video,
            }
            for p in validated
        ],
        "topics": topics,
        "chapters": chapters,
        "keywords": keywords,
        "summary": summary,
        "main_topic": main_topic,
        "social_links": social_links,
        "asr_generated": video.asr_generated,
        "description": description,
        "diagnostics": diagnostics,
    }


def _extract_main_topic(title: str) -> str:
    """Extract the main topic from the video title.

    Heuristic: take text after common separators like |, -, :.
    Picks the longest part that is NOT a show/podcast name.
    """
    for sep in ["|", " - ", ":"]:
        if sep in title:
            parts = [p.strip() for p in title.split(sep) if p.strip()]
            if len(parts) >= 2:
                # Skip the first part (usually the show name), pick the longest of the rest
                rest = parts[1:]
                return max(rest, key=len).strip()
            elif parts:
                return parts[0].strip()
    return title.strip()


if __name__ == "__main__":
    main()
