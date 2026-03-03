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
    get_search_errors,
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
    generate_all_content,
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

    Uses Gemini AI when GEMINI_API_KEY is set. Makes a SINGLE API call
    to generate summary, topics, chapters, and participant bios all at once.
    """
    use_ai = gemini_available()
    diagnostics = {
        "ai_configured": use_ai,
        "ai_used": False,
        # Keep old keys for backwards compat
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

    # --- Heuristic fallbacks (always computed) ---
    fallback_summary = generate_summary(
        title=video.title,
        description=video.description,
        transcript=video.transcript,
        participant_names=participant_names,
    )
    fallback_chapters = generate_chapters(
        existing_chapters=video.chapters,
        transcript=video.transcript,
        duration=video.duration,
    )
    fallback_topics = generate_topics(
        title=video.title,
        description=video.description,
        transcript=video.transcript,
        keywords=keywords,
    )

    # Generate mini-bios: description parsing → web search (DDG → Google)
    for person in validated:
        if not person.mini_bio:
            role, bio = generate_mini_bio(
                person.canonical, video.channel, video.description
            )
            person.role = role
            person.mini_bio = bio

    # --- AI generation: SINGLE call for everything ---
    summary = fallback_summary
    chapters = fallback_chapters
    topics = fallback_topics

    if use_ai:
        print("→ Gerando conteúdo via Gemini (1 chamada)...", file=sys.stderr)
        ai_result = generate_all_content(
            title=video.title,
            description=video.description,
            transcript=video.transcript,
            participant_names=participant_names,
            existing_chapters=fallback_chapters,
            duration=video.duration,
            channel_name=video.channel,
        )

        if ai_result:
            diagnostics["ai_used"] = True
            diagnostics["gemini_used"] = True

            if ai_result.get("summary"):
                summary = ai_result["summary"]
                diagnostics["ai_summary"] = True
                logger.info("Pipeline: resumo AI OK")

            if ai_result.get("topics"):
                topics = ai_result["topics"]
                diagnostics["ai_topics"] = True
                logger.info("Pipeline: tópicos AI OK (%d)", len(topics))

            if ai_result.get("chapters"):
                chapters = ai_result["chapters"]
                diagnostics["ai_chapters"] = True
                logger.info("Pipeline: capítulos AI OK (%d)", len(chapters))

            if ai_result.get("participants"):
                # Match AI bios to validated participants
                ai_bios = {p["name"].lower(): p for p in ai_result["participants"]}
                for person in validated:
                    match = ai_bios.get(person.canonical.lower())
                    if match:
                        if match.get("role"):
                            person.role = match["role"]
                        if match.get("bio") and match["bio"] != "Participante do programa":
                            person.mini_bio = match["bio"]
                        diagnostics["ai_participants"] = True
        else:
            logger.warning("Pipeline: Gemini não retornou conteúdo, usando fallbacks")
            diagnostics["errors"].append("Gemini returned empty")

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

    # Collect AI errors for diagnostics
    ai_errors = get_recent_errors()
    if ai_errors:
        diagnostics["ai_errors"] = ai_errors

    # Collect search errors for diagnostics
    search_errors = get_search_errors()
    if search_errors:
        diagnostics["search_errors"] = search_errors

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
