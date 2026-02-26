"""Descrição Arbitragem — Web interface using Flask."""

import json
import logging
import os
import threading
import uuid

from flask import Flask, render_template, request, jsonify

from .main import run_pipeline
from .ai import gemini_available

# Configure logging to stderr so Railway shows it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# In-memory job store: job_id -> {status, result, error}
_jobs: dict[str, dict] = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    """Health check showing configuration status."""
    yt_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    result = {
        "status": "ok",
        "youtube_api": bool(yt_key),
        "youtube_api_prefix": yt_key[:8] + "..." if yt_key else None,
        "gemini_api": bool(gemini_key),
        "gemini_api_prefix": gemini_key[:8] + "..." if gemini_key else None,
    }

    return jsonify(result)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start description generation as a background job."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL é obrigatória"}), 400

    if not os.environ.get("YOUTUBE_API_KEY", "").strip():
        return jsonify({
            "error": "YOUTUBE_API_KEY não configurada. "
            "Crie uma API key em console.cloud.google.com e "
            "adicione como variável de ambiente no Railway."
        }), 400

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "result": None, "error": None}

    thread = threading.Thread(target=_run_job, args=(job_id, url), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    """Poll job status."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job não encontrado"}), 404
    return jsonify(job)


def _run_job(job_id: str, url: str) -> None:
    logger.info("Job %s iniciado para URL: %s", job_id, url)
    try:
        result = run_pipeline(url)
        _jobs[job_id] = {"status": "done", "result": result, "error": None}
        diag = result.get("diagnostics", {})
        logger.info("Job %s concluído — Gemini usado: %s, erros: %s",
                     job_id, diag.get("gemini_used"), diag.get("errors"))
    except Exception as exc:
        logger.error("Job %s falhou: %s", job_id, exc, exc_info=True)
        _jobs[job_id] = {"status": "error", "result": None, "error": str(exc)}


def run_web(host: str = "0.0.0.0", port: int = 0, debug: bool = False):
    """Entry point for the web server."""
    import os
    if not port:
        port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_web(debug=True)
