"""Descrição Arbitragem — Web interface using Flask."""

import json
import threading
import uuid

from flask import Flask, render_template, request, jsonify

from .main import run_pipeline

app = Flask(__name__)

# In-memory job store: job_id -> {status, result, error}
_jobs: dict[str, dict] = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start description generation as a background job."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL é obrigatória"}), 400

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
    try:
        result = run_pipeline(url)
        _jobs[job_id] = {"status": "done", "result": result, "error": None}
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "result": None, "error": str(exc)}


def run_web(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Entry point for the web server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_web(debug=True)
