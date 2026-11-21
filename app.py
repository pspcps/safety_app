import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from workbench import ACTIVE, CATEGORIES, FORMATS, MAX_BYTES, MAX_FPS, MODEL, POLICY
from workbench import change, group_events, read, root, save

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=MAX_BYTES + 64 * 1024, MAX_FORM_PARTS=10)
DECISIONS = ("accepted", "dismissed", "uncertain")
NOTICE = "Model suggestions can miss events or raise false alerts. Review the supporting footage."


def lookup(job_id):
    directory = root() / job_id.hex
    if not (directory / "job.json").is_file():
        abort(404)
    return directory


def report(job):
    events = group_events(job["frames"], job["duration"], job["fps"],
                          CATEGORIES[job["category"]], **job["policy"])
    for event in events:
        event["review"] = job["reviews"].get(event["id"], "unreviewed")
    return {
        key: job[key] for key in ("id", "filename", "category", "status", "stage",
                                  "error", "fps", "duration", "frames", "model",
                                  "policy", "processing_seconds")
    } | {"events": events, "notice": NOTICE}


@app.after_request
def private_response(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/")
def index():
    return render_template("index.html", formats=", ".join(FORMATS), max_fps=MAX_FPS)


@app.post("/")
def upload():
    file = request.files.get("file")
    name = file.filename if file else ""
    suffix = Path(name).suffix.lower()
    if (not name or len(name) > 180 or "/" in name or "\\" in name
            or any(ord(char) < 32 for char in name)
            or not secure_filename(name) or suffix not in FORMATS):
        abort(400, "Choose a supported file with a valid filename.")
    category = request.form.get("category")
    try:
        fps = int(request.form.get("fps", "1"))
    except ValueError:
        abort(400, "Sampling FPS must be an integer.")
    if category not in CATEGORIES or not 1 <= fps <= MAX_FPS:
        abort(400, "Choose a valid camera view and sampling FPS.")

    job_id = uuid4()
    directory = root() / job_id.hex
    directory.mkdir(mode=0o700, parents=True)
    try:
        total = 0
        with (directory / "source").open("wb") as output:
            while chunk := file.stream.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    abort(413)
                output.write(chunk)
        if not total:
            abort(400, "The file is empty.")
        save(directory, {
            "id": job_id.hex, "filename": name, "suffix": suffix,
            "category": category, "fps": fps, "status": "queued", "stage": "extraction",
            "error": None, "duration": None, "frames": [], "reviews": {},
            "model": MODEL, "policy": POLICY, "processing_seconds": 0,
        })
    except Exception:
        shutil.rmtree(directory)
        raise
    return redirect(url_for("job_page", job_id=job_id), code=303)


@app.get("/jobs/<uuid:job_id>")
def job_page(job_id):
    job = report(read(lookup(job_id)))
    return render_template("job.html", job=job, job_id=job_id, decisions=DECISIONS,
                           active=job["status"] in ACTIVE)


@app.get("/jobs/<uuid:job_id>/status")
def status(job_id):
    job = read(lookup(job_id))
    return jsonify(status=job["status"], stage=job["stage"], total=len(job["frames"]),
                   processed=sum(frame["state"] != "pending" for frame in job["frames"]))


@app.post("/jobs/<uuid:job_id>/cancel")
def cancel(job_id):
    def apply(job):
        if job["status"] == "queued":
            job["status"] = "cancelled"
        elif job["status"] == "running":
            job["status"] = "cancelling"
    change(lookup(job_id), apply)
    return redirect(url_for("job_page", job_id=job_id), code=303)


@app.post("/jobs/<uuid:job_id>/retry")
def retry(job_id):
    def apply(job):
        if job["status"] not in {"failed", "partial", "cancelled"}:
            abort(409)
        for frame in job["frames"]:
            if frame["state"] == "failed":
                frame["state"] = "pending"
        job.update(status="queued", error=None)
    change(lookup(job_id), apply)
    return redirect(url_for("job_page", job_id=job_id), code=303)


@app.post("/jobs/<uuid:job_id>/review/<event_id>")
def review(job_id, event_id):
    decision = request.form.get("decision")
    if decision not in DECISIONS:
        abort(400, "Choose a review decision.")

    def apply(job):
        if job["status"] in ACTIVE:
            abort(409)
        if event_id not in {event["id"] for event in report(job)["events"]}:
            abort(404)
        job["reviews"][event_id] = decision

    change(lookup(job_id), apply)
    return redirect(url_for("job_page", job_id=job_id, _anchor=event_id), code=303)


@app.get("/jobs/<uuid:job_id>/frames/<filename>")
def evidence(job_id, filename):
    directory = lookup(job_id)
    if not re.fullmatch(r"[0-9]{4}\.jpg", filename):
        abort(404)
    return send_from_directory(directory / "frames", filename, mimetype="image/jpeg")


@app.get("/jobs/<uuid:job_id>/report.json")
def export(job_id):
    job = report(read(lookup(job_id)))
    return app.response_class(
        json.dumps(job, indent=2), mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="safety-{job_id.hex}.json"'},
    )


if __name__ == "__main__":
    app.run()
