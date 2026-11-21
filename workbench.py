"""Disk-backed jobs and the analysis worker. Requires a local POSIX filesystem."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from fractions import Fraction
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4

MAX_BYTES = 100 * 1024 * 1024
MAX_DURATION = 120
MAX_FRAMES = 240
MAX_FPS = 4
MODEL = os.getenv("SAFETY_MODEL", "llama3.2-vision:11b")
FORMATS = {
    ".mp4": "mov", ".mov": "mov", ".webm": "matroska",
    ".mkv": "matroska", ".png": "png_pipe", ".jpg": "jpeg_pipe",
    ".jpeg": "jpeg_pipe",
}
CATEGORIES = {
    "in_cabin": ("eating", "sleeping", "phone_use", "fatigue", "extra_passengers"),
    "road": ("accident_risk", "adverse_weather", "hazards"),
}
POLICY = {"min_hits": 2, "max_gap_samples": 1}
ACTIVE = {"queued", "running", "cancelling"}


class Cancelled(Exception):
    pass


def root():
    return Path(os.getenv("SAFETY_JOBS", "jobs")).resolve()


@contextmanager
def locked(directory, name=".lock", nonblocking=False):
    with (directory / name).open("a") as handle:
        mode = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(handle, mode)
        yield


def read(directory):
    return json.loads((directory / "job.json").read_text())


def save(directory, job):
    # Readers see either complete snapshot. Each writer holds the job lock.
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(job, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temp, directory / "job.json")
        finally:
            temp.unlink(missing_ok=True)


def change(directory, operation):
    with locked(directory):
        job = read(directory)
        operation(job)
        save(directory, job)
        return job


def cancelled(directory):
    if read(directory)["status"] == "cancelling":
        raise Cancelled()


def media_command(directory, command, timeout):
    # Poll with communicate so subprocess pipes cannot block a full decoder.
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        deadline = time.monotonic() + timeout
        try:
            while True:
                cancelled(directory)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("Media processing timed out")
                try:
                    output, _ = process.communicate(timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if process.returncode:
                    raise ValueError("The file could not be decoded as the selected media type")
                return output
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


def extract(directory, job):
    suffix = job["suffix"]
    image = suffix in {".png", ".jpg", ".jpeg"}
    source = directory / "source"
    media_input = ["-protocol_whitelist", "file", "-f", FORMATS[suffix]]
    probe = media_command(directory, [
        "ffprobe", "-v", "error", *media_input, "-show_entries",
        "format=duration:stream=width,height,avg_frame_rate,r_frame_rate",
        "-select_streams", "v:0", "-of", "json", str(source),
    ], 15)
    try:
        metadata = json.loads(probe)
        stream = metadata["streams"][0]
        width, height = stream["width"], stream["height"]
        duration = 0 if image else float(metadata["format"]["duration"])
        rates = [Fraction(stream[key]) for key in ("avg_frame_rate", "r_frame_rate")] if not image else []
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Media is missing usable video metadata") from exc
    if not (0 < width <= 4096 and 0 < height <= 4096 and width * height <= 9_000_000):
        raise ValueError("Unsupported frame dimensions")
    if not image and (not 0 < duration <= MAX_DURATION or
                      any(not 0 < rate <= 120 for rate in rates)):
        raise ValueError("Source must be at most 120 seconds and 120 FPS")
    fps = job["fps"]
    if math.ceil(duration * fps) > MAX_FRAMES:
        raise ValueError("Sampling would exceed 240 frames. Choose a lower FPS.")

    target = directory / "frames"
    for stale in directory.glob("frames.pending-*"):
        shutil.rmtree(stale, ignore_errors=True)
    staging = Path(tempfile.mkdtemp(prefix="frames.pending-", dir=directory))
    try:
        filters = "scale=w='min(960,iw)':h=-2"
        if not image:
            filters = f"fps={fps}:start_time=0:round=up:eof_action=pass," + filters
        media_command(directory, [
            "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "2",
            *media_input, "-i", str(source), "-map", "0:v:0", "-an", "-sn",
            "-filter_threads", "1", "-vf", filters, "-t", str(MAX_DURATION),
            "-frames:v", str(1 if image else MAX_FRAMES + 1), "-threads", "1",
            str(staging / "%04d.jpg"),
        ], 90)
        files = sorted(staging.glob("*.jpg"))
        if not 0 < len(files) <= MAX_FRAMES:
            raise ValueError("Decoded frame count outside allowed bounds")
        cancelled(directory)
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        frames = [
            {"file": path.name, "seconds": index / fps, "state": "pending"}
            for index, path in enumerate(files)
            if image or index / fps < duration
        ]
        if not frames:
            raise ValueError("No samples within the reported duration")
        return duration, frames
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def analyze_image(path, category, model):
    import ollama

    keys = CATEGORIES[category]
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "flags": {
                "type": "object", "required": list(keys),
                "properties": {key: {"type": "boolean"} for key in keys},
                "additionalProperties": False,
            },
        },
        "required": ["summary", "flags"], "additionalProperties": False,
    }
    response = ollama.Client(timeout=30).chat(
        model=model, format=schema, options={"temperature": 0, "num_predict": 250},
        messages=[{
            "role": "user", "images": [str(path)],
            "content": ("Describe observable evidence briefly. Flag only visible cues for "
                        + ", ".join(keys) + ". Treat fatigue and risk as cues requiring review. "
                        "Set adverse_weather only for adverse conditions visible in the frame."),
        }],
    )
    result = json.loads(response.message.content)
    flags = result["flags"]
    if (set(flags) != set(keys) or any(type(value) is not bool for value in flags.values())
            or not isinstance(result["summary"], str)):
        raise ValueError("Invalid model response")
    return {"summary": result["summary"][:1000], "flags": flags}


def group_events(frames, duration, fps, categories, *, min_hits=2, max_gap_samples=1):
    events = []
    if not duration:
        return events
    for category in categories:
        runs = []
        current = []
        for index, frame in enumerate(frames):
            positive = frame.get("flags", {}).get(category) is True
            if current and (index - current[-1] > max_gap_samples or
                            (frame["state"] == "done" and not positive)):
                runs.append(current)
                current = []
            if positive:
                current.append(index)
        if current:
            runs.append(current)
        for run in runs:
            if len(run) < min_hits:
                continue
            evidence = [{"file": frames[i]["file"], "seconds": frames[i]["seconds"]} for i in run]
            identity = category + ":" + ",".join(frame["file"] for frame in evidence)
            events.append({
                "id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                "category": category,
                "start": evidence[0]["seconds"],
                "end": min(duration, evidence[-1]["seconds"] + 1 / fps),
                "evidence": evidence,
            })
    return sorted(events, key=lambda event: (event["start"], event["category"]))


def process(directory, analyzer=analyze_image):
    job = read(directory)
    start = time.monotonic()
    status = "complete"
    error = None
    try:
        cancelled(directory)
        if not job["frames"]:
            duration, frames = extract(directory, job)
            job = change(directory, lambda state: state.update(
                duration=duration, frames=frames, stage="analysis"))
        for index, frame in enumerate(job["frames"]):
            cancelled(directory)
            if frame["state"] == "done":
                continue
            try:
                result = analyzer(directory / "frames" / frame["file"], job["category"], job["model"])
            except Exception:
                logging.exception("Frame analysis failed for %s/%s", directory.name, frame["file"])
                result = {"state": "failed"}
            else:
                result["state"] = "done"
            change(directory, lambda state: state["frames"][index].update(result))
        if any(frame["state"] == "failed" for frame in read(directory)["frames"]):
            status = "partial"
    except Cancelled:
        status = "cancelled"
    except ValueError as exc:
        status, error = "failed", str(exc)
    except Exception:
        logging.exception("Job %s failed", directory.name)
        status, error = "failed", "Processing failed. Check the worker logs."
    finally:
        def finish(state):
            state.update(
                status="cancelled" if state["status"] == "cancelling" else status,
                error=error,
                processing_seconds=state["processing_seconds"] + time.monotonic() - start,
            )
        change(directory, finish)


def worker(analyzer=analyze_image):
    base = root()
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    # One worker supervisor per job root. Two local slots share no media files.
    with locked(base, ".worker.lock", nonblocking=True):
        for directory in base.glob("*/job.json"):
            def recover(job):
                if job["status"] == "running":
                    job["status"] = "queued"
                elif job["status"] == "cancelling":
                    job["status"] = "cancelled"
            change(directory.parent, recover)
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = {}
            while True:
                for directory, future in list(running.items()):
                    if future.done():
                        future.result()
                        del running[directory]
                for record in sorted(base.glob("*/job.json"), key=lambda path: path.stat().st_mtime):
                    directory = record.parent
                    if len(running) >= 2:
                        break
                    if directory in running or read(directory)["status"] != "queued":
                        continue
                    with locked(directory):
                        job = read(directory)
                        if job["status"] != "queued":
                            continue
                        job["status"] = "running"
                        save(directory, job)
                    running[directory] = pool.submit(process, directory, analyzer)
                time.sleep(0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Safety App background worker")
    parser.parse_args()
    worker()
