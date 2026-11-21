from concurrent.futures import ThreadPoolExecutor
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from app import app
from evaluate import evaluate
from workbench import worker


def scripted_analysis(path, category, model):
    directory = path.parent.parent
    base = directory.parent
    job = json.loads((directory / "job.json").read_text())
    if job["filename"] == "cancel.mp4":
        while not (base / "release").exists():
            time.sleep(0.03)
    if path.name == "0004.jpg" and not (base / "recovered").exists():
        raise RuntimeError("Scripted inference failure")
    return {"summary": "Scripted observation", "flags": {"phone_use": path.name in {"0001.jpg", "0002.jpg"}}}


class WorkbenchTest(unittest.TestCase):
    def test_concurrent_upload_cancel_retry_review_and_export(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            video = base / "sample.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=s=160x120:r=10:d=4", "-c:v", "mpeg4", str(video)], check=True)
            with patch.dict(os.environ, SAFETY_JOBS=str(base / "jobs")):
                self.run_demo(base / "jobs", video.read_bytes())

    def run_demo(self, base, video):
        client = app.test_client()

        def upload(name, data=video):
            return app.test_client().post("/", data={
                "category": "in_cabin", "fps": "1",
                "file": (io.BytesIO(data), name),
            })

        def wait(path, condition):
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                progress = client.get(path + "/status").get_json()
                if condition(progress):
                    return progress
                time.sleep(0.05)
            self.fail(f"Timed out waiting for {path}: {progress}")

        self.assertEqual(upload("../escape.mp4").status_code, 400)
        with ThreadPoolExecutor(2) as pool:
            cancelled, kept = list(pool.map(upload, ("cancel.mp4", "review.mp4")))
        self.assertEqual((cancelled.status_code, kept.status_code), (303, 303))
        cancel_path, keep_path = cancelled.location, kept.location
        self.assertNotEqual(cancel_path, keep_path)
        process = multiprocessing.Process(target=worker, args=(scripted_analysis,))
        process.start()
        try:
            wait(cancel_path, lambda progress: progress["stage"] == "analysis")
            self.assertEqual(client.post(cancel_path + "/cancel").status_code, 303)
            (base / "release").touch()
            wait(cancel_path, lambda progress: progress["status"] == "cancelled")
            wait(keep_path, lambda progress: progress["status"] == "partial")
            first = client.get(keep_path + "/report.json").get_json()
            self.assertEqual([f["state"] for f in first["frames"]], ["done"] * 3 + ["failed"])
            event, = first["events"]
            self.assertEqual((event["start"], event["end"]), (0, 2))
            self.assertIn(b"Event timeline", client.get(keep_path).data)
            image = client.get(keep_path + "/frames/" + event["evidence"][0]["file"])
            self.assertEqual(image.mimetype, "image/jpeg")
            image.close()
            self.assertEqual(client.post(keep_path + "/review/" + event["id"],
                                         data={"decision": "dismissed"}).status_code, 303)
            (base / "recovered").touch()
            self.assertEqual(client.post(keep_path + "/retry").status_code, 303)
            wait(keep_path, lambda progress: progress["status"] == "complete")
            response = client.get(keep_path + "/report.json")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            result = response.get_json()
            self.assertEqual(result["events"][0]["review"], "dismissed")
            self.assertEqual(result["frames"][:3], first["frames"][:3])
            metrics = evaluate([({"condition": "normal", "labels": []}, result)], 0.5)["all"]
            self.assertEqual(metrics["false_alerts_per_video_hour"], 900)
            self.assertEqual(metrics["precision"], 0)
            self.assertIsNone(metrics["recall"])
            invalid = upload("fake.mp4", b"not video")
            wait(invalid.location, lambda progress: progress["status"] == "failed")
        finally:
            process.terminate()
            process.join(5)


if __name__ == "__main__":
    unittest.main()
