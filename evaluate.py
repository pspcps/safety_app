"""Score raw model events against independent time-interval labels."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from workbench import CATEGORIES, group_events


def overlap(left, right):
    intersection = max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    union = max(left["end"], right["end"]) - min(left["start"], right["start"])
    return intersection / union


def match(predictions, labels, threshold):
    """Maximum one-to-one matching by category and temporal intersection-over-union."""
    matches = {}

    def assign(index, visited):
        for target in edges[index]:
            if target in visited:
                continue
            visited.add(target)
            if target not in matches or assign(matches[target], visited):
                matches[target] = index
                return True
        return False

    edges = [
        [target for target, label in enumerate(labels)
         if pred["category"] == label["category"] and overlap(pred, label) >= threshold]
        for pred in predictions
    ]
    for index in range(len(edges)):
        assign(index, set())
    return len(matches)


def load(manifest):
    clips = json.loads(manifest.read_text())
    seen = set()
    for clip in clips:
        if not clip["source"].strip() or not clip["condition"].strip():
            raise ValueError("Every clip needs a source and condition")
        report = json.loads((manifest.parent / clip["report"]).read_text())
        if (report["status"] not in ("complete", "partial")
                or not report["duration"] or report["processing_seconds"] <= 0):
            raise ValueError(f"Expected a completed video attempt: {clip['report']}")
        if report["id"] in seen:
            raise ValueError("A report appears more than once")
        seen.add(report["id"])
        for label in clip["labels"]:
            if (label["category"] not in CATEGORIES[report["category"]]
                    or not 0 <= label["start"] < label["end"] <= report["duration"]):
                raise ValueError(f"Invalid label interval or category: {label}")
        yield clip, report


def evaluate(clips, iou, hits=None, gap=None):
    # tp, fp, fn, media seconds, worker seconds, failed samples, clips
    counts = defaultdict(lambda: [0] * 7)
    for clip, report in clips:
        policy = dict(report["policy"])
        if hits is not None:
            policy["min_hits"] = hits
        if gap is not None:
            policy["max_gap_samples"] = gap
        predictions = group_events(report["frames"], report["duration"], report["fps"],
                                   CATEGORIES[report["category"]], **policy)
        labels = clip["labels"]
        tp = match(predictions, labels, iou)
        for group in ("all", "condition/" + clip["condition"]):
            values = (tp, len(predictions) - tp, len(labels) - tp,
                      report["duration"], report["processing_seconds"],
                      sum(frame["state"] == "failed" for frame in report["frames"]), 1)
            counts[group] = [total + value for total, value in zip(counts[group], values)]
    return {
        key: {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_alerts_per_video_hour": fp * 3600 / media,
            "video_seconds_per_worker_second": media / elapsed,
            "video_hours": media / 3600, "processing_seconds": elapsed,
            "failed_samples": failures, "clips": count,
        }
        for key, (tp, fp, fn, media, elapsed, failures, count) in counts.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Held-out labeled clip manifest")
    parser.add_argument("--tune", type=Path, help="Independent development manifest")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    if not math.isfinite(args.iou) or not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")
    clips = list(load(args.manifest))
    if not clips:
        parser.error("The held-out manifest is empty")
    output = {"iou": args.iou, "models": sorted({report["model"] for _, report in clips})}
    if args.tune:
        development = list(load(args.tune))
        if not development:
            parser.error("The development manifest is empty")
        if ({clip["source"] for clip, _ in development} & {clip["source"] for clip, _ in clips}
                or {report["id"] for _, report in development} & {report["id"] for _, report in clips}):
            parser.error("Development and held-out clips must have independent sources and reports")
        candidates = []
        for hits, gap in ((2, 1), (2, 2), (3, 1), (3, 2)):
            score = evaluate(development, args.iou, hits, gap)["all"]
            denominator = 2 * score["true_positives"] + score["false_positives"] + score["false_negatives"]
            f1 = 2 * score["true_positives"] / denominator if denominator else 0
            candidates.append({"min_hits": hits, "max_gap_samples": gap, "f1": f1, **score})
        winner = max(candidates, key=lambda candidate: candidate["f1"])
        hits, gap = winner["min_hits"], winner["max_gap_samples"]
        output.update(development=candidates, selected={"min_hits": hits, "max_gap_samples": gap},
                      held_out=evaluate(clips, args.iou, hits, gap))
    else:
        output["held_out"] = evaluate(clips, args.iou)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
