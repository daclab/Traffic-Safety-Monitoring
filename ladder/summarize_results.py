"""Score pulled pipeline v2 runs and write results/plan_0_3_results.md.

Run in an environment with pycocotools (qai-hub-bench):
    python ladder/summarize_results.py --results results/ladder
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import ladder_report

PROJECT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT / "ladder" / "kit" / "coco300_instances_val2017.json"
MANIFEST = PROJECT / "data" / "coco300" / "manifest.json"

SECTIONS = [
    ("ref_", "Reference: v1 best pipeline (same session)"),
    ("c2_", "#2 CPU work outside inference (4 threads, unpinned)"),
    ("p3_", "#3 Thread count and core partitioning"),
    ("i1_", "#1 Weight-only vs W8A8 variants"),
    ("l_", "Live: queue depth 1, deploy mode"),
    ("r2_", "Round 2: W8A8 int8-I/O with 4 threads, backends repeated"),
    ("cam_", "Round 2: live camera mode (newest frame only)"),
    ("r3_", "Round 3: W8A8 (1,000-image calibration) resolution sweep vs weight-only 320"),
    ("cam3_", "Round 3: live camera mode by resolution"),
    ("base_", "Project baseline (FP32 640, one core, sequential) vs chosen model"),
]


class Scorer:
    def __init__(self, annotations=ANNOTATIONS, manifest=MANIFEST):
        from pycocotools.coco import COCO

        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(str(annotations))
        self.image_ids = [i["id"] for i in json.loads(Path(manifest).read_text())["images"]]

    def score(self, predictions_path, report_path):
        from pycocotools.cocoeval import COCOeval

        predictions = json.loads(predictions_path.read_text())
        if not predictions:
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation = COCOeval(self.coco, self.coco.loadRes(predictions), "bbox")
            evaluation.params.imgIds = self.image_ids
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
        report = {
            "images": len(self.image_ids),
            "predictions": len(predictions),
            "map_50_95": float(evaluation.stats[0]),
            "map_50": float(evaluation.stats[1]),
            "map_75": float(evaluation.stats[2]),
            "ar_100": float(evaluation.stats[8]),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return report


def fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def variant(model):
    name = Path(model).stem
    for key in ("weightonly_int8", "w8a8_floatio", "w8a8_int8io"):
        if key in name:
            return key
    return "weightonly_int8 (v1)" if name.endswith("w8a8_int8") else name


def load_runs(results, scorer):
    runs = []
    for summary_path in sorted(results.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        accuracy_path = summary_path.with_name("accuracy.json")
        accuracy = None
        if accuracy_path.exists():
            accuracy = json.loads(accuracy_path.read_text())
        elif (not summary.get("live") and summary["images"] == 300
              and summary_path.with_name("predictions.json").exists()):
            accuracy = scorer.score(summary_path.with_name("predictions.json"), accuracy_path)
        runs.append((summary_path.parent.name, summary, accuracy))
    return runs


def run_table(runs):
    header = ("| Run | Model | Backend | NMS | Thr | Pinning | FPS | ms/img | Infer ms "
              "| Pre CPU ms | Post CPU ms | Proc CPU ms/frame | Busy cores "
              "| Response p50/p95 ms | mAP50-95 | Max temp C |")
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for name, s, acc in runs:
        pin = (f"infer {s['infer_cpus']} / aux {s['aux_cpus']}"
               if s.get("infer_cpus") or s.get("aux_cpus") else "none")
        temps = [v["max"] for v in s["monitor"]["temperature_c"].values()]
        resp = s["response_ms"]
        lines.append(
            f"| {name} | {variant(s['model'])} | {s['backend']} | {s['nms']} | {s['threads']} "
            f"| {pin} | **{s['throughput_fps']:.2f}** | {s['amortized_ms_per_image']:.1f} "
            f"| {s['inference_wall_ms']['mean']:.1f} | {s['preprocess_cpu_ms']['mean']:.1f} "
            f"| {s['postprocess_cpu_ms']['mean']:.1f} | {s['process_cpu_ms_per_frame']:.1f} "
            f"| {s['process_cpu_cores_busy']:.2f} | {resp['median']:.0f} / {resp['p95']:.0f} "
            f"| {fmt(100 * acc['map_50_95'], 3) if acc and s['mode'] == 'eval' else '-'} "
            f"| {fmt(max(temps) if temps else None)} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS,
                        help="COCO annotations (full val2017 or the COCO-300 subset)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--ladder-out", type=Path, default=PROJECT / "OPTIMIZATION_LADDER.md")
    args = parser.parse_args()
    scorer = Scorer(args.annotations, args.manifest)
    runs = load_runs(args.results, scorer)
    out = ["# Pipeline v2: plan items 0-3 results", ""]
    report = ladder_report.build(runs)
    if report:
        # Shareable report in the project root; this file keeps the full log.
        standalone = args.ladder_out
        standalone.write_text(report + "\n", encoding="utf-8")
        print(f"Written: {standalone}")

    sysinfo = args.results / "sysinfo.txt"
    if sysinfo.exists():
        out += ["## System", "", "```text", sysinfo.read_text().strip(), "```", ""]

    sweep = args.results / "threads_sweep.json"
    if sweep.exists():
        out += ["## #0 Thread scaling (model only, pinned to first T cores)", "",
                "| Model | Threads | Mean ms | P95 ms | Speedup | Efficiency |",
                "|---|---:|---:|---:|---:|---:|"]
        for r in json.loads(sweep.read_text()):
            out.append(f"| {Path(r['model']).name} | {r['threads']} | {r['mean_ms']:.1f} "
                       f"| {r['p95_ms']:.1f} | {fmt(r.get('speedup_vs_1_thread'), 2)} "
                       f"| {fmt(r.get('parallel_efficiency'), 2)} |")
        out.append("")

    for log in sorted(args.results.glob("letterbox_test_*.log")):
        out += [f"## Zero-copy letterbox check ({log.stem})", "", "```text",
                log.read_text().strip(), "```", ""]

    for reference in sorted(args.results.glob("ref_*/coco300_pipeline_summary.json")):
        s = json.loads(reference.read_text())
        acc_path = reference.with_name("accuracy.json")
        acc = (json.loads(acc_path.read_text()) if acc_path.exists()
               else scorer.score(reference.with_name("coco300_predictions.json"), acc_path))
        out += [f"## {SECTIONS[0][1]}", "",
                f"- {reference.parent.name}: **{s['pipeline_throughput_images_s']:.2f} FPS**, "
                f"inference {s['model_only_mean_ms']:.1f} ms, response p50 "
                f"{s['response_median_ms']:.0f} ms, mAP50-95 "
                f"{fmt(100 * acc['map_50_95'], 3) if acc else '-'}", ""]

    for prefix, title in SECTIONS[1:]:
        section = [r for r in runs if r[0].startswith(prefix)]
        if section:
            out += [f"## {title}", ""] + run_table(section) + [""]

    live_runs = [r for r in runs if r[1].get("live")]
    if live_runs:
        out += ["### Live camera latency (arrival to result)", "",
                "| Run | Camera FPS | Processed FPS | Latency p50 / p95 / max ms "
                "| Captured | Processed | Dropped stale | Missed |",
                "|---|---:|---:|---|---:|---:|---:|---:|"]
        for name, s, _ in live_runs:
            lv, resp = s["live"], s["response_ms"]
            out.append(
                f"| {name} | {lv['camera_fps']:.0f} | {s['throughput_fps']:.2f} "
                f"| {resp['median']:.0f} / {resp['p95']:.0f} / {resp['max']:.0f} "
                f"| {lv['captured']} | {lv['processed']} | {lv['dropped_stale']} "
                f"| {lv['missed_by_preprocessing'] + lv['no_free_slot']} |")
        out.append("")

    other = [r for r in runs if not any(r[0].startswith(p) for p, _ in SECTIONS)]
    if other:
        out += ["## Other runs", ""] + run_table(other) + [""]

    out += ["Busy cores = process CPU time / wall time (4.0 = all cores saturated). "
            "Pre/Post CPU = CPU time of the preprocessing/postprocessing thread per image. "
            "mAP only for eval-mode runs (confidence 0.001).", ""]
    target = args.results / "plan_0_3_results.md"
    target.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nWritten: {target}")


if __name__ == "__main__":
    main()
