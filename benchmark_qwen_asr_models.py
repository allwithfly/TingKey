from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from asr_media_utils import (
    default_batch_size,
    expand_media_inputs,
    format_hms,
    get_media_duration_seconds,
    normalize_cli_path,
)
from speech_output import QwenSpeechOutput


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def similarity_ratio(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return 1.0 - (levenshtein_distance(a, b) / denom)


def cer(pred: str, ref: str) -> float:
    denom = max(len(ref), 1)
    return levenshtein_distance(pred, ref) / denom


def load_references(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    ref_path = Path(normalize_cli_path(path)).expanduser().resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    suffix = ref_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(ref_path.read_text(encoding="utf-8"))
        refs: dict[str, str] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                refs[str(Path(k).name)] = str(v)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "audio" in item and "text" in item:
                    refs[str(Path(str(item["audio"])).name)] = str(item["text"])
        return refs

    if suffix == ".csv":
        refs = {}
        with ref_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio = row.get("audio") or row.get("path") or row.get("file")
                text = row.get("text") or row.get("label")
                if audio and text is not None:
                    refs[str(Path(audio).name)] = str(text)
        return refs

    raise ValueError("Reference file must be .json or .csv")


def run_model(
    model_name: str,
    model_path: str,
    files: list[str],
    device: str,
    dtype: str,
    backend: str,
    gpu_memory_utilization: float,
    attn_implementation: str,
    enable_tf32: bool,
    cudnn_benchmark: bool,
    language: str | None,
    max_new_tokens: int,
    max_inference_batch_size: int,
    warmup_runs: int,
    repeat_per_file: int,
) -> tuple[dict[str, dict[str, Any]], float]:
    print(f"[INFO] Loading {model_name}: {model_path}")
    t0 = time.time()
    engine = QwenSpeechOutput(
        model_path=model_path,
        device=device,
        dtype=dtype,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        gpu_memory_utilization=gpu_memory_utilization,
        attn_implementation=attn_implementation,  # type: ignore[arg-type]
        enable_tf32=enable_tf32,
        cudnn_benchmark=cudnn_benchmark,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    load_time_s = time.time() - t0
    print(f"[INFO] {model_name} loaded in {load_time_s:.2f}s")

    outputs: dict[str, dict[str, Any]] = {}
    warmup_count = min(max(warmup_runs, 0), len(files))
    if warmup_count > 0:
        print(f"[INFO] {model_name} warmup runs: {warmup_count}")
        for warmup_audio in files[:warmup_count]:
            _ = engine.transcribe(
                audio=warmup_audio,
                language=language,
                return_time_stamps=False,
            )[0]

    outputs: dict[str, dict[str, Any]] = {}
    for idx, audio in enumerate(files, start=1):
        result: dict[str, Any] | None = None
        time_samples: list[float] = []
        for _ in range(max(1, repeat_per_file)):
            begin = time.perf_counter()
            one = engine.transcribe(
                audio=audio,
                language=language,
                return_time_stamps=False,
            )[0]
            time_samples.append(time.perf_counter() - begin)
            if result is None:
                result = one
        assert result is not None
        model_time = sum(time_samples) / len(time_samples)
        outputs[audio] = {
            "text": result.get("text", ""),
            "language": result.get("language"),
            "model_inference_time_s": model_time,
        }
        print(
            f"[{model_name} {idx}/{len(files)}] "
            f"{Path(audio).name} time={model_time:.3f}s"
        )

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs, load_time_s


def build_summary(rows: list[dict[str, Any]], has_ref: bool) -> dict[str, Any]:
    count = len(rows)
    avg_t06 = sum(r["time_06_s"] for r in rows) / count if count else 0.0
    avg_t17 = sum(r["time_17_s"] for r in rows) / count if count else 0.0

    valid_rtf_06 = [r["rtf_06"] for r in rows if r["rtf_06"] is not None]
    valid_rtf_17 = [r["rtf_17"] for r in rows if r["rtf_17"] is not None]
    avg_rtf_06 = sum(valid_rtf_06) / len(valid_rtf_06) if valid_rtf_06 else None
    avg_rtf_17 = sum(valid_rtf_17) / len(valid_rtf_17) if valid_rtf_17 else None

    avg_similarity = sum(r["text_similarity"] for r in rows) / count if count else 0.0
    summary: dict[str, Any] = {
        "file_count": count,
        "avg_model_inference_time_06_s": avg_t06,
        "avg_model_inference_time_17_s": avg_t17,
        "speedup_06_vs_17": (avg_t17 / avg_t06) if avg_t06 > 0 else None,
        "avg_rtf_06": avg_rtf_06,
        "avg_rtf_17": avg_rtf_17,
        "avg_text_similarity_between_models": avg_similarity,
    }

    if has_ref:
        valid_cer_06 = [r["cer_06"] for r in rows if r["cer_06"] is not None]
        valid_cer_17 = [r["cer_17"] for r in rows if r["cer_17"] is not None]
        summary["avg_cer_06"] = (
            sum(valid_cer_06) / len(valid_cer_06) if valid_cer_06 else None
        )
        summary["avg_cer_17"] = (
            sum(valid_cer_17) / len(valid_cer_17) if valid_cer_17 else None
        )
        summary["cer_samples"] = len(valid_cer_06)

    return summary


def build_markdown(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    has_ref: bool,
) -> str:
    lines: list[str] = []
    lines.append("# Qwen ASR Benchmark Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- files: {summary['file_count']}")
    lines.append(
        f"- avg model_inference_time (0.6b): {summary['avg_model_inference_time_06_s']:.3f}s"
    )
    lines.append(
        f"- avg model_inference_time (1.7b): {summary['avg_model_inference_time_17_s']:.3f}s"
    )
    if summary.get("model_load_time_06_s") is not None:
        lines.append(f"- model load time (0.6b): {summary['model_load_time_06_s']:.3f}s")
    if summary.get("model_load_time_17_s") is not None:
        lines.append(f"- model load time (1.7b): {summary['model_load_time_17_s']:.3f}s")
    if summary.get("speedup_06_vs_17") is not None:
        lines.append(f"- speed ratio (1.7b / 0.6b): {summary['speedup_06_vs_17']:.3f}x")
    lines.append(
        f"- avg text similarity between models: {summary['avg_text_similarity_between_models']:.3f}"
    )
    if summary.get("avg_rtf_06") is not None:
        lines.append(f"- avg RTF (0.6b): {summary['avg_rtf_06']:.3f}")
    if summary.get("avg_rtf_17") is not None:
        lines.append(f"- avg RTF (1.7b): {summary['avg_rtf_17']:.3f}")
    if has_ref:
        lines.append(f"- avg CER (0.6b): {summary.get('avg_cer_06')}")
        lines.append(f"- avg CER (1.7b): {summary.get('avg_cer_17')}")
    else:
        lines.append("- accuracy: N/A (no reference labels provided)")
    lines.append("")
    lines.append("## Per File")

    header = [
        "file",
        "duration",
        "time_0.6b",
        "time_1.7b",
        "sim(06,17)",
        "text_0.6b",
        "text_1.7b",
    ]
    if has_ref:
        header.extend(["ref", "cer_0.6b", "cer_1.7b"])

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in rows:
        cells = [
            Path(row["audio"]).name,
            format_hms(row["audio_duration_s"]),
            f"{row['time_06_s']:.3f}s",
            f"{row['time_17_s']:.3f}s",
            f"{row['text_similarity']:.3f}",
            row["text_06"].replace("\n", " "),
            row["text_17"].replace("\n", " "),
        ]
        if has_ref:
            cells.extend(
                [
                    (row.get("reference_text") or "").replace("\n", " "),
                    "" if row["cer_06"] is None else f"{row['cer_06']:.4f}",
                    "" if row["cer_17"] is None else f"{row['cer_17']:.4f}",
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-ASR 0.6B vs 1.7B on local audio files",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help='Audio file paths/patterns, e.g. "C:\\\\...\\\\recordings\\\\*.wav"',
    )
    parser.add_argument("--model-06-dir", default="model")
    parser.add_argument("--model-17-dir", default="model-1.7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="vLLM gpu memory utilization",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation for transformers backend",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 matmul/cudnn acceleration on CUDA",
    )
    parser.add_argument(
        "--disable-cudnn-benchmark",
        action="store_true",
        help="Disable cudnn benchmark autotune",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-06", type=int, default=default_batch_size("0.6b"))
    parser.add_argument("--batch-17", type=int, default=default_batch_size("1.7b"))
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-per-file", type=int, default=1)
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Low-latency preset for short utterances (may reduce long-audio robustness)",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Optional .json/.csv with ground truth text for CER",
    )
    parser.add_argument("--out-json", default="benchmark_report.json")
    parser.add_argument("--out-md", default="benchmark_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast_mode:
        if args.dtype == "auto":
            args.dtype = "float16"
        if args.max_new_tokens == 256:
            args.max_new_tokens = 96
        if args.attn_implementation == "auto":
            args.attn_implementation = "flash_attention_2"
    if args.backend == "vllm" and platform.system().lower().startswith("win"):
        print(
            "[ERROR] vLLM backend is not supported natively on Windows. Run in WSL2/Linux.",
            file=sys.stderr,
        )
        return

    files, missing = expand_media_inputs(args.inputs)
    files = [f for f in files if Path(f).suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}]
    if missing:
        print(f"[WARN] Skipped unmatched inputs: {missing}")
    if not files:
        raise RuntimeError("No matched audio files.")
    files.sort()
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    model_06_dir = str(Path(normalize_cli_path(args.model_06_dir)).expanduser().resolve())
    model_17_dir = str(Path(normalize_cli_path(args.model_17_dir)).expanduser().resolve())
    references = load_references(args.reference_file)

    durations = {f: get_media_duration_seconds(f) for f in files}

    results_06, load_06_s = run_model(
        model_name="0.6b",
        model_path=model_06_dir,
        files=files,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attn_implementation=args.attn_implementation,
        enable_tf32=not args.disable_tf32,
        cudnn_benchmark=not args.disable_cudnn_benchmark,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=args.batch_06,
        warmup_runs=args.warmup_runs,
        repeat_per_file=args.repeat_per_file,
    )

    results_17, load_17_s = run_model(
        model_name="1.7b",
        model_path=model_17_dir,
        files=files,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attn_implementation=args.attn_implementation,
        enable_tf32=not args.disable_tf32,
        cudnn_benchmark=not args.disable_cudnn_benchmark,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=args.batch_17,
        warmup_runs=args.warmup_runs,
        repeat_per_file=args.repeat_per_file,
    )

    rows: list[dict[str, Any]] = []
    for audio in files:
        text_06 = results_06[audio]["text"]
        text_17 = results_17[audio]["text"]
        duration = durations.get(audio)
        time_06 = float(results_06[audio]["model_inference_time_s"])
        time_17 = float(results_17[audio]["model_inference_time_s"])
        ref_text = references.get(Path(audio).name)

        row: dict[str, Any] = {
            "audio": audio,
            "audio_duration_s": duration,
            "time_06_s": time_06,
            "time_17_s": time_17,
            "rtf_06": (time_06 / duration) if duration and duration > 0 else None,
            "rtf_17": (time_17 / duration) if duration and duration > 0 else None,
            "text_06": text_06,
            "text_17": text_17,
            "text_similarity": similarity_ratio(text_06, text_17),
            "reference_text": ref_text,
            "cer_06": cer(text_06, ref_text) if ref_text is not None else None,
            "cer_17": cer(text_17, ref_text) if ref_text is not None else None,
        }
        rows.append(row)

    summary = build_summary(rows, has_ref=bool(references))
    summary["model_load_time_06_s"] = load_06_s
    summary["model_load_time_17_s"] = load_17_s
    report = {
        "summary": summary,
        "rows": rows,
        "config": {
            "model_06_dir": model_06_dir,
            "model_17_dir": model_17_dir,
            "device": args.device,
            "backend": args.backend,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "attn_implementation": args.attn_implementation,
            "enable_tf32": not args.disable_tf32,
            "cudnn_benchmark": not args.disable_cudnn_benchmark,
            "dtype": args.dtype,
            "language": args.language,
            "max_new_tokens": args.max_new_tokens,
            "batch_06": args.batch_06,
            "batch_17": args.batch_17,
            "warmup_runs": args.warmup_runs,
            "repeat_per_file": args.repeat_per_file,
        },
    }

    out_json = Path(normalize_cli_path(args.out_json)).expanduser().resolve()
    out_md = Path(normalize_cli_path(args.out_md)).expanduser().resolve()
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(
        build_markdown(summary, rows, has_ref=bool(references)),
        encoding="utf-8",
    )

    print("[DONE] Benchmark finished.")
    print(f"[DONE] JSON: {out_json}")
    print(f"[DONE] Markdown: {out_md}")


if __name__ == "__main__":
    main()
