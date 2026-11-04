#!/usr/bin/env python3
import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EPOCH_RE = re.compile(r"\|\s*([0-9]+\.[0-9]+)s\s*$")
TRAIN_IMAGES_RE = re.compile(r"Train images:\s*(\d+)")


def default_python():
    candidate = SCRIPT_DIR / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    return str(candidate if candidate.exists() else Path(sys.executable))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline and Thrust STL-10 training, then compare timings"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=default_python())
    return parser.parse_args()


def run_training(python, script, args):
    cmd = [
        python, "-u", str(script),
        "--epochs", str(args.epochs),
        "--data-dir", args.data_dir,
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
    ]
    print(f"\n{'=' * 78}")
    print(f"Running {script.name} for {args.epochs} epochs")
    print(f"{'=' * 78}")
    print(" ".join(cmd), flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    epoch_times = []
    train_images = None

    proc = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            match = EPOCH_RE.search(line.rstrip())
            if match:
                epoch_times.append(float(match.group(1)))
            match = TRAIN_IMAGES_RE.search(line)
            if match:
                train_images = int(match.group(1))
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    wall = time.perf_counter() - start

    if returncode != 0:
        raise SystemExit(f"{script.name} exited with code {returncode}")
    if not epoch_times:
        raise SystemExit(f"Could not parse per-epoch timings from {script.name}")
    if train_images is None:
        raise SystemExit(f"Could not parse train image count from {script.name}")
    return epoch_times, train_images, wall


def summarize(times, train_images, wall):
    total = sum(times)
    return {
        "wall": wall,
        "total": total,
        "avg": total / len(times),
        "median": statistics.median(times),
        "throughput": train_images * len(times) / total,
        "best": min(times),
        "worst": max(times),
    }


def print_comparison(base, thrust):
    print(f"\n{'=' * 78}")
    print("Comparison")
    print(f"{'=' * 78}")
    header = f"{'Metric':<26}{'Baseline':>13}{'Thrust':>13}{'Change':>13}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Total training (s)", "total"),
        ("Avg epoch (s)", "avg"),
        ("Median epoch (s)", "median"),
        ("Throughput (img/s)", "throughput"),
        ("Best epoch (s)", "best"),
        ("Worst epoch (s)", "worst"),
        ("Wall incl. setup (s)", "wall"),
    ]
    for label, key in rows:
        b = base[key]
        t = thrust[key]
        change = (t - b) / b * 100.0 if b else 0.0
        print(f"{label:<26}{b:>13.2f}{t:>13.2f}{change:>12.2f}%")
    print("-" * len(header))
    ratio = thrust["total"] / base["total"]
    direction = "faster" if ratio < 1 else "slower"
    print(
        f"Thrust total training time: {ratio:.3f}x baseline "
        f"({abs(1 - ratio) * 100:.1f}% {direction} than baseline)"
    )
    print("Change = thrust vs baseline; positive means a larger value.")


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")

    base_times, base_images, base_wall = run_training(
        args.python, SCRIPT_DIR / "train_stl10.py", args
    )
    thrust_times, thrust_images, thrust_wall = run_training(
        args.python, SCRIPT_DIR / "train_stl10_thrust.py", args
    )

    base = summarize(base_times, base_images, base_wall)
    thrust = summarize(thrust_times, thrust_images, thrust_wall)
    print_comparison(base, thrust)


if __name__ == "__main__":
    main()
