#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn.functional as F

from train_stl10_thrust import thrust_ext


def parse_args():
    parser = argparse.ArgumentParser(
        description="Microbenchmark the Thrust conv3x3 against cuDNN"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--only", choices=["both", "cudnn", "thrust"], default="both"
    )
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser.parse_args()


def time_call(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    device = torch.device("cuda")
    n, c, h, w, o = (
        args.batch_size,
        args.in_channels,
        args.height,
        args.width,
        args.out_channels,
    )
    x = torch.randn(n, c, h, w, device=device)
    weight = torch.randn(o, c, 3, 3, device=device) * 0.05
    bias = torch.randn(o, device=device) * 0.05

    macs = n * o * h * w * c * 9
    flops = 2 * macs

    with torch.no_grad():
        y_cudnn = F.conv2d(x, weight, bias, padding=1)
        y_thrust = thrust_ext.thrust_conv3x3(x, weight, bias, 1)
    max_error = (y_cudnn - y_thrust).abs().max().item()

    print(
        f"Device: {torch.cuda.get_device_name(0)} | "
        f"torch {torch.__version__} | CUDA {torch.version.cuda}"
    )
    print(
        f"Shape: N={n} C={c} H={h} W={w} O={o} | "
        f"kernel=3x3 stride=1 pad=1 | dtype=float32"
    )
    print(
        f"Work: {macs / 1e6:.2f} MMAC ({flops / 1e9:.2f} GFLOP) per iteration | "
        f"max |cudnn-thrust| = {max_error:.2e}"
    )
    print()
    print(f"{'Method':<10}{'ms/iter':>12}{'GFLOP/s':>12}{'Speedup':>10}")

    results = {}
    if args.only in ("both", "cudnn"):
        ms = time_call(
            lambda: F.conv2d(x, weight, bias, padding=1),
            args.iters,
            args.warmup,
        )
        results["cuDNN"] = ms
        print(f"{'cuDNN':<10}{ms:>12.3f}{flops / (ms * 1e6):>12.1f}{1.0:>10.2f}x")

    if args.only in ("both", "thrust"):
        ms = time_call(
            lambda: thrust_ext.thrust_conv3x3(x, weight, bias, 1),
            args.iters,
            args.warmup,
        )
        results["Thrust"] = ms
        speedup = results["cuDNN"] / ms if "cuDNN" in results else 1.0
        print(
            f"{'Thrust':<10}{ms:>12.3f}{flops / (ms * 1e6):>12.1f}{speedup:>10.2f}x"
        )


if __name__ == "__main__":
    main()
