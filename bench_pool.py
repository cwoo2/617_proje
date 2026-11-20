"""Run one maxpool implementation, for timing and for ncu/nsys.

    python bench_pool.py torch  --shape 32,64,28,28
    python bench_pool.py custom --shape 32,64,28,28
"""

import argparse

import torch
import torch.nn.functional as F

from my_cnn_ext import my_cnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("impl", choices=["torch", "custom"])
    parser.add_argument("--shape", default="32,64,28,28", help="N,C,H,W")
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    n, c, h, w = (int(v) for v in args.shape.split(","))
    x = torch.randn(n, c, h, w, device="cuda")
    k = args.kernel_size
    run = (lambda: F.max_pool2d(x, k)) if args.impl == "torch" else (lambda: my_cnn.my_maxpool(x, k))

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    torch.cuda.synchronize()
    print(f"{args.impl}: {start.elapsed_time(end) / args.iters * 1000:.1f} us/iter")


if __name__ == "__main__":
    main()
