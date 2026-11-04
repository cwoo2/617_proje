# STL-10 CNN with a Thrust conv3x3

A small PyTorch CNN where the middle convolution (32 -> 64 channels) is
implemented with a Thrust `transform` kernel (`thrust_conv.cu`), compiled at
runtime with `torch.utils.cpp_extension.load_inline`.

## Environment

Tested with:

- Python 3.14, PyTorch 2.9.1+cu130, torchvision 0.24.1+cu130
- CUDA 13.3 (`nvcc`) and driver 13.5
- NVIDIA GTX 1660 Ti (compute capability 7.5)

The CPU-only `torch` package from the distro will not work; the CUDA wheel is
required.

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.1 torchvision==0.24.1
```

`train_stl10_thrust.py` sets `TORCH_CUDA_ARCH_LIST` from the detected GPU, so
the extension is built for the local card on first import. The build is cached
in `~/.cache/torch_extensions/`.

## Dataset

Torchvision downloads STL-10 from `ai.stanford.edu`, which may be unreachable.
The same original archive is available from Hugging Face:

```bash
mkdir -p data
curl -L -o /tmp/stl10_binary.tar.gz \
    https://huggingface.co/datasets/xingslong/stl10_binary/resolve/main/stl10_binary.tar.gz
tar -xzf /tmp/stl10_binary.tar.gz -C data
```

This should create `data/stl10_binary/{train,test,unlabeled}_*.bin`.

## Running

```bash
# Thrust model (the first run also compiles the CUDA extension)
.venv/bin/python train_stl10_thrust.py --epochs 10

# Baseline / cuDNN model
.venv/bin/python train_stl10.py --epochs 10

# Kernel microbenchmark against cuDNN
.venv/bin/python bench_conv.py

# Run both trainings and compare timings
.venv/bin/python compare_training.py --epochs 10
```

`compare_training.py` automatically uses `.venv/bin/python` when that venv
exists.

## Benchmarking

Use three levels, from cheapest to most detailed.

### 1. End-to-end op timing

`bench_conv.py` times the whole conv call with CUDA events, so the result
includes kernel launch and Python overhead:

```bash
.venv/bin/python bench_conv.py
.venv/bin/python bench_conv.py --iters 200 --warmup 50
.venv/bin/python bench_conv.py --cudnn-benchmark
```

Example on the GTX 1660 Ti for `N=64 C=32 H=48 W=48 O=64`:

| setup | cuDNN | Thrust | Thrust vs cuDNN |
|---|---:|---:|---:|
| default heuristics | 2.07 ms (2.62 TFLOP/s) | 12.67 ms (0.43 TFLOP/s) | 0.16x |
| `--cudnn-benchmark` | 1.70 ms (3.21 TFLOP/s) | 9.52 ms (0.57 TFLOP/s) | 0.18x |

`--only cudnn` and `--only thrust` run one implementation. The printed
`max |cudnn-thrust|` confirms both versions compute the same result.

### 2. Training comparison

```bash
.venv/bin/python compare_training.py --epochs 10
```

This runs both training scripts and compares total/mean/median epoch time and
images/s. Use the same `--epochs`, `--batch-size`, `--seed` and `--workers` for
both, which the script already does.

### 3. GPU timeline with Nsight Systems

Nsight Systems shows which kernels run, how long each takes, and how much of
the wall time is CPU/launch overhead:

```bash
nsys profile -t cuda -o /tmp/conv_trace --force-overwrite true \
    .venv/bin/python bench_conv.py --iters 20 --warmup 5
nsys stats --report cuda_gpu_kern_sum /tmp/conv_trace.nsys-rep
```

For the Thrust path you will see one CUB `static_kernel` per conv call. The
cuDNN path uses several Winograd kernels plus an SGEMM, so compare the *sum* of
those kernels, not a single row.

### 4. Kernel metrics with Nsight Compute

`ncu` is installed outside `PATH` on this machine:

```bash
NCU=/usr/local/cuda-13.3/bin/ncu

# Thrust kernel
$NCU --kernel-name regex:static_kernel --launch-count 1 --set full \
    .venv/bin/python bench_conv.py --only thrust --iters 1 --warmup 0

# cuDNN Winograd kernels
$NCU --kernel-name regex:winogradForward --launch-count 6 --set full \
    .venv/bin/python bench_conv.py --only cudnn --iters 1 --warmup 0
```

The Thrust kernel is named `static_kernel` because it is the CUB kernel behind
`thrust::transform`; filtering by `Conv3x3Functor` will not match.

For a short, comparable metric table instead of the full report:

```bash
METRICS=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed
METRICS=$METRICS,l1tex__throughput.avg.pct_of_peak_sustained_elapsed
METRICS=$METRICS,dram__throughput.avg.pct_of_peak_sustained_elapsed
METRICS=$METRICS,sm__warps_active.avg.pct_of_peak_sustained_active
METRICS=$METRICS,launch__registers_per_thread

$NCU --kernel-name regex:static_kernel --launch-count 1 --metrics "$METRICS" \
    .venv/bin/python bench_conv.py --only thrust --iters 1 --warmup 0
```

What the metrics mean:

- `gpu__time_duration.sum`: actual kernel time, excluding launch gaps.
- `sm__throughput`: how busy the compute pipelines are.
- `l1tex__throughput`: L1/load-store pipeline pressure. Near 100% means memory
  access or address computation is the limit.
- `dram__throughput`: external memory bandwidth. Low means the kernel is not
  DRAM-bound.
- `sm__warps_active`: achieved occupancy.
- `launch__registers_per_thread`: register pressure; high values can limit
  occupancy.

Measured for the Thrust kernel: ~98% occupancy, ~83% SM throughput, ~93% L1TEX
throughput, but only ~4% DRAM throughput. That points at the per-output index
arithmetic and repeated global loads, not at memory bandwidth or occupancy.

Note that Nsight Compute replays kernels to collect metrics, so the `ms/iter`
printed during profiling is not a valid performance measurement. Use
`bench_conv.py` without a profiler for timings, and `ncu` only for the kernel
counters.

