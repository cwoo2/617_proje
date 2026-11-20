from pathlib import Path

from torch.utils.cpp_extension import load

_SOURCE_DIR = Path(__file__).resolve().parent

my_cnn = load(
    name="my_cnn",
    sources=[
        str(_SOURCE_DIR / "conv.cu"),
        str(_SOURCE_DIR / "relu.cu"),
        str(_SOURCE_DIR / "bindings.cpp"),
    ],
    extra_cuda_cflags=["-O3"],
)
