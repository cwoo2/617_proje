#include <torch/extension.h>

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_RADIUS (KERNEL_WIDTH / 2)

__global__ void my_convolution_kernel(const float *x, float *output,
                                      const float *weight, const int C,
                                      const int H, const int W, const int O) {
	// Calculate the global thread coordinates
	const int xp = blockIdx.x * blockDim.x + threadIdx.x;
	const int yp = blockIdx.y * blockDim.y + threadIdx.y;
	const int zp = blockIdx.z;

	if (xp >= W || yp >= H) {
		return;
	}
}

torch::Tensor conv3x3(torch::Tensor input, torch::Tensor weight,
                      int64_t padding) {
	// input is  N, C, H, W
	// weight is O, C, 3, 3
	TORCH_CHECK(input.is_cuda() && weight.is_cuda(),
	            "all tensors must be CUDA tensors");
	TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
	                weight.scalar_type() == torch::kFloat32,
	            "only float32 is supported");
	TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
	            "expected 4D input and weight");
	TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3,
	            "kernel must be 3x3");
	TORCH_CHECK(input.size(1) == weight.size(1), "channel mismatch");
	TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
	            "tensors must be contiguous");
	int N = input.size(0);
	int C_in = input.size(1);
	int H = input.size(2);
	int W = input.size(3);
	int C_out = weight.size(0);

	torch::Tensor output = torch::zeros(
	    {N, C_out, H - (KERNEL_WIDTH - 1), W - (KERNEL_HEIGHT - 1)},
	    input.options());
}