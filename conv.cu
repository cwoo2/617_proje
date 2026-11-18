#include <torch/extension.h>

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_RADIUS (KERNEL_WIDTH / 2)

__global__ void my_convolution_kernel(const float *const input, float *output,
                               const float *const mask, const int width,
                               const int height) {
	// Calculate the global thread coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Ensure the thread is within the valid output range
	if (x >= KERNEL_RADIUS && x < width - KERNEL_RADIUS && y >= KERNEL_RADIUS &&
	    y < height - KERNEL_RADIUS) {
		float sum = 0.0f;

		// Apply the convolution kernel
		for (int ky = 0; ky < KERNEL_HEIGHT; ++ky) {
			for (int kx = 0; kx < KERNEL_WIDTH; ++kx) {
				int ix = x + kx - KERNEL_RADIUS;
				int iy = y + ky - KERNEL_RADIUS;

				sum += input[iy * width + ix] * mask[ky * KERNEL_WIDTH + kx];
			}
		}

		// Write the result to the output image
		output[y * width + x] = sum;
	}
}

torch::Tensor conv3x3(torch::Tensor input, torch::Tensor weight, int64_t padding) {
	// input is  N, C, H, W
	// weight is O, C, 3, 3
	TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
				"all tensors must be CUDA tensors");
	TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
					weight.scalar_type() == torch::kFloat32 &&
					bias.scalar_type() == torch::kFloat32,
				"only float32 is supported");
	TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
				"expected 4D input and weight");
	TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3,
				"kernel must be 3x3");
	TORCH_CHECK(input.size(1) == weight.size(1), "channel mismatch");
	TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
					bias.is_contiguous(),
				"tensors must be contiguous");
	int N = input.size(0);
	int C = input.size(1);
	int H = input.size(2); 
	int W = input.size(3);
	int O = weight.size(0);

}