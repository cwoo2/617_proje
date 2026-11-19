#include <torch/extension.h>

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_RADIUS (KERNEL_WIDTH / 2)

__global__ void my_convolution_kernel(
	const float *x,
	float *output,
	const float *weight,
	const int C,
	const int H,
	const int W,
	const int O)
{
	// Calculate the global thread coordinates
	const int xp = blockIdx.x * blockDim.x + threadIdx.x;
	const int yp = blockIdx.y * blockDim.y + threadIdx.y;
	const int zp = blockIdx.z;

	if (xp >= W || yp >= H)
	{
		return;
	}
}

__device__ void cross_correlation(
	const float *const input,
	const float *const kernel,
	float *output,
	const int H,
	const int W)
{
	const int H_out = H - (KERNEL_HEIGHT - 1);
	const int W_out = W - (KERNEL_WIDTH - 1);

	for (int i = 0; i < H_out; i++)
	{
		for (int j = 0; j < W_out; j++)
		{
			float acc = 0.0f;
			for (int p = 0; p < KERNEL_HEIGHT; p++)
			{
				for (int q = 0; q < KERNEL_WIDTH; q++)
				{
					const int ix = j + q;
					const int iy = i + p;
					const int row = iy * W;
					acc += input[row + ix] * kernel[(p * KERNEL_WIDTH) + q];
				}
			}
			output[i * W_out + j] += acc;
		}
	}
}

__device__ void conv2d(
	const float *const input,
	const float *const weights,
	float *output,
	const int N,
	const int C_in,
	const int H,
	const int W,
	const int C_out)
{
	const int H_out = H - (KERNEL_HEIGHT - 1);
	const int W_out = W - (KERNEL_WIDTH - 1);
	const int feature_map_size = W_out * H_out;
	for (int i = 0; i < N; i++)
	{
		const int image_addr = i * C_in * H * W;
		for (int j = 0; j < C_out; j++)
		{
			const int output_channel_addr = j * feature_map_size;
			const int feature_map_addr = i * C_out * feature_map_size;
			float *out_ij = output + feature_map_addr + output_channel_addr;
			for (int k = 0; k < C_in; k++)
			{
				const int channel_addr = k * W * H;
				const float *const input_ik = input + image_addr + channel_addr;

				const int kernel_addr = k * KERNEL_WIDTH * KERNEL_HEIGHT;
				const int weight_addr = j * C_in * KERNEL_WIDTH * KERNEL_HEIGHT;

				const float *const weight_jk =
					weights + weight_addr + kernel_addr;
				cross_correlation(input_ik, weight_jk, out_ij, H, W);
			}
		}
	}
}

torch::Tensor conv3x3(torch::Tensor input, torch::Tensor weight)
{
	// input is  N, C, H, W
	// weight is O, C, 3, 3
	TORCH_CHECK(
		input.is_cuda() && weight.is_cuda(),
		"all tensors must be CUDA tensors");
	TORCH_CHECK(
		input.scalar_type() == torch::kFloat32 &&
			weight.scalar_type() == torch::kFloat32,
		"only float32 is supported");
	TORCH_CHECK(
		input.dim() == 4 && weight.dim() == 4,
		"expected 4D input and weight");
	TORCH_CHECK(
		weight.size(2) == 3 && weight.size(3) == 3,
		"kernel must be 3x3");
	TORCH_CHECK(input.size(1) == weight.size(1), "channel mismatch");
	TORCH_CHECK(
		input.is_contiguous() && weight.is_contiguous(),
		"tensors must be contiguous");
	int N = input.size(0);
	int C_in = input.size(1);
	int H = input.size(2);
	int W = input.size(3);
	int C_out = weight.size(0);

	torch::Tensor output = torch::zeros(
		{N, C_out, H - (KERNEL_HEIGHT - 1), W - (KERNEL_WIDTH - 1)},
		input.options());

	return output;
}