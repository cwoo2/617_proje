#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

__global__ void maxpool_kernel(
	const float *__restrict__ input,
	const int kernel_size,
	const int stride,
	const int C,
	const int H,
	const int W,
	float *__restrict__ output)
{
	const int h = blockIdx.y * blockDim.y + threadIdx.y;
	const int w = blockIdx.x * blockDim.x + threadIdx.x;
	const int nc = blockIdx.z;

	const int n = nc / C;
	const int c = nc % C;

	const int H_out = (H - kernel_size) / stride + 1;
	const int W_out = (W - kernel_size) / stride + 1;

	if (h >= H_out || w >= W_out)
	{
		return;
	}

	const int input_channel_addr = n * C * H * W + c * H * W;
	const float *input_ij = input + input_channel_addr;

	const int output_channel_addr = n * C * H_out * W_out + c * H_out * W_out;
	float *output_ij = output + output_channel_addr;

	for (int p = 0; p < kernel_size; p++)
	{
		const int iy = h * stride + p;
		for (int q = 0; q < kernel_size; q++)
		{
			const int ix = w * stride + q;
			const int row = iy * W;
			float val = input_ij[row + ix];
			if (p == 0 && q == 0)
			{
				output_ij[h * W_out + w] = val;
			}
			else
			{
				output_ij[h * W_out + w] = fmaxf(output_ij[h * W_out + w], val);
			}
		}
	}
}

torch::Tensor my_maxpool(
	torch::Tensor input,
	const int kernel_size,
	int stride = -1)
{
	TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
	TORCH_CHECK(
		input.scalar_type() == torch::kFloat32,
		"only float32 is supported");
	TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

	if (stride < 0)
	{
		stride = kernel_size;
	}

	const int64_t n = input.numel();
	const int N = input.size(0);
	const int C = input.size(1);
	const int H = input.size(2);
	const int W = input.size(3);

	const int H_out = (H - kernel_size) / stride + 1;
	const int W_out = (W - kernel_size) / stride + 1;

	torch::Tensor output = torch::empty({N, C, H_out, W_out}, input.options());
	if (n == 0)
	{
		return output;
	}

	// one thread per output pixel, one blockIdx.z per (batch, channel)
	const dim3 block(32, 8);
	const dim3 grid((W_out + 31) / 32, (H_out + 7) / 8, N * C);
	maxpool_kernel<<<
		grid,
		block,
		0,
		at::cuda::getCurrentCUDAStream()>>>(
		input.data_ptr<float>(),
		kernel_size,
		stride,
		C,
		H,
		W,
		output.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return output;
}
