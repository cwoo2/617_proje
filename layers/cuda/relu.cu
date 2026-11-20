#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

__global__ void relu_kernel(
	const float *__restrict__ x,
	float *__restrict__ y,
	const int64_t n)
{
	const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
	if (i < n)
	{
		y[i] = x[i] > 0.0f ? x[i] : 0.0f;
	}
}

torch::Tensor my_relu(torch::Tensor input)
{
	TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
	TORCH_CHECK(
		input.scalar_type() == torch::kFloat32,
		"only float32 is supported");
	TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

	torch::Tensor output = torch::empty_like(input);

	const int64_t n = input.numel();
	if (n == 0)
	{
		return output;
	}

	const int threads = 256;
	const int64_t blocks = (n + threads - 1) / threads;
	relu_kernel<<<
		static_cast<unsigned int>(blocks),
		threads,
		0,
		at::cuda::getCurrentCUDAStream()>>>(
		input.data_ptr<float>(), output.data_ptr<float>(), n);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return output;
}
