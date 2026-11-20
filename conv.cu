#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <torch/extension.h>

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_RADIUS (KERNEL_WIDTH / 2)
#define BLOCK_X 16
#define BLOCK_Y 16

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

__device__ void conv2d_inline(
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

				for (int y = 0; y < H_out; y++)
				{
					for (int x = 0; x < W_out; x++)
					{
						float acc = 0.0f;
						for (int p = 0; p < KERNEL_HEIGHT; p++)
						{
							for (int q = 0; q < KERNEL_WIDTH; q++)
							{
								const int ix = x + q;
								const int iy = y + p;
								const int row = iy * W;
								acc += input_ik[row + ix] *
									   weight_jk[(p * KERNEL_WIDTH) + q];
							}
						}
						out_ij[y * W_out + x] += acc;
					}
				}
			}
		}
	}
}

__global__ void conv2d_kernel(
	const float *const input,
	const float *const weights,
	float *output,
	const int C_in,
	const int H,
	const int W,
	const int C_out)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// tensor is 4D but we only get 3 dims with the grid, so
	// N and C are folded into the grid's z dim
	const int nc = blockIdx.z;
	const int n = nc / C_out;
	const int c = nc % C_out;

	const int H_out = H - (KERNEL_HEIGHT - 1);
	const int W_out = W - (KERNEL_WIDTH - 1);

	if (x >= W_out || y >= H_out)
	{
		return;
	}

	const int feature_map_size = W_out * H_out;
	const int kernel_size = KERNEL_WIDTH * KERNEL_HEIGHT;
	const int image_addr = n * C_in * H * W;

	const int output_channel_addr = c * feature_map_size;
	const int feature_map_addr = n * C_out * feature_map_size;
	float *out_nc = output + feature_map_addr + output_channel_addr;
	for (int k = 0; k < C_in; k++)
	{
		const int channel_addr = k * W * H;
		const float *const input_nk = input + image_addr + channel_addr;

		const int kernel_addr = k * kernel_size;
		const int weight_addr = c * C_in * kernel_size;

		const float *const weight_ck = weights + weight_addr + kernel_addr;

		float acc = 0.0f;
		for (int p = 0; p < KERNEL_HEIGHT; p++)
		{
			for (int q = 0; q < KERNEL_WIDTH; q++)
			{
				const int ix = x + q;
				const int iy = y + p;
				const int row = iy * W;
				acc += input_nk[row + ix] * weight_ck[(p * KERNEL_WIDTH) + q];
			}
		}
		out_nc[y * W_out + x] += acc;
	}
}

torch::Tensor my_convolution(torch::Tensor input, torch::Tensor weight)
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

	const int grid_x =
		std::ceil(static_cast<double>(W - (KERNEL_WIDTH - 1)) / BLOCK_X);
	const int grid_y =
		std::ceil(static_cast<double>(H - (KERNEL_HEIGHT - 1)) / BLOCK_Y);
	const int grid_z = N * C_out;
	conv2d_kernel<<<
		dim3(grid_x, grid_y, grid_z),
		dim3(BLOCK_X, BLOCK_Y, 1),
		0,
		at::cuda::getCurrentCUDAStream()>>>(
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		output.data_ptr<float>(),
		C_in,
		H,
		W,
		C_out);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("my_convolution", &my_convolution, "3x3 conv, padding=0");
}