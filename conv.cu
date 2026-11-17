#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_RADIUS (KERNEL_WIDTH / 2)

__global__ void my_convolution(const float *const input, float *output,
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