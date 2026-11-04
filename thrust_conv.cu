#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

struct Conv3x3Functor {
  const float* input;
  const float* weight;
  const float* bias;
  int N, C, H, W, O, pad;

  __host__ __device__ float operator()(long long idx) const {
    int ow = static_cast<int>(idx % W);
    long long t = idx / W;
    int oh = static_cast<int>(t % H);
    t /= H;
    int oc = static_cast<int>(t % O);
    int n = static_cast<int>(t / O);

    float acc = bias[oc];
    for (int ic = 0; ic < C; ++ic) {
      const float* in_base = input + static_cast<long long>(n * C + ic) * H * W;
      const float* w_base = weight + static_cast<long long>(oc * C + ic) * 9;
      for (int kh = 0; kh < 3; ++kh) {
        int ih = oh + kh - pad;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
          int iw = ow + kw - pad;
          if (iw < 0 || iw >= W) continue;
          acc += in_base[ih * W + iw] * w_base[kh * 3 + kw];
        }
      }
    }
    return acc;
  }
};

torch::Tensor thrust_conv3x3(torch::Tensor input, torch::Tensor weight,
                             torch::Tensor bias, int64_t padding) {
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
  TORCH_CHECK(weight.size(0) == bias.size(0), "bias size mismatch");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  bias.is_contiguous(),
              "tensors must be contiguous");

  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t H = input.size(2);
  int64_t W = input.size(3);
  int64_t O = weight.size(0);

  auto out = torch::empty({N, O, H, W}, input.options());
  long long total = N * O * H * W;
  if (total == 0) return out;

  Conv3x3Functor functor{input.data_ptr<float>(), weight.data_ptr<float>(),
                         bias.data_ptr<float>(), static_cast<int>(N),
                         static_cast<int>(C), static_cast<int>(H),
                         static_cast<int>(W), static_cast<int>(O),
                         static_cast<int>(padding)};

  thrust::transform(thrust::device,
                    thrust::counting_iterator<long long>(0),
                    thrust::counting_iterator<long long>(total),
                    thrust::device_ptr<float>(out.data_ptr<float>()), functor);
  return out;
}
