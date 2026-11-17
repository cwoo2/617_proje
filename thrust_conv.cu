#include <torch/extension.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

struct Conv3x3Functor {
  static constexpr int kOcTile = 4;
  static constexpr int kOwTile = 4;

  const float* input;
  const float* weight;
  const float* bias;
  float* output;
  int N, C, H, W, O, OG, WG, pad;

  __host__ __device__ void operator()(long long idx) const {
    int ow_group = static_cast<int>(idx % WG);
    long long t = idx / WG;
    int oh = static_cast<int>(t % H);
    t /= H;
    int oc_base = static_cast<int>(t % OG) * kOcTile;
    int n = static_cast<int>(t / OG);

    int ow_base = ow_group * kOwTile;
    int ow_count = W - ow_base < kOwTile ? W - ow_base : kOwTile;
    int oc_count = O - oc_base < kOcTile ? O - oc_base : kOcTile;

    // One thread computes a kOcTile x kOwTile output block. The input window
    // is loaded once per row and reused across output channels and columns,
    // and each weight is reused across the output columns.
    float acc[kOcTile][kOwTile];
#pragma unroll
    for (int j = 0; j < kOcTile; ++j) {
#pragma unroll
      for (int i = 0; i < kOwTile; ++i) {
        acc[j][i] = (j < oc_count && i < ow_count) ? bias[oc_base + j] : 0.0f;
      }
    }

    for (int ic = 0; ic < C; ++ic) {
      const float* in_base = input + static_cast<long long>(n * C + ic) * H * W;
      const float* w_base =
          weight + static_cast<long long>(oc_base * C + ic) * 9;
      for (int kh = 0; kh < 3; ++kh) {
        int ih = oh + kh - pad;
        if (ih < 0 || ih >= H) continue;
        const float* row = in_base + ih * W;

        float v[kOwTile + 2];
#pragma unroll
        for (int x = 0; x < kOwTile + 2; ++x) {
          int iw = ow_base + x - pad;
          v[x] = (iw >= 0 && iw < W) ? row[iw] : 0.0f;
        }

        for (int kw = 0; kw < 3; ++kw) {
#pragma unroll
          for (int j = 0; j < kOcTile; ++j) {
            if (j < oc_count) {
              float w = w_base[j * C * 9 + kh * 3 + kw];
#pragma unroll
              for (int i = 0; i < kOwTile; ++i) {
                if (i < ow_count) {
                  acc[j][i] += v[i + kw] * w;
                }
              }
            }
          }
        }
      }
    }

#pragma unroll
    for (int j = 0; j < kOcTile; ++j) {
      if (j < oc_count) {
        float* out_row =
            output + (static_cast<long long>(n * O + oc_base + j) * H + oh) * W +
            ow_base;
#pragma unroll
        for (int i = 0; i < kOwTile; ++i) {
          if (i < ow_count) out_row[i] = acc[j][i];
        }
      }
    }
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
  int64_t OG = (O + Conv3x3Functor::kOcTile - 1) / Conv3x3Functor::kOcTile;
  int64_t WG = (W + Conv3x3Functor::kOwTile - 1) / Conv3x3Functor::kOwTile;

  auto out = torch::empty({N, O, H, W}, input.options());
  long long total = N * OG * H * WG;
  if (total == 0) return out;

  Conv3x3Functor functor{input.data_ptr<float>(), weight.data_ptr<float>(),
                         bias.data_ptr<float>(), out.data_ptr<float>(),
                         static_cast<int>(N), static_cast<int>(C),
                         static_cast<int>(H), static_cast<int>(W),
                         static_cast<int>(O), static_cast<int>(OG),
                         static_cast<int>(WG), static_cast<int>(padding)};

  thrust::for_each(thrust::device, thrust::counting_iterator<long long>(0),
                   thrust::counting_iterator<long long>(total), functor);
  return out;
}
