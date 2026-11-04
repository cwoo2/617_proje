#include <torch/extension.h>

torch::Tensor thrust_conv3x3(torch::Tensor input, torch::Tensor weight,
                             torch::Tensor bias, int64_t padding);
