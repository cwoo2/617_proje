#include <torch/extension.h>

torch::Tensor my_convolution(torch::Tensor input, torch::Tensor weight);
torch::Tensor my_relu(torch::Tensor input);
torch::Tensor my_maxpool(torch::Tensor input, const int kernel_size, int stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("my_convolution", &my_convolution, "3x3 conv, padding=0");
	m.def("my_relu", &my_relu, "ReLU");
	m.def(
		"my_maxpool",
		&my_maxpool,
		"MaxPool2d",
		py::arg("input"),
		py::arg("kernel_size"),
		py::arg("stride") = -1);
}
