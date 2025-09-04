#include <torch/extension.h>

// Declare forward function
torch::Tensor layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps);

// Bind it to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CPU)");
}