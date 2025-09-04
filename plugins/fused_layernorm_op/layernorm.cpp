#include <torch/extension.h>
#include <vector>
#include <cmath>

torch::Tensor layernorm_forward(
    torch::Tensor x,        // [batch, hidden]
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps)
{
    x = x.contiguous();

    auto orig_dtype = x.dtype(); // save original dtype
    torch::Tensor x_fp32 = x.to(torch::kFloat32); // promote to FP32

    int64_t batch_size = x_fp32.size(0);
    int64_t hidden_size = x_fp32.size(1);

    auto y = torch::empty_like(x_fp32);

    for (int64_t b = 0; b < batch_size; b++) {
        auto row = x_fp32[b];
        auto mean = row.mean();
        auto var = row.var(false); // unbiased=false
        float inv_std = 1.0f / std::sqrt(var.item<float>() + eps);

        for (int64_t h = 0; h < hidden_size; h++) {
            float norm = (row[h].item<float>() - mean.item<float>()) * inv_std;
            y[b][h] = gamma[h].item<float>() * norm + beta[h].item<float>();
        }
    }

    // Cast back to original dtype (FP16 or FP32)
    return y.to(orig_dtype);
}