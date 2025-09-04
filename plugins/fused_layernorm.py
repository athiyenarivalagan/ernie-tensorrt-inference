import torch
import torch.nn as nn

class FusedDenseLayerNorm(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.eps = eps
        self.dense = nn.Linear(in_features, out_features, bias=False) # bias handled by LayerNorm

    def forward(self, x, residual):
        # Dense + residual
        x = self.dense(x)
        x = x + residual

        # FP16-safe LayerNorm
        orig_dtype = x.dtype
        x_fp32 = x.float() # convert to fp32 for numerical stability
        mean = x_fp32.mean(-1, keepdim=True)
        var = ((x_fp32 - mean) ** 2).mean(-1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        
        # fused scale + bias
        x_fp32 = (x_fp32 - mean) * inv_std # Multiplication is cheaper than division on the GPU
        x_fp32 = x_fp32 * self.weight.view(1, -1) + self.bias.view(1, -1)
        
        return x_fp32.to(orig_dtype)