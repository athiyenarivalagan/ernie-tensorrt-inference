import torch
from torch.utils.cpp_extension import load

# Compile and load C++ extension inline
layernorm_cpp = load(
    name="layernorm_cpp",
    sources=["layernorm.cpp", "binding.cpp"], # C++ files
    verbose=True
)