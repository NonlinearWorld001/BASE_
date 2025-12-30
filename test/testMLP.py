from src.module.MLP import MLP
import torch


if __name__ == "__main__":
  gpu_available = torch.cuda.is_available()
  print(f"GPU是否可用: {gpu_available}")

  if gpu_available:
    # 1. 查看可用的GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数量: {gpu_count}")
    
    # 2. 查看GPU的具体型号（比如：NVIDIA GeForce RTX 3060）
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU型号: {gpu_name}")
    
    # 3. 查看当前使用的GPU序号
    current_gpu = torch.cuda.current_device()
    print(f"当前使用的GPU序号: {current_gpu}")
    
    # 4. 查看PyTorch对应的CUDA版本
    cuda_version = torch.version.cuda
    print(f"PyTorch适配的CUDA版本: {cuda_version}")

  mlp = MLP(
    input_dim=2,
    hidden_dim=4,
    output_dim=2,
    num_layers=2,
  )

  x = torch.tensor([[1, 2]], dtype=torch.float64)

  out = mlp(x)

  print(out)
  print(out.size())

