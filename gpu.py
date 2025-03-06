import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  - Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")
else:
    print("CUDA is not available on this machine.")
