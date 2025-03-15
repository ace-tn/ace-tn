import torch

def print_device_info():
    import os
    import platform
    print("== CPU Info ==")
    print("Processor type:", platform.machine())
    print("Number of logical processors:", os.cpu_count())
    print("Number of threads PyTorch is using:", torch.get_num_threads())
    print("Number of inter-op threads PyTorch is using:", torch.get_num_interop_threads())

    print("\n== GPU Info ==")
    if torch.cuda.is_available():
        print("Number of CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            device_props = torch.cuda.get_device_properties(i)
            print(f"  - CUDA Capability: {device_props.major}.{device_props.minor}")
            print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  - Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
            print(f"  - Max Threads per Multiprocessor: {torch.cuda.get_device_properties(i).max_threads_per_multi_processor}")
    else:
        print("\nCUDA is not available on this system.")

    print("\n== PyTorch Backend Info ==")
    print("OpenMP enabled:", torch.backends.openmp.is_available())
    print("MKL enabled:", torch.backends.mkl.is_available())
    print('\n')
