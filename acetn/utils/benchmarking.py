import torch
import time
import torch.cuda.nvtx as nvtx
import os
from tqdm import tqdm

@torch.no_grad()
def record_runtime(func):
    """
    A decorator that records the runtime of a function execution.

    This function measures the time taken by the decorated function to execute. 
    If a GPU is available, it uses CUDA events to measure the execution time in seconds.
    If no GPU is available, it falls back to using CPU time with `time.time()`.

    Args:
        func (function): The function whose runtime needs to be recorded.

    Returns:
        A tuple of the function's output and its runtime in seconds, or just the runtime if the function returns None.
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end)/1000
        else:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            runtime = end - start
        if result is None:
            return runtime
        elif isinstance(result, tuple):
            return (*result, runtime)
        else:
            return result, runtime
    return wrapper


@torch.no_grad()
def record_runtime_ave(func, num_record=20, num_warmup=10):
    """
    A decorator that records the average runtime of a function over multiple executions.

    This function runs the decorated function multiple times, discards the results of
    a warmup phase, and then calculates the average runtime over a specified number
    of executions. If a GPU is available, it uses CUDA events for timing. If profiling 
    is enabled via environment variables, it also supports integration with `nsys` profiling.

    Args:
        func (function): The function whose average runtime needs to be recorded.
        num_record (int, optional): The number of times to run the function for benchmarking. Default is 20.
        num_warmup (int, optional): The number of warmup iterations to run before benchmarking. Default is 10.

    Returns:
        A tuple of the function's output and its average runtime in seconds, or just the average runtime if the function returns None.
    """
    def wrapper(*args, **kwargs):
        for _ in tqdm(range(num_warmup), "Running warmup"):
            result = func(*args, **kwargs)

        runtime_ave = 0
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            is_nsys_profiling = bool(os.environ.get("NSYS_PROFILING_SESSION_ID", None))
            if is_nsys_profiling:
                torch.cuda.profiler.cudart().cudaProfilerStart()

            for i in tqdm(range(num_record), "Running benchmark"):
                nvtx.range_push(f"Iteration {i+1}")
                start.record()
                result = func(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                nvtx.range_pop()
                runtime = start.elapsed_time(end)/1000
                runtime_ave += runtime

            if is_nsys_profiling:
                torch.cuda.synchronize()
                torch.cuda.profiler.cudart().cudaProfilerStop()
        else:
            for _ in range(num_record):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                runtime = end - start
                runtime_ave += runtime
        runtime_ave /= num_record

        if result is None:
            return runtime_ave
        elif isinstance(result, tuple):
            return (*result, runtime_ave)
        else:
            return result, runtime_ave
    return wrapper
