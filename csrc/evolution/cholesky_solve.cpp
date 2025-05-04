#include <torch/extension.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <stdexcept>

#define CUSOLVER_CHECK(call) { cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        throw std::runtime_error("cuSOLVER error code: " + std::to_string(err)); } }

inline cusolverDnHandle_t CreateCusolverHandle() {
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    return handle;
}

inline cusolverDnHandle_t GetCusolverHandle() {
    static thread_local cusolverDnHandle_t handle = CreateCusolverHandle();
    return handle;
}

torch::Tensor cusolver_cholesky_solve(torch::Tensor A, torch::Tensor B) {
    int64_t n = A.size(0);
    int64_t nrhs = B.size(1);

    auto A_ = A.clone();
    auto B_ = B.clone();

    double* A_ptr = A_.data_ptr<double>();
    double* B_ptr = B_.data_ptr<double>();

    cusolverDnHandle_t handle = GetCusolverHandle();

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, A_ptr, n, &lwork));

    auto workspace = torch::empty({lwork}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    int* devInfo = nullptr;
    cudaMalloc(&devInfo, sizeof(int));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
    CUSOLVER_CHECK(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, A_ptr, n, workspace.data_ptr<double>(), lwork, devInfo));
    CUSOLVER_CHECK(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, nrhs, A_ptr, n, B_ptr, n, devInfo));

    cudaFree(devInfo);

    return B_;
}

TORCH_LIBRARY(cusolver_ext, m) {
    m.def("cholesky_solve", &cusolver_cholesky_solve);
}
