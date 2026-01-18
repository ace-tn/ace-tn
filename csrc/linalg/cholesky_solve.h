#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include <stdexcept>

#define CUSOLVER_CHECK(call) { cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        throw std::runtime_error("cuSOLVER error code: " + std::to_string(err)); } }

class CusolverHandle {
public:
    CusolverHandle() {
        CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    }
    ~CusolverHandle() {
        cusolverDnDestroy(handle_);
    }
    static cusolverDnHandle_t get() {
        static CusolverHandle instance;
        return instance.handle_;
    }
    CusolverHandle(const CusolverHandle&) = delete;
    CusolverHandle& operator=(const CusolverHandle&) = delete;
    CusolverHandle(CusolverHandle&&) = delete;
    CusolverHandle& operator=(CusolverHandle&&) = delete;
private:
    cusolverDnHandle_t handle_;
};

template<typename T> struct CudaType;
template<> struct CudaType<float> { using type = float; };
template<> struct CudaType<double> { using type = double; };
template<> struct CudaType<c10::complex<float>> { using type = cuComplex; };
template<> struct CudaType<c10::complex<double>> { using type = cuDoubleComplex; };

// Overloaded wrappers for cuSOLVER functions
inline cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, cublasFillMode_t u, int n, float* A, int lda, int* lwork) {
    return cusolverDnSpotrf_bufferSize(h, u, n, A, lda, lwork);
}
inline cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, cublasFillMode_t u, int n, double* A, int lda, int* lwork) {
    return cusolverDnDpotrf_bufferSize(h, u, n, A, lda, lwork);
}
inline cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, cublasFillMode_t u, int n, cuComplex* A, int lda, int* lwork) {
    return cusolverDnCpotrf_bufferSize(h, u, n, A, lda, lwork);
}
inline cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, cublasFillMode_t u, int n, cuDoubleComplex* A, int lda, int* lwork) {
    return cusolverDnZpotrf_bufferSize(h, u, n, A, lda, lwork);
}

inline cusolverStatus_t potrf(cusolverDnHandle_t h, cublasFillMode_t u, int n, float* A, int lda, float* work, int lwork, int* info) {
    return cusolverDnSpotrf(h, u, n, A, lda, work, lwork, info);
}
inline cusolverStatus_t potrf(cusolverDnHandle_t h, cublasFillMode_t u, int n, double* A, int lda, double* work, int lwork, int* info) {
    return cusolverDnDpotrf(h, u, n, A, lda, work, lwork, info);
}
inline cusolverStatus_t potrf(cusolverDnHandle_t h, cublasFillMode_t u, int n, cuComplex* A, int lda, cuComplex* work, int lwork, int* info) {
    return cusolverDnCpotrf(h, u, n, A, lda, work, lwork, info);
}
inline cusolverStatus_t potrf(cusolverDnHandle_t h, cublasFillMode_t u, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* work, int lwork, int* info) {
    return cusolverDnZpotrf(h, u, n, A, lda, work, lwork, info);
}

inline cusolverStatus_t potrs(cusolverDnHandle_t h, cublasFillMode_t u, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* info) {
    return cusolverDnSpotrs(h, u, n, nrhs, A, lda, B, ldb, info);
}
inline cusolverStatus_t potrs(cusolverDnHandle_t h, cublasFillMode_t u, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* info) {
    return cusolverDnDpotrs(h, u, n, nrhs, A, lda, B, ldb, info);
}
inline cusolverStatus_t potrs(cusolverDnHandle_t h, cublasFillMode_t u, int n, int nrhs, const cuComplex* A, int lda, cuComplex* B, int ldb, int* info) {
    return cusolverDnCpotrs(h, u, n, nrhs, A, lda, B, ldb, info);
}
inline cusolverStatus_t potrs(cusolverDnHandle_t h, cublasFillMode_t u, int n, int nrhs, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, int* info) {
    return cusolverDnZpotrs(h, u, n, nrhs, A, lda, B, ldb, info);
}

namespace cusolver_ext {

template<typename scalar_t>
torch::Tensor cholesky_solve_impl(torch::Tensor A, torch::Tensor B) {
    using cuda_t = typename CudaType<scalar_t>::type;

    int n = static_cast<int>(A.size(0));
    int nrhs = static_cast<int>(B.size(1));

    auto A_ = A.t().contiguous().t().clone();
    auto B_ = B.t().contiguous().t().clone();

    auto A_ptr = reinterpret_cast<cuda_t*>(A_.data_ptr<scalar_t>());
    auto B_ptr = reinterpret_cast<cuda_t*>(B_.data_ptr<scalar_t>());

    cusolverDnHandle_t handle = CusolverHandle::get();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

    int lwork = 0;
    CUSOLVER_CHECK(potrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, A_ptr, n, &lwork));

    auto workspace = torch::empty({lwork}, A.options());
    auto work_ptr = reinterpret_cast<cuda_t*>(workspace.data_ptr<scalar_t>());

    auto devInfo = torch::empty({1}, A.options().dtype(torch::kInt32));
    auto info_ptr = devInfo.data_ptr<int>();

    CUSOLVER_CHECK(potrf(handle, CUBLAS_FILL_MODE_LOWER, n, A_ptr, n, work_ptr, lwork, info_ptr));
    CUSOLVER_CHECK(potrs(handle, CUBLAS_FILL_MODE_LOWER, n, nrhs, A_ptr, n, B_ptr, n, info_ptr));

    return B_.contiguous();
}

inline torch::Tensor cholesky_solve(torch::Tensor A, torch::Tensor B) {
    torch::Tensor result;
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "cholesky_solve", [&] {
        result = cholesky_solve_impl<scalar_t>(A, B);
    });
    return result;
}

} // namespace cusolver_ext
