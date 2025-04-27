#include <torch/extension.h>
#include <cutensor.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <stdexcept>

#define CUTENSOR_CHECK(call) { cutensorStatus_t err = call; \
    if (err != CUTENSOR_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuTENSOR error: ") + cutensorGetErrorString(err)); } }

inline cutensorHandle_t CreateCuTensorHandle() {
    cutensorHandle_t handle;
    cutensorCreate(&handle);
    return handle;
}

inline cutensorHandle_t GetCuTensorHandle() {
    static thread_local cutensorHandle_t handle = CreateCuTensorHandle();
    return handle;
}

class CutensorContraction {
public:
    CutensorContraction(const std::vector<int32_t>& modeA,
                        const std::vector<int32_t>& modeB,
                        const std::vector<int32_t>& modeC,
                        const std::vector<int64_t>& extentA,
                        const std::vector<int64_t>& extentB,
                        const std::vector<int64_t>& extentC)
        : workspace_(nullptr)
    {
        handle_ = GetCuTensorHandle();

        const uint32_t alignment = 128;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &descA_, extentA.size(), extentA.data(), nullptr, CUTENSOR_R_64F, alignment));
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &descB_, extentB.size(), extentB.data(), nullptr, CUTENSOR_R_64F, alignment));
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &descC_, extentC.size(), extentC.data(), nullptr, CUTENSOR_R_64F, alignment));

        CUTENSOR_CHECK(cutensorCreateContraction(handle_, &opDesc_,
            descA_, modeA.data(), CUTENSOR_OP_IDENTITY,
            descB_, modeB.data(), CUTENSOR_OP_IDENTITY,
            descC_, modeC.data(), CUTENSOR_OP_IDENTITY,
            descC_, modeC.data(), CUTENSOR_COMPUTE_DESC_64F));

        cutensorPlanPreference_t pref;
        CUTENSOR_CHECK(cutensorCreatePlanPreference(handle_, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

        uint64_t workspaceEstimate = 0;
        CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle_, opDesc_, pref, CUTENSOR_WORKSPACE_DEFAULT, &workspaceEstimate));
        CUTENSOR_CHECK(cutensorCreatePlan(handle_, &plan_, opDesc_, pref, workspaceEstimate));
        CUTENSOR_CHECK(cutensorPlanGetAttribute(handle_, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                                &workspaceSize_, sizeof(workspaceSize_)));

        if (workspaceSize_ > 0) { cudaMalloc(&workspace_, workspaceSize_); }

        C_ = torch::empty(extentC, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    }

    ~CutensorContraction() {
        if (workspace_) cudaFree(workspace_);
        cutensorDestroyPlan(plan_);
        cutensorDestroyOperationDescriptor(opDesc_);
        cutensorDestroyTensorDescriptor(descA_);
        cutensorDestroyTensorDescriptor(descB_);
        cutensorDestroyTensorDescriptor(descC_);
    }

    void build(const torch::Tensor& A, const torch::Tensor& B) {
        const void* A_raw = A.data_ptr<double>();
        const void* B_raw = B.data_ptr<double>();
        void* C_raw = C_.data_ptr<double>();

        double alpha = 1.0, beta = 0.0;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        CUTENSOR_CHECK(cutensorContract(handle_, plan_,
            &alpha, A_raw, B_raw,
            &beta,  C_raw, C_raw,
            workspace_, workspaceSize_, stream));
    }

    torch::Tensor tensor() const { return C_; }

private:
    cutensorHandle_t handle_;
    cutensorPlan_t plan_;
    cutensorOperationDescriptor_t opDesc_;
    cutensorTensorDescriptor_t descA_, descB_, descC_;
    uint64_t workspaceSize_;
    void* workspace_;
    torch::Tensor C_;
};

torch::Tensor cutensor_build_s1(torch::Tensor A, torch::Tensor B) {
    static int64_t nD = B.size(0);
    static int64_t bD = B.size(1);
    static int64_t pD = B.size(2);

    static const std::vector<int64_t> extentA_s1 = {nD, nD, pD, pD};
    static const std::vector<int64_t> extentB_s1 = {nD, bD, pD};
    static const std::vector<int64_t> extentC_s1 = {nD, bD, pD};

    static const std::vector<int32_t> modeA_s1 = {'Y','X','p','Q'};
    static const std::vector<int32_t> modeB_s1 = {'X','U','Q'};
    static const std::vector<int32_t> modeC_s1 = {'Y','U','p'};

    static CutensorContraction s1(modeA_s1, modeB_s1, modeC_s1, 
                                  extentA_s1, extentB_s1, extentC_s1);
    s1.build(A, B);
    return s1.tensor();
}

torch::Tensor cutensor_build_s2(torch::Tensor A, torch::Tensor B) {
    static int64_t nD = B.size(0);
    static int64_t bD = B.size(1);
    static int64_t pD = B.size(2);

    static const std::vector<int64_t> extentA_s2 = {nD, nD, pD, pD};
    static const std::vector<int64_t> extentB_s2 = {nD, bD, pD};
    static const std::vector<int64_t> extentC_s2 = {nD, bD, pD};

    static const std::vector<int32_t> modeA_s2 = {'Y','X','P','q'};
    static const std::vector<int32_t> modeB_s2 = {'Y','V','P'};
    static const std::vector<int32_t> modeC_s2 = {'X','V','q'};

    static CutensorContraction s2(modeA_s2, modeB_s2, modeC_s2, 
                                  extentA_s2, extentB_s2, extentC_s2);
    s2.build(A, B);
    return s2.tensor();
}

TORCH_LIBRARY(cutensor_ext, m) {
    m.def("build_s1", &cutensor_build_s1);
    m.def("build_s2", &cutensor_build_s2);
}
