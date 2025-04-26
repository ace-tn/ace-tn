#include <torch/extension.h>
#define CUTENSOR_VERSION_MAJOR 2
#include <cutensor.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
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

class CutensorS1 {
public:
    CutensorS1(int64_t nD, int64_t bD, int64_t pD) {
        handle_ = GetCuTensorHandle();

        std::vector<int32_t> modeA = {'Y','X','p','Q'};
        std::vector<int32_t> modeB = {'X','U','Q'};
        std::vector<int32_t> modeC = {'Y','U','p'};

        std::vector<int64_t> extentA = {nD, nD, pD, pD};
        std::vector<int64_t> extentB = {nD, bD, pD};
        std::vector<int64_t> extentC = {nD, bD, pD};

        const uint32_t alignment = 128;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &descA_, 4, extentA.data(), nullptr, CUTENSOR_R_64F, alignment));
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &descB_, 3, extentB.data(), nullptr, CUTENSOR_R_64F, alignment));
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &descC_, 3, extentC.data(), nullptr, CUTENSOR_R_64F, alignment));

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

        if (workspaceSize_ > 0) {
            cudaMalloc(&workspace_, workspaceSize_);
        }

        S_ = torch::empty({nD, bD, pD}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    }

    ~CutensorS1() {
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
        S_ = torch::empty({B.size(0), B.size(1), B.size(2)}, A.options());
        void* S_raw = S_.data_ptr<double>();

        double alpha = 1.0;
        double beta = 0.0;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        CUTENSOR_CHECK(cutensorContract(handle_, plan_,
            (void*)&alpha, A_raw, B_raw,
            (void*)&beta,  S_raw, S_raw,
            workspace_, workspaceSize_, stream));
    }

    torch::Tensor tensor() const { return S_; }

private:
    cutensorHandle_t handle_;
    cutensorPlan_t plan_;
    cutensorOperationDescriptor_t opDesc_;
    cutensorTensorDescriptor_t descA_, descB_, descC_;
    uint64_t workspaceSize_;
    void* workspace_ = nullptr;
    torch::Tensor S_;
};

torch::Tensor cutensor_build_s1(torch::Tensor n12g, torch::Tensor a2r) {
    int64_t nD = a2r.size(0); 
    int64_t bD = a2r.size(1);
    int64_t pD = a2r.size(2);
    static CutensorS1 s1(nD, bD, pD);
    s1.build(n12g, a2r);
    return s1.tensor();
}

static auto registry = torch::RegisterOperators("cutensor_ext::build_s1", &cutensor_build_s1);
