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

        extentC_ = extentC;
    }

    ~CutensorContraction() {
        if (workspace_) cudaFree(workspace_);
        cutensorDestroyPlan(plan_);
        cutensorDestroyOperationDescriptor(opDesc_);
        cutensorDestroyTensorDescriptor(descA_);
        cutensorDestroyTensorDescriptor(descB_);
        cutensorDestroyTensorDescriptor(descC_);
    }

    torch::Tensor build(const torch::Tensor& A, const torch::Tensor& B) {
        const void* A_raw = A.data_ptr<double>();
        const void* B_raw = B.data_ptr<double>();
        torch::Tensor C = torch::empty(extentC_, A.options());
        void* C_raw = C.data_ptr<double>();

        double alpha = 1.0, beta = 0.0;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        CUTENSOR_CHECK(cutensorContract(handle_, plan_,
            &alpha, A_raw, B_raw,
            &beta,  C_raw, C_raw,
            workspace_, workspaceSize_, stream));
        return C;
    }

private:
    cutensorHandle_t handle_;
    cutensorPlan_t plan_;
    cutensorOperationDescriptor_t opDesc_;
    cutensorTensorDescriptor_t descA_, descB_, descC_;
    uint64_t workspaceSize_;
    void* workspace_;
    std::vector<int64_t> extentC_;
};

torch::Tensor cutensor_build_s1(torch::Tensor n12g, torch::Tensor a2r) {
    static int64_t nD = a2r.size(0);
    static int64_t bD = a2r.size(1);
    static int64_t pD = a2r.size(2);

    static const std::vector<int64_t> extentA = {nD, nD, pD, pD};
    static const std::vector<int64_t> extentB = {nD, bD, pD};
    static const std::vector<int64_t> extentC = {nD, bD, pD};

    static const std::vector<int32_t> modeA = {'Y','X','p','Q'};
    static const std::vector<int32_t> modeB = {'X','U','Q'};
    static const std::vector<int32_t> modeC = {'Y','U','p'};

    static CutensorContraction s1(modeA, modeB, modeC, 
                                  extentA, extentB, extentC);
    return s1.build(n12g, a2r);
}

torch::Tensor cutensor_build_s2(torch::Tensor n12g, torch::Tensor a1r) {
    static int64_t nD = a1r.size(0);
    static int64_t bD = a1r.size(1);
    static int64_t pD = a1r.size(2);

    static const std::vector<int64_t> extentA = {nD, nD, pD, pD};
    static const std::vector<int64_t> extentB = {nD, bD, pD};
    static const std::vector<int64_t> extentC = {nD, bD, pD};

    static const std::vector<int32_t> modeA = {'Y','X','P','q'};
    static const std::vector<int32_t> modeB = {'Y','V','P'};
    static const std::vector<int32_t> modeC = {'X','V','q'};

    static CutensorContraction s2(modeA, modeB, modeC,
                                  extentA, extentB, extentC);
    return s2.build(n12g, a1r);
}

torch::Tensor cutensor_build_r1(torch::Tensor n12, torch::Tensor a2r) {
    static int64_t nD = a2r.size(0);
    static int64_t bD = a2r.size(1);
    static int64_t pD = a2r.size(2);

    static const std::vector<int64_t> extentA_intm = {nD, nD, nD, nD};
    static const std::vector<int64_t> extentB_intm = {nD, bD, pD};
    static const std::vector<int64_t> extentC_intm = {nD, nD, nD, bD, pD};

    static const std::vector<int32_t> modeA_intm = {'y','x','Y','X'};
    static const std::vector<int32_t> modeB_intm = {'x','u','q'};
    static const std::vector<int32_t> modeC_intm = {'y','Y','X','u','q'};

    static CutensorContraction r1_intm(modeA_intm, modeB_intm, modeC_intm, 
                                       extentA_intm, extentB_intm, extentC_intm);

    static const std::vector<int64_t> extentA = {nD, nD, nD, bD, pD};
    static const std::vector<int64_t> extentB = {nD, bD, pD};
    static const std::vector<int64_t> extentC = {nD, bD, nD, bD};

    static const std::vector<int32_t> modeA = {'y','Y','X','u','Q'};
    static const std::vector<int32_t> modeB = {'X','U','Q'};
    static const std::vector<int32_t> modeC = {'Y','U','y','u'};

    static CutensorContraction r1(modeA, modeB, modeC, 
                                  extentA, extentB, extentC);
    return r1.build(r1_intm.build(n12, a2r), a2r);
}

torch::Tensor cutensor_build_r2(torch::Tensor n12, torch::Tensor a1r) {
    static int64_t nD = a1r.size(0);
    static int64_t bD = a1r.size(1);
    static int64_t pD = a1r.size(2);

    static const std::vector<int64_t> extentA_intm = {nD, nD, nD, nD};
    static const std::vector<int64_t> extentB_intm = {nD, bD, pD};
    static const std::vector<int64_t> extentC_intm = {nD, nD, nD, bD, pD};

    static const std::vector<int32_t> modeA_intm = {'y','x','Y','X'};
    static const std::vector<int32_t> modeB_intm = {'y','v','p'};
    static const std::vector<int32_t> modeC_intm = {'x','Y','X','v','p'};

    static CutensorContraction r2_intm(modeA_intm, modeB_intm, modeC_intm, 
                                       extentA_intm, extentB_intm, extentC_intm);

    static const std::vector<int64_t> extentA = {nD, nD, nD, bD, pD};
    static const std::vector<int64_t> extentB = {nD, bD, pD};
    static const std::vector<int64_t> extentC = {nD, bD, nD, bD};

    static const std::vector<int32_t> modeA = {'x','Y','X','v','P'};
    static const std::vector<int32_t> modeB = {'Y','V','P'};
    static const std::vector<int32_t> modeC = {'X','V','x','v'};

    static CutensorContraction r2(modeA, modeB, modeC, 
                                  extentA, extentB, extentC);
    return r2.build(r2_intm.build(n12, a1r), a1r);
}

TORCH_LIBRARY(cutensor_ext, m) {
    m.def("build_s1", &cutensor_build_s1);
    m.def("build_s2", &cutensor_build_s2);
    m.def("build_r1", &cutensor_build_r1);
    m.def("build_r2", &cutensor_build_r2);
}
