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
        CUTENSOR_CHECK(cutensorPlanGetAttribute(handle_, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspaceSize_, sizeof(workspaceSize_)));

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

    static CutensorContraction s1({'Y','X','p','Q'}, {'X','U','Q'}, {'Y','U','p'}, 
                                  {nD, nD, pD, pD},  {nD, bD, pD},  {nD, bD, pD});
    return s1.build(n12g, a2r);
}

torch::Tensor cutensor_build_s2(torch::Tensor n12g, torch::Tensor a1r) {
    static int64_t nD = a1r.size(0);
    static int64_t bD = a1r.size(1);
    static int64_t pD = a1r.size(2);

    static CutensorContraction s2({'Y','X','P','q'}, {'Y','V','P'}, {'X','V','q'},
                                  {nD, nD, pD, pD},  {nD, bD, pD},  {nD, bD, pD});
    return s2.build(n12g, a1r);
}

torch::Tensor cutensor_build_r1(torch::Tensor n12, torch::Tensor a2r) {
    static int64_t nD = a2r.size(0);
    static int64_t bD = a2r.size(1);
    static int64_t pD = a2r.size(2);

    static CutensorContraction r1_intm({'y','x','Y','X'}, {'x','u','q'}, {'y','Y','X','u','q'}, 
                                       {nD, nD, nD, nD},  {nD, bD, pD},  {nD, nD, nD, bD, pD});

    static CutensorContraction r1({'y','Y','X','u','Q'}, {'X','U','Q'}, {'Y','U','y','u'}, 
                                  {nD, nD, nD, bD, pD},  {nD, bD, pD},  {nD, bD, nD, bD});

    return r1.build(r1_intm.build(n12, a2r), a2r);
}

torch::Tensor cutensor_build_r2(torch::Tensor n12, torch::Tensor a1r) {
    static int64_t nD = a1r.size(0);
    static int64_t bD = a1r.size(1);
    static int64_t pD = a1r.size(2);

    static CutensorContraction r2_intm({'y','x','Y','X'}, {'y','v','p'}, {'x','Y','X','v','p'}, 
                                       {nD, nD, nD, nD},  {nD, bD, pD},  {nD, nD, nD, bD, pD});

    static CutensorContraction r2({'x','Y','X','v','P'}, {'Y','V','P'}, {'X','V','x','v'}, 
                                  {nD, nD, nD, bD, pD},  {nD, bD, pD},  {nD, bD, nD, bD});

    return r2.build(r2_intm.build(n12, a1r), a1r);
}

torch::Tensor cutensor_calculate_cost(torch::Tensor a1r,
                                      torch::Tensor a2r,
                                      torch::Tensor a12g,
                                      torch::Tensor n12)
{
    static int64_t nD = a1r.size(0);
    static int64_t bD = a1r.size(1);
    static int64_t pD = a1r.size(2);

    static CutensorContraction a12({'y','u','p'}, {'x','u','q'}, {'y','x','p','q'}, 
                                   {nD, bD, pD},  {nD, bD, pD},  {nD, nD, pD, pD});

    static CutensorContraction d_intm({'y','x','Y','X'}, {'y','x','p','q'}, {'Y','X','p','q'}, 
                                      {nD, nD, nD, nD},  {nD, nD, pD, pD},  {nD, nD, pD, pD});

    static CutensorContraction d({'Y','X','p','q'}, {'Y','X','p','q'}, {}, 
                                 {nD, nD, pD, pD},  {nD, nD, pD, pD},  {});

    torch::Tensor a12n = a12.build(a1r, a2r);
    torch::Tensor d2 = d.build(d_intm.build(n12, a12n), a12n);
    torch::Tensor d3 = d.build(d_intm.build(n12, a12g), a12n);
    return d2 - 2*d3;
}

torch::Tensor cutensor_build_norm_tensor(
    torch::Tensor c12, torch::Tensor e12, torch::Tensor e11,
    torch::Tensor c13, torch::Tensor e13, torch::Tensor a1q,
    torch::Tensor c21, torch::Tensor e21, torch::Tensor e24,
    torch::Tensor c24, torch::Tensor e23, torch::Tensor a2q) {

    static int64_t nD = a1q.size(3);
    static int64_t bD = a1q.size(0);
    static int64_t cD = c12.size(0);

    static CutensorContraction tmp1({'a','b'}, {'b','c','r','R'}, {'a','c','r','R'},
                                     {cD, cD}, {cD, cD, bD, bD},  {cD, cD, bD, bD});

    static CutensorContraction tmp2({'a','c','r','R'}, {'e','a','u','U'}, {'c','r','R','e','u','U'},
                                     {cD, cD, bD, bD}, {cD, cD, bD, bD},  {cD, bD, bD, cD, bD, bD});

    static CutensorContraction tmp3({'c','r','R','e','u','U'}, {'R','D','U','Y'}, {'c','r','e','u','D','Y'},
                                    {cD, bD, bD, cD, bD, bD},  {bD, bD, bD, nD},  {cD, bD, cD, bD, bD, nD});

    static CutensorContraction tmp4({'c','r','e','u','D','Y'}, {'r','d','u','y'}, {'c','e','D','Y','d','y'},
                                    {cD, bD, cD, bD, bD, nD},  {bD, bD, bD, nD},  {cD, cD, bD, nD, bD, nD});

    static CutensorContraction tmp5({'a','b'}, {'b','f','d','D'}, {'a','f','d','D'},
                                    {cD, cD},  {cD, cD, bD, bD},  {cD, cD, bD, bD});

    static CutensorContraction tmp6({'a','f','d','D'}, {'a','e','D','Y','d','y'}, {'f','e','Y','y'},
                                    {cD, cD, bD, bD},  {cD, cD, bD, nD, bD, nD},  {cD, cD, nD, nD});

    static CutensorContraction tmp7({'a','b'}, {'b','c','u','U'}, {'a','c','u','U'},
                                    {cD, cD},  {cD, cD, bD, bD},  {cD, cD, bD, bD});

    static CutensorContraction tmp8({'a','c','u','U'}, {'e','a','l','L'}, {'c','u','U','e','l','L'},
                                    {cD, cD, bD, bD},  {cD, cD, bD, bD},  {cD, bD, bD, cD, bD, bD});

    static CutensorContraction tmp9({'c','u','U','e','l','L'}, {'D','L','U','X'}, {'c','u','e','l','X','D'},
                                    {cD, bD, bD, cD, bD, bD},  {bD, bD, bD, nD},  {cD, bD, cD, bD, nD, bD});

    static CutensorContraction tmp10({'c','u','e','l','X','D'}, {'d','l','u','x'}, {'c','e','X','D','x','d'},
                                     {cD, bD, cD, bD, nD, bD},  {bD, bD, bD, nD},  {cD, cD, nD, bD, nD, bD});

    static CutensorContraction tmp11({'a','b'}, {'f','a','d','D'}, {'b','f','d','D'},
                                     {cD, cD},  {cD, cD, bD, bD},  {cD, cD, bD, bD});

    static CutensorContraction tmp12({'b','f','d','D'}, {'c','b','X','D','x','d'}, {'f','c','X','x'},
                                     {cD, cD, bD, bD},  {cD, cD, nD, bD, nD, bD},  {cD, cD, nD, nD});

    static CutensorContraction finalC({'f','c','Y','y'}, {'f','c','X','x'}, {'y','x','Y','X'},
                                      {cD, cD, nD, nD},  {cD, cD, nD, nD},  {nD, nD, nD, nD});

    torch::Tensor t1  = tmp1.build(c12, e12);
    torch::Tensor t2  = tmp2.build(t1, e11);   t1.reset();
    torch::Tensor t3  = tmp3.build(t2, a1q);   t2.reset();
    torch::Tensor t4  = tmp4.build(t3, a1q);   t3.reset();
    torch::Tensor t5  = tmp5.build(c13, e13);
    torch::Tensor t6  = tmp6.build(t5, t4);    t4.reset(); t5.reset();
    torch::Tensor t7  = tmp7.build(c21, e21);
    torch::Tensor t8  = tmp8.build(t7, e24);   t7.reset();
    torch::Tensor t9  = tmp9.build(t8, a2q);   t8.reset();
    torch::Tensor t10 = tmp10.build(t9, a2q);  t9.reset();
    torch::Tensor t11 = tmp11.build(c24, e23);
    torch::Tensor t12 = tmp12.build(t11, t10); t10.reset(); t11.reset();

    return finalC.build(t6, t12);
}

TORCH_LIBRARY(cutensor_ext, m) {
    m.def("build_s1", &cutensor_build_s1);
    m.def("build_s2", &cutensor_build_s2);
    m.def("build_r1", &cutensor_build_r1);
    m.def("build_r2", &cutensor_build_r2);
    m.def("calculate_cost", &cutensor_calculate_cost);
    m.def("build_norm_tensor", &cutensor_build_norm_tensor);
}
