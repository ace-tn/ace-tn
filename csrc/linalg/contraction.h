#include <cutensor.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cassert>

using torch::Tensor;

#define CUTENSOR_CHECK(call) { \
    cutensorStatus_t err = call; \
    if (err != CUTENSOR_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuTENSOR error: ") + cutensorGetErrorString(err)); \
    } \
}

template<typename T>
struct CuTensorTypeTraits;

template<>
struct CuTensorTypeTraits<float> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_32F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_32F;}
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<double> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_64F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_64F;}
  typedef double ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<float>> {
  static cutensorDataType_t getDataType() {return CUTENSOR_C_32F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_32F;}
  typedef c10::complex<float> ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<double>> {
  static cutensorDataType_t getDataType() {return CUTENSOR_C_64F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_64F;}
  typedef c10::complex<double> ScalarType;
};

inline cutensorComputeDescriptor_t resolveComputeDescriptor(std::string& compute_type) {
  if (compute_type == "float") {
      return CUTENSOR_COMPUTE_DESC_32F;
  } else if (compute_type == "double") {
      return CUTENSOR_COMPUTE_DESC_64F;
  } else {
      throw std::runtime_error("Unknown compute type: " + compute_type);
  }
}

inline std::vector<int64_t> compute_row_major_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

class CutensorHandle {
public:
  CutensorHandle() {
    CUTENSOR_CHECK(cutensorCreate(&handle_));
  }
  ~CutensorHandle() {
    cutensorDestroy(handle_);
  }
  static cutensorHandle_t get() { 
    static CutensorHandle instance;
    return instance.handle_;
  }
  CutensorHandle(const CutensorHandle&) = delete;
  CutensorHandle& operator=(const CutensorHandle&) = delete;
  CutensorHandle(CutensorHandle&&) = delete;
  CutensorHandle& operator=(CutensorHandle&&) = delete;
private:
  cutensorHandle_t handle_;
};

struct ContractionPlan {
  void* contraction = nullptr;
  uint64_t workspace_size = 0;
  Tensor workspace;
  Tensor input0;
  Tensor input1;
  Tensor tensorC;
  std::string einsum_expr;
  std::string data_type;
  std::string compute_type;
};

// Parse einsum expression to extract modes and compute output shape
inline void parseEinsumExpression(std::string const& einsum_expr,
                                   std::vector<int64_t> const& extentA,
                                   std::vector<int64_t> const& extentB,
                                   std::vector<int64_t>& extentC,
                                   std::vector<int32_t>& modeA,
                                   std::vector<int32_t>& modeB,
                                   std::vector<int32_t>& modeC) {
    // Parse format: "labelsA,labelsB->labelsC" or "labelsA,labelsB->"
    size_t arrow_pos = einsum_expr.find("->");
    if (arrow_pos == std::string::npos) {
        throw std::runtime_error("Invalid einsum expression: missing '->'");
    }
    
    std::string left = einsum_expr.substr(0, arrow_pos);
    std::string right = einsum_expr.substr(arrow_pos + 2);
    
    size_t comma_pos = left.find(',');
    if (comma_pos == std::string::npos) {
        throw std::runtime_error("Invalid einsum expression: missing ','");
    }
    
    std::string labelsA = left.substr(0, comma_pos);
    std::string labelsB = left.substr(comma_pos + 1);
    std::string labelsC = right;
    
    // Extract mode labels for A and B
    modeA.clear();
    modeB.clear();
    modeC.clear();
    
    for (char c : labelsA) {
        modeA.push_back(static_cast<int32_t>(c));
    }
    for (char c : labelsB) {
        modeB.push_back(static_cast<int32_t>(c));
    }
    for (char c : labelsC) {
        modeC.push_back(static_cast<int32_t>(c));
    }
    
    // Validate dimensions
    if (modeA.size() != extentA.size()) {
        throw std::runtime_error("Mode A size mismatch");
    }
    if (modeB.size() != extentB.size()) {
        throw std::runtime_error("Mode B size mismatch");
    }
    
    // Compute output shape from labelsC
    extentC.clear();
    if (labelsC.empty()) {
        // Scalar output
        extentC = {};
    } else {
        // Create a map from label to dimension
        std::unordered_map<char, int64_t> label_to_dim;
        
        // First, collect dimensions from A
        for (size_t i = 0; i < modeA.size(); ++i) {
            char label = static_cast<char>(modeA[i]);
            if (label_to_dim.find(label) == label_to_dim.end()) {
                label_to_dim[label] = extentA[i];
            }
        }
        
        // Then from B (may override if same label appears in both)
        for (size_t i = 0; i < modeB.size(); ++i) {
            char label = static_cast<char>(modeB[i]);
            if (label_to_dim.find(label) == label_to_dim.end()) {
                label_to_dim[label] = extentB[i];
            } else if (label_to_dim[label] != extentB[i]) {
                throw std::runtime_error("Dimension mismatch for label in A and B");
            }
        }
        
        // Build output shape from labelsC
        for (char label : labelsC) {
            if (label_to_dim.find(label) == label_to_dim.end()) {
                throw std::runtime_error("Label '" + std::string(1, label) + "' in output not found in inputs. Expression: " + einsum_expr);
            }
            extentC.push_back(label_to_dim[label]);
        }
    }
}

template<typename ComputeType>
class Contraction {
public:
    Contraction(std::string const& einsum_expr,
                Tensor const& tensorA,
                Tensor const& tensorB,
                cutensorComputeDescriptor_t const& computeDesc)
            : numModesA_(tensorA.dim()),
              numModesB_(tensorB.dim()),
              extentA_(tensorA.sizes().vec()),
              extentB_(tensorB.sizes().vec()),
              strideA_(compute_row_major_strides(extentA_)),
              strideB_(compute_row_major_strides(extentB_)),
              workspace_(nullptr),
              isInitialized_(false),
              workspaceSet_(false),
              workspaceSize_(0),
              computeDesc_(computeDesc),
              plan_(nullptr),
              opDesc_(nullptr) {
        handle_ = CutensorHandle::get();
        parseEinsumExpression(einsum_expr, extentA_, extentB_, extentC_, modeA_, modeB_, modeC_);
        numModesC_ = extentC_.size();
        strideC_ = compute_row_major_strides(extentC_);
        isInitialized_ = true;
    }

    ~Contraction() {
        if (isInitialized_ && plan_) { 
            cutensorDestroyPlan(plan_); 
        }
        if (opDesc_) {
            cutensorDestroyOperationDescriptor(opDesc_);
        }
        if (workspace_) {
            cudaFree(workspace_);
        }
    }

    bool isInitialized() const { return isInitialized_; }
    
    std::vector<int64_t> getOutputShape() const { return extentC_; }
    
    uint64_t getWorkspaceSize() const { return workspaceSize_; }

    bool plan(cutensorHandle_t handle) {
        if (!isInitialized_) { return false; }

        cutensorDataType_t const dataType = CuTensorTypeTraits<ComputeType>::getDataType();
        uint32_t const alignment = 128;

        cutensorTensorDescriptor_t descA = nullptr;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            handle, &descA, numModesA_, extentA_.data(), strideA_.data(), dataType, alignment));

        cutensorTensorDescriptor_t descB = nullptr;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            handle, &descB, numModesB_, extentB_.data(), strideB_.data(), dataType, alignment));

        cutensorTensorDescriptor_t descC = nullptr;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            handle, &descC, numModesC_, extentC_.data(), strideC_.data(), dataType, alignment));

        CUTENSOR_CHECK(cutensorCreateContraction(handle, &opDesc_,
            descA, modeA_.data(), CUTENSOR_OP_IDENTITY,
            descB, modeB_.data(), CUTENSOR_OP_IDENTITY,
            descC, modeC_.data(), CUTENSOR_OP_IDENTITY,
            descC, modeC_.data(), computeDesc_));

        cutensorPlanPreference_t planPreference;
        CUTENSOR_CHECK(cutensorCreatePlanPreference(handle, &planPreference, CUTENSOR_ALGO_TTGT, CUTENSOR_JIT_MODE_NONE));

        uint64_t workspaceEstimate = 0;
        cutensorWorksizePreference_t worksizePreference = CUTENSOR_WORKSPACE_DEFAULT;
        CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle, opDesc_, planPreference, worksizePreference, &workspaceEstimate));

        CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan_, opDesc_, planPreference, workspaceEstimate));
        CUTENSOR_CHECK(cutensorPlanGetAttribute(handle, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspaceSize_, sizeof(workspaceSize_)));

        if (workspaceSize_ > 0) { cudaMalloc(&workspace_, workspaceSize_); }

        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descA));
        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descB));
        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descC));

        return true;
    }

    bool execute(cutensorHandle_t handle,
                 void* const A_raw,
                 void* const B_raw, 
                 void* C_raw,
                 void* workspace,
                 cudaStream_t stream) const {
      if (!isInitialized_) { return false; }

      uint32_t const alignment = 128;
      assert(uintptr_t(A_raw) % alignment == 0);
      assert(uintptr_t(B_raw) % alignment == 0);
      assert(uintptr_t(C_raw) % alignment == 0);

      typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1.0;
      typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0.0;

      CUTENSOR_CHECK(cutensorContract(handle, plan_,
                                      &alpha, A_raw, B_raw,
                                      &beta,  C_raw, C_raw,
                                      workspace, workspaceSize_, stream));
      return true;
    }

private:
    cutensorHandle_t handle_;
    cutensorPlan_t plan_;
    cutensorOperationDescriptor_t opDesc_;
    cutensorComputeDescriptor_t computeDesc_;
    int32_t numModesA_, numModesB_, numModesC_;
    std::vector<int32_t> modeA_, modeB_, modeC_;
    std::vector<int64_t> extentA_, extentB_, extentC_;
    std::vector<int64_t> strideA_, strideB_, strideC_;
    uint64_t workspaceSize_;
    void* workspace_;
    bool isInitialized_;
    bool workspaceSet_;
};

ContractionPlan create_contraction_plan(std::string const& einsum_expr,
                                        Tensor const& tensorA,
                                        Tensor const& tensorB,
                                        std::string const& data_type,
                                        std::string const& compute_type) {
  ContractionPlan plan;
  plan.einsum_expr = einsum_expr;
  plan.data_type = data_type;
  plan.compute_type = compute_type;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(tensorA.scalar_type(), "create_contraction_plan", [&] {
    cutensorComputeDescriptor_t const resolved_compute_desc = CuTensorTypeTraits<scalar_t>::getComputeDesc();

    auto* contraction = new Contraction<scalar_t>(einsum_expr, tensorA, tensorB, resolved_compute_desc);
    plan.contraction = contraction;
    if (!contraction->isInitialized()) {
      delete contraction;
      throw std::runtime_error("cuTENSOR error: Contraction not initialized");
    }
    plan.tensorC = torch::empty(contraction->getOutputShape(), tensorA.options());

    auto handle = CutensorHandle::get();
    if (!contraction->plan(handle)) {
      delete contraction;
      throw std::runtime_error("cuTENSOR error: Contraction plan failed");
    }

    plan.workspace_size = contraction->getWorkspaceSize();
    if (plan.workspace_size > 0) {
      plan.workspace = at::empty(plan.workspace_size, at::CUDA(at::kByte));
    }
  });

  return plan;
}

cutensorStatus_t execute_contraction(ContractionPlan& plan, Tensor const& tensorA, Tensor const& tensorB, Tensor& tensorC) {
  if (!plan.contraction) {
    throw std::runtime_error("cuTENSOR error: Invalid contraction plan");
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(tensorA.scalar_type(), "execute_contraction", [&] {
    auto* contraction = static_cast<Contraction<scalar_t>*>(plan.contraction);
    auto handle = CutensorHandle::get();
    auto stream = at::cuda::getCurrentCUDAStream();
    
    void* workspace = plan.workspace_size > 0 ? plan.workspace.data_ptr<uint8_t>() : nullptr;
    if (!contraction->execute(handle,
                              tensorA.data_ptr<scalar_t>(),
                              tensorB.data_ptr<scalar_t>(),
                              tensorC.data_ptr<scalar_t>(),
                              workspace,
                              stream)) {
      throw std::runtime_error("cuTENSOR error: Contraction execution failed");
    }
  });

  return cutensorStatus_t::CUTENSOR_STATUS_SUCCESS;
}
