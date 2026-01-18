#ifndef CONTRACTION_H
#define CONTRACTION_H

#include <cutensor.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cassert>
#include <type_traits>
#include <memory>

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

inline cutensorComputeDescriptor_t resolveComputeDescriptor(std::string const& compute_type) {
  if (compute_type == "float" || compute_type == "float32" || compute_type == "complex64") {
      return CUTENSOR_COMPUTE_DESC_32F;
  } else if (compute_type == "double" || compute_type == "float64" || compute_type == "complex128") {
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

class ContractionBase {
public:
    virtual ~ContractionBase() = default;
    virtual bool plan(cutensorHandle_t handle, 
                      cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT) = 0;
    virtual bool execute(cutensorHandle_t handle, void* A, void* B, void* C,
                         void* workspace, cudaStream_t stream) const = 0;
    virtual bool isInitialized() const = 0;
    virtual uint64_t getWorkspaceSize() const = 0;
};

struct ContractionPlan {
  std::unique_ptr<ContractionBase> contraction;
  uint64_t workspace_size = 0;
  Tensor workspace;
  std::string einsum_expr;
  std::string compute_type;
};

inline void parseEinsumExpression(std::string const& einsum_expr,
                                   std::vector<int64_t> const& extentA,
                                   std::vector<int64_t> const& extentB,
                                   std::vector<int64_t>& extentC,
                                   std::vector<int32_t>& modeA,
                                   std::vector<int32_t>& modeB,
                                   std::vector<int32_t>& modeC) {
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
    
    if (modeA.size() != extentA.size()) {
        throw std::runtime_error("Mode A size mismatch");
    }
    if (modeB.size() != extentB.size()) {
        throw std::runtime_error("Mode B size mismatch");
    }
    
    extentC.clear();
    if (labelsC.empty()) {
        extentC = {};
    } else {
        std::unordered_map<char, int64_t> label_to_dim;
        
        for (size_t i = 0; i < modeA.size(); ++i) {
            char label = static_cast<char>(modeA[i]);
            if (label_to_dim.find(label) == label_to_dim.end()) {
                label_to_dim[label] = extentA[i];
            }
        }
        
        for (size_t i = 0; i < modeB.size(); ++i) {
            char label = static_cast<char>(modeB[i]);
            if (label_to_dim.find(label) == label_to_dim.end()) {
                label_to_dim[label] = extentB[i];
            } else if (label_to_dim[label] != extentB[i]) {
                throw std::runtime_error("Dimension mismatch for label in A and B");
            }
        }
        
        for (char label : labelsC) {
            if (label_to_dim.find(label) == label_to_dim.end()) {
                throw std::runtime_error("Label '" + std::string(1, label) + "' in output not found in inputs. Expression: " + einsum_expr);
            }
            extentC.push_back(label_to_dim[label]);
        }
    }
}

template<typename T>
constexpr bool is_complex_type() {
    return std::is_same_v<T, c10::complex<float>> || std::is_same_v<T, c10::complex<double>>;
}

template<typename ComputeType>
class Contraction : public ContractionBase {
public:
    Contraction(std::string const& einsum_expr,
                Tensor const& tensorA,
                Tensor const& tensorB,
                cutensorComputeDescriptor_t const& computeDesc,
                bool conjA = false,
                bool conjB = false)
            : plan_(nullptr),
              opDesc_(nullptr),
              computeDesc_(computeDesc),
              opA_((conjA && is_complex_type<ComputeType>()) ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY),
              opB_((conjB && is_complex_type<ComputeType>()) ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY),
              einsum_expr_(einsum_expr),
              numModesA_(tensorA.dim()),
              numModesB_(tensorB.dim()),
              extentA_(tensorA.sizes().vec()),
              extentB_(tensorB.sizes().vec()),
              strideA_(compute_row_major_strides(extentA_)),
              strideB_(compute_row_major_strides(extentB_)),
              workspaceSize_(0),
              isInitialized_(false) {
    }

    ~Contraction() {
        if (isInitialized_ && plan_) { 
            cutensorDestroyPlan(plan_); 
        }
        if (opDesc_) {
            cutensorDestroyOperationDescriptor(opDesc_);
        }
    }

    bool isInitialized() const override { return isInitialized_; }

    uint64_t getWorkspaceSize() const override { return workspaceSize_; }

    bool plan(cutensorHandle_t handle,
              cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT) override {
        parseEinsumExpression(einsum_expr_, extentA_, extentB_, extentC_, modeA_, modeB_, modeC_);
        numModesC_ = extentC_.size();
        strideC_ = compute_row_major_strides(extentC_);
        isInitialized_ = true;

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
            descA, modeA_.data(), opA_,
            descB, modeB_.data(), opB_,
            descC, modeC_.data(), CUTENSOR_OP_IDENTITY,
            descC, modeC_.data(), computeDesc_));

        cutensorPlanPreference_t planPreference;
        CUTENSOR_CHECK(cutensorCreatePlanPreference(handle, &planPreference, CUTENSOR_ALGO_TTGT, CUTENSOR_JIT_MODE_NONE));

        uint64_t workspaceEstimate = 0;
        CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle, opDesc_, planPreference, workspacePref, &workspaceEstimate));

        CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan_, opDesc_, planPreference, workspaceEstimate));
        CUTENSOR_CHECK(cutensorDestroyPlanPreference(planPreference));
        CUTENSOR_CHECK(cutensorPlanGetAttribute(handle, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspaceSize_, sizeof(workspaceSize_)));

        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descA));
        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descB));
        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(descC));

        return true;
    }

    bool execute(cutensorHandle_t handle,
                 void* A_raw,
                 void* B_raw, 
                 void* C_raw,
                 void* workspace,
                 cudaStream_t stream) const override {
      if (!isInitialized_) { return false; }

      typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1.0;
      typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0.0;

      CUTENSOR_CHECK(cutensorContract(handle, plan_,
                                      &alpha, A_raw, B_raw,
                                      &beta,  C_raw, C_raw,
                                      workspace, workspaceSize_, stream));
      return true;
    }

private:
    cutensorPlan_t plan_;
    cutensorOperationDescriptor_t opDesc_;
    cutensorComputeDescriptor_t computeDesc_;
    cutensorOperator_t opA_, opB_;
    std::string einsum_expr_;
    int32_t numModesA_, numModesB_, numModesC_;
    std::vector<int32_t> modeA_, modeB_, modeC_;
    std::vector<int64_t> extentA_, extentB_, extentC_;
    std::vector<int64_t> strideA_, strideB_, strideC_;
    uint64_t workspaceSize_;
    bool isInitialized_;
};

ContractionPlan create_contraction_plan(std::string const& einsum_expr,
                                         Tensor const& tensorA,
                                         Tensor const& tensorB,
                                         std::string const& compute_type = "",
                                         bool conjA = false,
                                         bool conjB = false);

cutensorStatus_t contract(ContractionPlan& plan,
                          Tensor const& tensorA,
                          Tensor const& tensorB,
                          Tensor& tensorC);

#endif // CONTRACTION_H
