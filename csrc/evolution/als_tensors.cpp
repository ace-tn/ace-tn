#include <torch/extension.h>
#include "linalg/contraction.h"
#include "als_tensors.h"
#include <memory>
#include <tuple>
#include <optional>
#include <string>

using torch::Tensor;

inline std::string scalar_type_to_string(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return "float";
        case torch::kFloat64: return "double";
        case torch::kComplexFloat: return "complex64";
        case torch::kComplexDouble: return "complex128";
        default: return "double";
    }
}

AlsTensors::AlsTensors(Tensor a1r, Tensor a2r, Tensor n12g, Tensor n12, Tensor a12g,
                       std::optional<torch::ScalarType> data_type,
                       std::optional<torch::ScalarType> compute_type,
                       std::optional<torch::Device> device)
    : a1r_(a1r), a2r_(a2r), n12g_(n12g), n12_(n12), a12g_(a12g),
      data_type_(data_type.value_or(a1r.scalar_type())),
      compute_type_(scalar_type_to_string(compute_type.value_or(data_type_))),
      device_(device.value_or(a1r.device())) {

    nD_ = a1r.size(0);
    bD_ = a1r.size(1);
    pD_ = a1r.size(2);

    auto tensor_options = torch::TensorOptions().dtype(data_type_).device(device_);

    // ALS tensors
    S_ = torch::empty({nD_, bD_, pD_}, tensor_options);
    R_ = torch::empty({nD_, bD_, nD_, bD_}, tensor_options);

    // intermediate buffers
    R_intm_ = torch::empty({nD_, nD_, nD_, bD_, pD_}, tensor_options);
    a12n_   = torch::empty({nD_, nD_, pD_, pD_}, tensor_options);
    d_intm_ = torch::empty({nD_, nD_, pD_, pD_}, tensor_options);
    d2_     = torch::empty({}, tensor_options);
    d3_     = torch::empty({}, tensor_options);

    // contraction plans
    std::string data_type_str = scalar_type_to_string(data_type_);
    auto add_plan = [&](const char* key, const char* eq, const Tensor& A, const Tensor& B) {
      plan_cache_[key] = std::make_unique<ContractionPlan>(create_contraction_plan(eq, A, B, data_type_str, compute_type_));
    };
    add_plan("S1", "YXpQ,XUQ->YUp", n12g_, a2r_);
    add_plan("S2", "YXpQ,YVP->XVq", n12g_, a1r_);
    add_plan("R1_intm", "yYxX,xUq->yYxXUq", n12_, a2r_);
    add_plan("R1", "yYXUQ,XUQ->yYUy", R_intm_, a2r_);
    add_plan("R2_intm", "yxYX,yVp->yxYXVp", n12_, a1r_);
    add_plan("R2", "xYXVp,YVP->xYVx", R_intm_, a1r_);
    add_plan("a12n", "yup,xuq->yxpq", a1r_, a2r_);
    add_plan("d_intm", "yYxX,yxpq->YXpq", n12_, a12n_);
    add_plan("d", "YXpq,YXpq->", d_intm_, a12n_);
}

AlsTensors::~AlsTensors() {
    plan_cache_.clear();
}

void AlsTensors::set_a1r(Tensor a1r) { a1r_ = a1r; }
void AlsTensors::set_a2r(Tensor a2r) { a2r_ = a2r; }

Tensor const& AlsTensors::build_S1() {
    execute_contraction(*plan_cache_["S1"], n12g_, a2r_, S_);
    return S_;
}

Tensor const& AlsTensors::build_S2() {
    execute_contraction(*plan_cache_["S2"], n12g_, a1r_, S_);
    return S_;
}

Tensor const& AlsTensors::build_R1() {
    execute_contraction(*plan_cache_["R1_intm"], n12_, a2r_, R_intm_);
    execute_contraction(*plan_cache_["R1"], R_intm_, a2r_, R_);
    return R_;
}

Tensor const& AlsTensors::build_R2() {
    execute_contraction(*plan_cache_["R2_intm"], n12_, a1r_, R_intm_);
    execute_contraction(*plan_cache_["R2"], R_intm_, a1r_, R_);
    return R_;
}

double AlsTensors::calculate_cost() {
    execute_contraction(*plan_cache_["a12n"], a1r_, a2r_, a12n_);
    execute_contraction(*plan_cache_["d_intm"], n12_, a12n_, d_intm_);
    execute_contraction(*plan_cache_["d"], d_intm_, a12n_, d2_);
    execute_contraction(*plan_cache_["d_intm"], n12_, a12g_, d_intm_);
    execute_contraction(*plan_cache_["d"], d_intm_, a12n_, d3_);
    Tensor result = d2_ - 2*d3_;
    if (result.is_complex()) {
        // For complex tensors, take the real part
        Tensor real_part = torch::real(result);
        return real_part.item<double>();
    } else {
        return result.item<double>();
    }
}
