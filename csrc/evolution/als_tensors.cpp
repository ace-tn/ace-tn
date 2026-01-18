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

template<typename scalar_t>
inline scalar_t get_real_scalar(const Tensor& t) {
    if (t.is_complex()) {
        return torch::real(t).item<scalar_t>();
    }
    return t.item<scalar_t>();
}

template<typename scalar_t>
AlsTensors<scalar_t>::AlsTensors(Tensor a1r, Tensor a2r, Tensor n12g, Tensor n12, Tensor a12g,
                       std::optional<torch::ScalarType> data_type,
                       std::optional<torch::ScalarType> compute_type,
                       std::optional<torch::Device> device)
    : nD_(a1r.size(0)), bD_(a1r.size(1)), pD_(a1r.size(2)),
      data_type_(data_type.value_or(a1r.scalar_type())),
      compute_type_(scalar_type_to_string(compute_type.value_or(data_type_))),
      device_(device.value_or(a1r.device())),
      a1r_(a1r), a2r_(a2r), n12g_(n12g), n12_(n12), a12g_(a12g) {

    auto tensor_options = torch::TensorOptions().dtype(data_type_).device(device_);

    S_ = torch::empty({nD_, bD_, pD_}, tensor_options);
    R_ = torch::empty({nD_, bD_, nD_, bD_}, tensor_options);
    R_intm_ = torch::empty({nD_, nD_, nD_, bD_, pD_}, tensor_options);
    a12n_   = torch::empty({nD_, nD_, pD_, pD_}, tensor_options);
    d_intm_ = torch::empty({nD_, nD_, pD_, pD_}, tensor_options);
    d2_     = torch::empty({}, tensor_options);
    d3_     = torch::empty({}, tensor_options);

    auto add_plan = [&](const char* key, const char* eq, const Tensor& A, const Tensor& B,
                        bool conjA = false, bool conjB = false) {
      plan_cache_[key] = create_contraction_plan(eq, A, B, compute_type_, conjA, conjB);
    };
    add_plan("S1", "YXpQ,XUQ->YUp", n12g_, a2r_, false, true);
    add_plan("S2", "YXPq,YVP->XVq", n12g_, a1r_, false, true);
    add_plan("R1_intm", "yxYX,xuq->yYXuq", n12_, a2r_);
    add_plan("R1", "yYXuQ,XUQ->YUyu", R_intm_, a2r_, false, true);
    add_plan("R2_intm", "yxYX,yvp->xYXvp", n12_, a1r_);
    add_plan("R2", "xYXvP,YVP->XVxv", R_intm_, a1r_, false, true);
    add_plan("a12n", "yup,xuq->yxpq", a1r_, a2r_);
    add_plan("d_intm", "yxYX,yxpq->YXpq", n12_, a12n_);
    add_plan("d", "YXpq,YXpq->", d_intm_, a12n_, false, true);
}

template<typename scalar_t>
AlsTensors<scalar_t>::~AlsTensors() {
    plan_cache_.clear();
}

template<typename scalar_t>
void AlsTensors<scalar_t>::set_a1r(Tensor a1r) { a1r_ = a1r; }

template<typename scalar_t>
void AlsTensors<scalar_t>::set_a2r(Tensor a2r) { a2r_ = a2r; }

template<typename scalar_t>
Tensor const& AlsTensors<scalar_t>::build_S1() {
    contract(plan_cache_["S1"], n12g_, a2r_, S_);
    return S_;
}

template<typename scalar_t>
Tensor const& AlsTensors<scalar_t>::build_S2() {
    contract(plan_cache_["S2"], n12g_, a1r_, S_);
    return S_;
}

template<typename scalar_t>
Tensor const& AlsTensors<scalar_t>::build_R1() {
    contract(plan_cache_["R1_intm"], n12_, a2r_, R_intm_);
    contract(plan_cache_["R1"], R_intm_, a2r_, R_);
    return R_;
}

template<typename scalar_t>
Tensor const& AlsTensors<scalar_t>::build_R2() {
    contract(plan_cache_["R2_intm"], n12_, a1r_, R_intm_);
    contract(plan_cache_["R2"], R_intm_, a1r_, R_);
    return R_;
}

template<typename scalar_t>
scalar_t AlsTensors<scalar_t>::calculate_cost() {
    contract(plan_cache_["a12n"], a1r_, a2r_, a12n_);

    contract(plan_cache_["d_intm"], n12_, a12n_, d_intm_);
    contract(plan_cache_["d"], d_intm_, a12n_, d2_);

    contract(plan_cache_["d_intm"], n12_, a12g_, d_intm_);
    contract(plan_cache_["d"], d_intm_, a12n_, d3_);

    scalar_t d2_real = get_real_scalar<scalar_t>(d2_);
    scalar_t d3_real = get_real_scalar<scalar_t>(d3_);
    return d2_real - scalar_t(2.0) * d3_real;
}

template class AlsTensors<float>;
template class AlsTensors<double>;
