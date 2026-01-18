#include <torch/extension.h>
#include "als_tensors.h"
#include "linalg/cholesky_solve.h"
#include <cmath>
#include <string>
#include <tuple>

using torch::Tensor;

enum class SolverMethod {
    Cholesky,
    Pinv
};

inline SolverMethod parse_solver_method(const std::string& method) {
    if (method == "cholesky") {
        return SolverMethod::Cholesky;
    } else if (method == "pinv") {
        return SolverMethod::Pinv;
    } else {
        throw std::runtime_error("Unknown solver method: " + method);
    }
}

template<typename scalar_t>
inline Tensor regularize_tensor(Tensor A, scalar_t epsilon) {
    scalar_t max_val = A.abs().max().item<scalar_t>();
    auto tensor_options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    A += epsilon * max_val * torch::eye(A.size(0), tensor_options);
    return A;
}

template<typename scalar_t>
Tensor solve_ar(Tensor R, Tensor S, SolverMethod method, scalar_t epsilon) {
    int64_t nD = S.size(0);
    int64_t bD = S.size(1);
    int64_t pD = S.size(2);
    
    S = S.reshape({nD * bD, pD});
    R = R.reshape({nD * bD, nD * bD});
    R = 0.5 * (R + R.mH());
    switch (method) {
        case SolverMethod::Cholesky: {
            R = regularize_tensor<scalar_t>(R, epsilon);
            return cusolver_ext::cholesky_solve(R, S).reshape({nD, bD, pD});
        }
        case SolverMethod::Pinv: {
            auto R_inv = torch::linalg_pinv(R, epsilon, true);
            return (R_inv.matmul(S)).reshape({nD, bD, pD});
        }
    }
    __builtin_unreachable();
}

template<typename scalar_t>
struct AlsUpdate {
public:
    AlsUpdate(scalar_t cost_initial, SolverMethod method, scalar_t epsilon) : 
              cost_(cost_initial), method_(method), epsilon_(epsilon) {}

    scalar_t operator()(AlsTensors<scalar_t>& als_tensors) {
        Tensor const& S1 = als_tensors.build_S1();
        Tensor const& R1 = als_tensors.build_R1();
        Tensor a1r = solve_ar<scalar_t>(R1, S1, method_, epsilon_);
        als_tensors.set_a1r(a1r);

        Tensor const& S2 = als_tensors.build_S2();
        Tensor const& R2 = als_tensors.build_R2();
        Tensor a2r = solve_ar<scalar_t>(R2, S2, method_, epsilon_);
        als_tensors.set_a2r(a2r);

        scalar_t cost_new = als_tensors.calculate_cost();
        scalar_t error = std::abs(cost_new - cost_) / std::abs(cost_);
        cost_ = cost_new;
        return error;
    }

private:
    scalar_t cost_;
    SolverMethod method_;
    scalar_t epsilon_;
};

template<typename scalar_t>
std::tuple<Tensor, Tensor> als_solve_impl(
    Tensor a1r,
    Tensor a2r,
    Tensor n12g,
    Tensor n12,
    Tensor a12g,
    int64_t niter,
    scalar_t tol,
    SolverMethod method,
    scalar_t epsilon) {

    AlsTensors<scalar_t> als_tensors(a1r, a2r, n12g, n12, a12g);
    scalar_t cost_initial = als_tensors.calculate_cost();
    AlsUpdate<scalar_t> als_update(cost_initial, method, epsilon);

    for (int64_t i = 0; i < niter; ++i) {
        scalar_t error = als_update(als_tensors);
        if (i > 1 && error < tol) { break; }
    }
    return std::make_tuple(als_tensors.get_a1r(), als_tensors.get_a2r());
}

std::tuple<Tensor, Tensor> als_solve(
    Tensor a1r,
    Tensor a2r,
    Tensor n12g,
    Tensor n12,
    Tensor a12g,
    int64_t niter,
    double tol,
    const std::string& method,
    double epsilon) {

    // Ensure contiguous memory layout for cuTENSOR
    a1r = a1r.contiguous();
    a2r = a2r.contiguous();
    n12g = n12g.contiguous();
    n12 = n12.contiguous();
    a12g = a12g.contiguous();

    SolverMethod solver_method = parse_solver_method(method);
    auto real_dtype = a1r.is_complex() ? c10::toRealValueType(a1r.scalar_type()) : a1r.scalar_type();
    if (real_dtype == torch::kFloat32) {
        return als_solve_impl<float>(a1r, a2r, n12g, n12, a12g, niter, static_cast<float>(tol), solver_method, static_cast<float>(epsilon));
    } else {
        return als_solve_impl<double>(a1r, a2r, n12g, n12, a12g, niter, tol, solver_method, epsilon);
    }
}

PYBIND11_MODULE(_C_cutensor, m) {
    m.doc() = "cuTENSOR accelerated ALS solver for acetn";
    m.def("als_solve", &als_solve, "ALS solver using cuTENSOR contractions and cuSOLVER");
}
