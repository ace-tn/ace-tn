#include <torch/extension.h>
#include "als_tensors.h"
#include "linalg/cholesky_solve.h"
#include <cmath>
#include <string>
#include <tuple>

using torch::Tensor;

inline Tensor regularize_tensor(Tensor A, double epsilon) {
    double max_val = A.abs().max().item<double>();
    auto tensor_options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    A += epsilon * max_val * torch::eye(A.size(0), tensor_options);
    return A;
}

Tensor solve_ar(Tensor R, Tensor S, const std::string& method, double epsilon) {
    int64_t nD = S.size(0);
    int64_t bD = S.size(1);
    int64_t pD = S.size(2);
    
    S = S.reshape({nD * bD, pD});
    R = R.reshape({nD * bD, nD * bD});
    R = 0.5 * (R + R.conj());
    if (method == "cholesky") {
        R = regularize_tensor(R, epsilon);
        return cusolver_ext::cholesky_solve(R, S).reshape({nD, bD, pD});
    } else if (method == "pinv") {
        auto R_inv = torch::linalg_pinv(R, epsilon, true);
        return (R_inv.matmul(S)).reshape({nD, bD, pD});
    } else {
        throw std::runtime_error("Unknown method: " + method);
    }
}

struct AlsUpdate {
public:
    AlsUpdate(double cost_initial, std::string method, double epsilon) : 
              cost_(cost_initial), method_(method), epsilon_(epsilon) {}

    double operator()(AlsTensors& als_tensors) {
        Tensor const& S1 = als_tensors.build_S1();
        Tensor const& R1 = als_tensors.build_R1();
        Tensor a1r = solve_ar(R1, S1, method_, epsilon_);
        als_tensors.set_a1r(a1r);

        Tensor const& S2 = als_tensors.build_S2();
        Tensor const& R2 = als_tensors.build_R2();
        Tensor a2r = solve_ar(R2, S2, method_, epsilon_);
        als_tensors.set_a2r(a2r);

        double cost_new = als_tensors.calculate_cost();
        double error = std::abs(cost_new - cost_) / std::abs(cost_);
        cost_ = cost_new;
        return error;
    }

private:
    double cost_;
    std::string method_;
    double epsilon_;
};

std::tuple<Tensor, Tensor> als_solve(
    Tensor a1r,
    Tensor a2r,
    Tensor n12g,
    Tensor n12,
    Tensor a12g,
    int64_t niter,
    double tol,
    std::string method,
    double epsilon) {

    AlsTensors als_tensors(a1r, a2r, n12g, n12, a12g);
    double cost_initial = als_tensors.calculate_cost();
    AlsUpdate als_update(cost_initial, method, epsilon);

    for (int64_t i = 0; i < niter; ++i) {
        double error = als_update(als_tensors);
        if (i > 1 && error < tol) { break; }
    }
    return std::make_tuple(als_tensors.get_a1r(), als_tensors.get_a2r());
}

TORCH_LIBRARY(cutensor_ext, m) {
    m.def("als_solve", &als_solve);
}
