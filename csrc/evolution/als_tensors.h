#ifndef ALS_TENSORS_H
#define ALS_TENSORS_H

#include <torch/extension.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <string>

class ContractionPlan;

class AlsTensors {
public:
    AlsTensors(torch::Tensor a1r, torch::Tensor a2r, torch::Tensor n12g, torch::Tensor n12, torch::Tensor a12g,
               std::optional<torch::ScalarType> data_type = std::nullopt,
               std::optional<torch::ScalarType> compute_type = std::nullopt,
               std::optional<torch::Device> device = std::nullopt);
    
    ~AlsTensors();

    void set_a1r(torch::Tensor a1r);
    void set_a2r(torch::Tensor a2r);
    
    torch::Tensor get_a1r() const { return a1r_; }
    torch::Tensor get_a2r() const { return a2r_; }
    
    torch::Tensor const& build_S1();
    torch::Tensor const& build_S2();
    torch::Tensor const& build_R1();
    torch::Tensor const& build_R2();
    double calculate_cost();

private:
    int64_t nD_;
    int64_t bD_;
    int64_t pD_;

    torch::ScalarType data_type_;
    std::string compute_type_;
    torch::Device device_;

    torch::Tensor a1r_;
    torch::Tensor a2r_;
    torch::Tensor n12g_;
    torch::Tensor n12_;
    torch::Tensor a12g_;

    // ALS tensors
    torch::Tensor S_;
    torch::Tensor R_;

    // intermediate buffers
    torch::Tensor R_intm_;
    torch::Tensor a12n_;
    torch::Tensor d_intm_;
    torch::Tensor d2_;
    torch::Tensor d3_;

    // contraction plans
    std::unordered_map<std::string, std::unique_ptr<ContractionPlan>> plan_cache_;
};

#endif // ALS_TENSORS_H

