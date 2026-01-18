#include "contraction.h"

ContractionPlan create_contraction_plan(std::string const& einsum_expr,
                                         Tensor const& tensorA,
                                         Tensor const& tensorB,
                                         std::string const& compute_type,
                                         bool conjA,
                                         bool conjB) {
  ContractionPlan plan;
  plan.einsum_expr = einsum_expr;
  plan.compute_type = compute_type;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(tensorA.scalar_type(), "create_contraction_plan", [&] {
    cutensorComputeDescriptor_t resolved_compute_desc;
    if (compute_type.empty()) {
      resolved_compute_desc = CuTensorTypeTraits<scalar_t>::getComputeDesc();
    } else {
      resolved_compute_desc = resolveComputeDescriptor(compute_type);
    }

    auto contraction = std::make_unique<Contraction<scalar_t>>(einsum_expr, tensorA, tensorB, resolved_compute_desc, conjA, conjB);

    auto handle = CutensorHandle::get();
    if (!contraction->plan(handle, CUTENSOR_WORKSPACE_DEFAULT)) {
      throw std::runtime_error("cuTENSOR error: Contraction plan failed");
    }

    plan.workspace_size = contraction->getWorkspaceSize();
    if (plan.workspace_size > 0) {
      try {
        plan.workspace = at::empty(plan.workspace_size, at::CUDA(at::kByte));
      } catch (std::exception& e) {
        // Fallback to minimum workspace if default allocation fails
        if (!contraction->plan(handle, CUTENSOR_WORKSPACE_MIN)) {
          throw std::runtime_error("cuTENSOR error: Contraction plan with minimum workspace failed");
        }
        plan.workspace_size = contraction->getWorkspaceSize();
        if (plan.workspace_size > 0) {
          plan.workspace = at::empty(plan.workspace_size, at::CUDA(at::kByte));
        }
      }
    }

    plan.contraction = std::move(contraction);
  });

  return plan;
}

cutensorStatus_t contract(ContractionPlan& plan,
                          Tensor const& tensorA,
                          Tensor const& tensorB,
                          Tensor& tensorC) {
  if (!plan.contraction) {
    throw std::runtime_error("cuTENSOR error: Invalid contraction plan");
  }

  auto handle = CutensorHandle::get();
  auto stream = at::cuda::getCurrentCUDAStream();
  void* workspace = plan.workspace_size > 0 ? plan.workspace.data_ptr<uint8_t>() : nullptr;

  if (!plan.contraction->execute(handle,
                                 tensorA.data_ptr(),
                                 tensorB.data_ptr(),
                                 tensorC.data_ptr(),
                                 workspace,
                                 stream)) {
    throw std::runtime_error("cuTENSOR error: Contraction execution failed");
  }

  return cutensorStatus_t::CUTENSOR_STATUS_SUCCESS;
}
