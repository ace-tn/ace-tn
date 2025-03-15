from ipeps.ipeps import Ipeps
from model.model_factory import model_factory
from evolution.full_update import *
from evolution.gate import Gate
from utils.benchmarking import record_runtime_ave
from print_device_info import print_device_info
import sys

print_device_info()

@record_runtime_ave
def record_full_update(full_updater, a1, a2, bond):
    a1,a2,*_ = full_updater.tensor_update(a1, a2, bond)

@record_runtime_ave
def record_decompose_site_tensors(full_updater, a1, a2, bD, pD):
    a1q,a1r,a2q,a2r,nD = full_updater.decompose_site_tensors(a1, a2, bD, pD)
    return a1q,a1r,a2q,a2r,nD

@record_runtime_ave
def record_build_norm_tensor(ipeps, bond, a1q, a2q):
    n12 = build_norm_tensor(ipeps, bond, a1q, a2q)
    return n12

@record_runtime_ave
def record_reduced_full_update(full_updater, a1r, a2r, n12, gate, pD, bD, nD):
    a1r,a2r,*_ = full_updater.update_reduced_tensors(a1r, a2r, n12, gate, pD, bD, nD)

@record_runtime_ave
def record_als(als_solver):
    a1r,a2r,*_ = als_solver.solve()

def benchmark_full_update(full_updater, a1, a2, bond):
    print("start benchmarking full_update...")
    runtime_ave = record_full_update(full_updater, a1, a2, bond)
    print("done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return runtime_ave

def benchmark_decompose_site_tensors(full_updater, a1, a2, bD, pD):
    print("start benchmarking decompose_site_tensors...")
    a1q,a1r,a2q,a2r,nD,runtime_ave = record_decompose_site_tensors(full_updater, a1, a2, bD, pD)
    print("done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return a1q,a1r,a2q,a2r,nD,runtime_ave

def benchmark_build_norm_tensor(ipeps, bond, a1q, a2q):
    print("start benchmarking build_norm_tensor...")
    n12,runtime_ave = record_build_norm_tensor(ipeps, bond, a1q, a2q)
    print("done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return n12,runtime_ave

def benchmark_reduced_full_update(full_updater, a1r, a2r, n12, gate, pD, bD, nD):
    print("start benchmarking reduced_full_update...")
    runtime_ave = record_reduced_full_update(full_updater, a1r, a2r, n12, gate, pD, bD, nD)
    print("done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return runtime_ave

def benchmark_als(als_solver, method):
    print(f"start benchmarking als_sweep ({method})...")
    als_solver.method = method
    runtime_ave = record_als(als_solver)
    print("done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return runtime_ave

def main(ipeps_config):

    ipeps = Ipeps(ipeps_config)

    pD = dims['phys']
    bD = dims['bond']

    model = model_factory.create(ipeps.config.model)
    gate = Gate(model, 0.01, ipeps.bond_list, ipeps.site_list)
    full_updater = FullUpdater(ipeps, gate, ipeps.config.evolution)

    bond = ipeps.bond_list[0]
    a1 = ipeps[bond[0]]['A']
    a2 = ipeps[bond[1]]['A']

    full_update_runtime_ave = benchmark_full_update(full_updater, a1, a2, bond)

    a1q,a1r,a2q,a2r,nD,decompose_site_tensors_runtime_ave = \
        benchmark_decompose_site_tensors(full_updater, a1, a2, bD, pD)
    decompose_site_tensors_fraction = decompose_site_tensors_runtime_ave/full_update_runtime_ave
    print(f"Fraction of full update runtime: {decompose_site_tensors_fraction:.7f}")

    n12,build_norm_tensor_runtime_ave = \
        benchmark_build_norm_tensor(ipeps, bond, a1q, a2q)
    norm_build_fraction = build_norm_tensor_runtime_ave/full_update_runtime_ave
    print(f"Fraction of full update runtime: {norm_build_fraction:.7f}")

    reduced_full_update_runtime_ave = \
        benchmark_reduced_full_update(full_updater, a1r, a2r, n12, gate[bond], pD, bD, nD)
    reduced_upd_fraction = reduced_full_update_runtime_ave/full_update_runtime_ave
    print(f"Fraction of full update runtime: {reduced_upd_fraction:.7f}")

    a12g = einsum("yup,xuq->ypxq", a1r, a2r)
    a12g = einsum("ypxq,pqrs->yxrs", a12g, gate[bond])
    nz = positive_approx(n12, nD)
    n12 = einsum("xyz,XYz->xyXY", nz, conj(nz))
    als_solver = ALSSolver(n12, a12g, bD, pD, nD, ipeps.config.evolution)

    als_cholesky_runtime_ave = benchmark_als(als_solver, "cholesky")
    print(f"Fraction of reduced full update runtime: {als_cholesky_runtime_ave/reduced_full_update_runtime_ave:.7f}")
    als_fraction = als_cholesky_runtime_ave/full_update_runtime_ave
    print(f"Fraction of full update runtime: {als_cholesky_runtime_ave/full_update_runtime_ave:.7f}")

    als_pinv_runtime_ave = benchmark_als(als_solver, "pinv")
    print(f"Fraction of reduced full update runtime: {als_pinv_runtime_ave/reduced_full_update_runtime_ave:.7f}")
    als_pinv_fraction = als_pinv_runtime_ave/full_update_runtime_ave
    print(f"Fraction of full update runtime: {als_pinv_runtime_ave/full_update_runtime_ave:.7f}")

    print("\nRuntime Summary:")
    print(f"full update:\t\t {full_update_runtime_ave:.7f} s")
    print(f"decompose site tensors:\t {decompose_site_tensors_runtime_ave:.7f} s\t {100*decompose_site_tensors_fraction} %")
    print(f"build norm tensor:\t {build_norm_tensor_runtime_ave:.7f} s\t {100*norm_build_fraction} %")
    print(f"update reduced tensors:\t {reduced_full_update_runtime_ave:.7f} s\t {100*reduced_upd_fraction} %")
    print(f"ALS solve (Cholesky):\t {als_cholesky_runtime_ave:.7f} s\t {100*als_fraction} %")
    print(f"ALS solve (pinv):\t {als_pinv_runtime_ave:.7f} s\t {100*als_pinv_fraction} %")


if __name__=="__main__":
    dims = {}
    dims['phys'] = 2
    dims['bond'] = 6
    dims['chi'] = dims['bond']**2

    if len(sys.argv) == 2:
        dims['bond'] = int(sys.argv[1])
        dims['chi'] = dims['bond']**2
    elif len(sys.argv) == 3:
        dims['bond'] = int(sys.argv[1])
        dims['chi'] = int(sys.argv[2])

    dtype = torch.float64
    device = torch.device('cpu')

    ipeps_config = {
        'dtype': dtype,
        'device': device,
        'TN':{
            'nx': 2,
            'ny': 2,
            'dims': dims,
        },
        'model':{
            'name':'heisenberg',
            'params':{'J': 1.0,},
        },
        'evolution':{
            'als_niter':10,
            'als_tol':-torch.inf,
        },
    }

    main(ipeps_config)
