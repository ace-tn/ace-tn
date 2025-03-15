import torch
from acetn.ipeps.ipeps import Ipeps
from acetn.renormalization.projectors import *
from acetn.utils.benchmarking import record_runtime_ave
from print_device_info import print_device_info
import sys

print_device_info()

@record_runtime_ave
def record_calculate_projectors(projector_calculator, ipeps, sites, k=0):
    projector_calculator.calculate(ipeps, sites, k)

@record_runtime_ave
def record_contraction_half_system(q1, q4):
    return torch.einsum("rRcdDf,dDfsSe->rRcsSe", q1, q4)

@record_runtime_ave
def record_contraction_full_system(q1, q2, q3, q4):
    r1 = einsum("xXclLf,lLfdDe->xXcdDe", q2, q1)
    r2 = einsum("uUclLf,lLfyYe->uUcyYe", q4, q3)
    return einsum("xXcdDf,dDfyYe->xXcyYe", r1, r2)

@record_runtime_ave
def record_rsvd(r1, cD):
    return torch.svd_lowrank(r1, q=cD + 2, niter=2)

@record_runtime_ave
def record_svd(r1):
    return torch.linalg.svd(r1)

def benchmark_calculate_projectors(projector_calculator, ipeps, sites, k=0):
    print("Start benchmarking calculate projectors...")
    runtime_ave = record_calculate_projectors(projector_calculator, ipeps, sites, k)
    print("Done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return runtime_ave

def benchmark_contraction_half_system(q1, q4):
    print("Start benchmarking half-system contraction...")
    _,runtime_ave = record_contraction_half_system(q1, q4)
    print("Done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return runtime_ave

def benchmark_contraction_full_system(q1, q2, q3, q4):
    print("Start benchmarking full-system contraction...")
    r1,runtime_ave = record_contraction_full_system(q1, q2, q3, q4)
    print("Done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return r1,runtime_ave

def benchmark_rsvd(r1, cD):
    print("Start benchmarking rsvd...")
    _,_,_,runtime_ave = record_rsvd(r1, cD)
    print("Done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return runtime_ave

def benchmark_svd(r1):
    print("Start benchmarking full-rank svd...")
    _,_,_,runtime_ave = record_svd(r1)
    print("Done benchmarking")
    print(f"Average runtime: {runtime_ave:.7f} s")
    return runtime_ave


def main(ipeps_config):
    ipeps = Ipeps(ipeps_config)

    sites = [(0, 0),(1, 0),(1, 1),(0, 1)]
    projector_calculator = ProjectorCalculator(ipeps.config.ctmrg)
    calculate_projectors_runtime_ave = benchmark_calculate_projectors(projector_calculator, ipeps, sites, k=0)

    bD = ipeps.dims['bond']
    cD = ipeps.dims['chi']
    q1 = torch.rand(bD, bD, cD, bD, bD, cD, dtype=dtype, device=device)
    q4 = torch.rand_like(q1)

    contraction_half_system_runtime_ave = benchmark_contraction_half_system(q1, q4)
    contraction_half_system_fraction = contraction_half_system_runtime_ave / calculate_projectors_runtime_ave
    print(f"Fraction of calculate_projectors runtime: {contraction_half_system_fraction:.7f}")

    q2 = torch.rand_like(q1)
    q3 = torch.rand_like(q1)
    r1,contraction_full_system_runtime_ave = benchmark_contraction_full_system(q1, q2, q3, q4)
    contraction_full_system_fraction = contraction_full_system_runtime_ave / calculate_projectors_runtime_ave
    print(f"Fraction of calculate_projectors runtime: {contraction_full_system_fraction:.7f}")

    rD = r1.shape
    r1 = r1.reshape(rD[0] * rD[1] * rD[2], rD[3] * rD[4] * rD[5])

    rsvd_runtime_ave = benchmark_rsvd(r1, cD)
    rsvd_fraction = rsvd_runtime_ave / calculate_projectors_runtime_ave
    print(f"Fraction of calculate_projectors runtime: {rsvd_runtime_ave / calculate_projectors_runtime_ave:.7f}")

    if compare_full_rank_svd:
        svd_runtime_ave = benchmark_svd(r1)
        svd_fraction = svd_runtime_ave / calculate_projectors_runtime_ave
        print(f"Fraction of calculate_projectors runtime: {svd_runtime_ave / calculate_projectors_runtime_ave:.7f}")

    print("\nRuntime Summary:")
    print(f"Calculate projectors ({projector_type}):\t {calculate_projectors_runtime_ave:.7f} s")
    print(f"O(D^12) contractions (half-system):\t {contraction_half_system_runtime_ave:.7f} s\t {100 * contraction_half_system_fraction} %")
    print(f"O(D^12) contractions (full-system):\t {contraction_full_system_runtime_ave:.7f} s\t {100 * contraction_full_system_fraction} %")
    print(f"Randomized SVD:\t\t\t\t {rsvd_runtime_ave:.7f} s\t {100 * rsvd_fraction} %")
    if compare_full_rank_svd:
        print(f"Full-rank SVD:\t\t\t\t {svd_runtime_ave:.7f} s\t {100 * svd_fraction} %")


if __name__ == "__main__":
    dims = {}
    dims['phys'] = 2
    dims['bond'] = 6
    dims['chi'] = 36

    if len(sys.argv) == 2:
        dims['bond'] = int(sys.argv[1])
        dims['chi'] = dims['bond']**2
    elif len(sys.argv) == 3:
        dims['bond'] = int(sys.argv[1])
        dims['chi'] = int(sys.argv[2])

    dtype = torch.float64
    device = torch.device('cpu')

    projector_type = 'full-system'

    compare_full_rank_svd = False

    ipeps_config = {
        'dtype': dtype,
        'device': device,
        'TN': {
            'nx': 2,
            'ny': 2,
            'dims': dims,
        },
        'ctmrg': {
            'projectors': projector_type,
        },
    }

    main(ipeps_config)
