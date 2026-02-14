"""
Compare simple update vs full update energies for the transverse field Ising model.

Runs 1000 simple update steps for each of 10 field values, then 100 full update
steps for each. Compares the resulting Energy, sx, and sz measurements.
"""
from acetn.ipeps import Ipeps


def run_evolution(hx, update_type, steps, bond=2, chi=20, dtau=0.01):
    """Run imaginary time evolution and return measurements."""
    config = {
        'dtype': "float64",
        'device': "cpu",
        'TN': {
            'nx': 2,
            'ny': 2,
            'dims': {'phys': 2, 'bond': bond, 'chi': chi},
        },
        'evolution': {
            'update_type': update_type,
        },
        'model': {
            'name': 'ising',
            'params': {
                'jz': 1.0,
                'hx': hx,
            },
        },
    }

    ipeps = Ipeps(config)
    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.evolve(dtau=dtau, steps=steps)
    return ipeps.measure()


def main():
    hx_values = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]
    observables = ['Energy', 'sx', 'sz']

    # Run all simple updates first
    print("Running simple updates (1000 steps each)...")
    simple_results = {}
    for hx in hx_values:
        simple_results[hx] = run_evolution(hx, "simple", steps=1000)

    # Run all full updates
    print("\nRunning full updates (100 steps each)...")
    full_results = {}
    for hx in hx_values:
        full_results[hx] = run_evolution(hx, "full", steps=100)

    # Print comparison table
    header = f"{'hx':>6s}"
    for obs in observables:
        header += f"  {obs+'_s':>12s}  {obs+'_f':>12s}  {'rel_diff':>12s}"
    print("\n" + header)
    print("-" * len(header))

    for hx in hx_values:
        row = f"{hx:6.2f}"
        for obs in observables:
            vs = float(simple_results[hx][obs])
            vf = float(full_results[hx][obs])
            rel_diff = abs(vs - vf) / abs(vf) if abs(vf) > 1e-12 else abs(vs - vf)
            row += f"  {vs:12.6f}  {vf:12.6f}  {rel_diff:12.6e}"
        print(row)


if __name__ == '__main__':
    main()
