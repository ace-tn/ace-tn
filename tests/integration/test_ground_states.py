import pytest
from acetn.ipeps import Ipeps
import toml
from pathlib import Path
import csv

def test_ipeps_ground_state_energy():
    file_path = str(Path(__file__).parent)
    input_cases = ["ising_dims_2_20_dtau_001_hx_295",
                   "heisenberg_dims_3_16_dtau_001"]

    energy_dict = {}
    with open(file_path + "/ipeps_gs/energies.csv", newline='', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input_case = row[0]
            energy = float(row[1])
            energy_dict[input_case] = energy

    for input_case in input_cases:

        ipeps_config = toml.load(file_path + "/input/" + input_case + ".toml")
        ipeps = Ipeps(ipeps_config)
        ipeps.load(file_path + "/ipeps_gs/" + input_case + ".pt")
        assert ipeps.site_states_initialized, "site_states_initialized not true after loading tensors"

        # Check that measured energy matches
        measured_energy = ipeps.measure()['Energy']
        converged_energy = energy_dict[input_case]
        assert measured_energy.item() == pytest.approx(converged_energy, rel=1e-10)

        # Check that measured energy does not change after evolution
        ipeps.evolve(dtau=0.01, steps=10)
        measured_energy = ipeps.measure()['Energy']
        assert measured_energy.item() == pytest.approx(converged_energy, rel=1e-4)
