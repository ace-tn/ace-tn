from torch import einsum
from measurement.rdm import RDM
from tqdm import tqdm

def measure(ipeps, model):
    """
    Perform measurements on the iPEPS system, including both site and bond-level measurements.

    Args:
        ipeps (Ipeps): The iPEPS tensor network object.
        model (Model): The model containing Hamiltonians and observables for the measurements.

    Returns:
        dict: A dictionary containing the averaged measurements, including Energy and other observables.
    """
    measurements = {}
    measurements_ave = {}
    measurements_ave['Energy'] = 0
    rdm = RDM(ipeps)

    calculate_one_site_measurements(ipeps.site_list, model, rdm, measurements, measurements_ave)
    calculate_two_site_measurements(ipeps.bond_list, model, rdm, measurements, measurements_ave)

    measurements_ave['Energy'] /= len(ipeps.site_list)

    tqdm.write(f"Measurement averages:")
    tqdm.write("\n".join(f"{key}: {value}" for key, value in measurements_ave.items()))
    return measurements_ave


def measure_site_energy(site, site_rdm, site_norm, model):
    """
    Measure the energy of a single site in the iPEPS system.

    Args:
        site (tuple): The indices (x,y) of the site.
        site_rdm (Tensor): The reduced density matrix for the site.
        site_norm (float): The normalization factor for the site.
        model (Model): The model containing the one-site Hamiltonian for the measurement.

    Returns:
        float: The energy of the site.
    """
    site_ham = model.one_site_hamiltonian(site)
    if site_ham is not None:
        site_ham = site_ham.mat if hasattr(site_ham, 'mat') else site_ham
        return measure_one_site(site_rdm, site_ham)/site_norm
    else:
        return 0.


def measure_bond_energy(bond, bond_rdm, bond_norm, model):
    """
    Measure the energy of a bond in the iPEPS system.

    Args:
        bond (list): A list representing the bond (e.g., [site1, site2, k]).
        bond_rdm (Tensor): The reduced density matrix for the bond.
        bond_norm (float): The normalization factor for the bond.
        model (Model): The model containing the two-site Hamiltonian for the measurement.

    Returns:
        float: The energy of the bond.
    """
    bond_ham = model.two_site_hamiltonian(bond)
    bond_ham = bond_ham.mat if hasattr(bond_ham, 'mat') else bond_ham
    bond_ham = bond_ham.reshape((model.dim,)*4)
    return measure_two_site(bond_rdm, bond_ham)/bond_norm


def measure_site_observables(site, site_rdm, site_norm, model, measurements, measurements_ave):
    """
    Measure the observables for a single site in the iPEPS system and update the averages.

    Args:
        site (int): The index of the site.
        site_rdm (Tensor): The reduced density matrix for the site.
        site_norm (float): The normalization factor for the site.
        model (Model): The model containing the one-site observables.
        measurements (dict): A dictionary to store the measurements for the site.
        measurements_ave (dict): A dictionary to store the averaged measurements.
    """
    measurement_str = f"{site}"
    for op_name,site_op in model.one_site_observables(site).items():
        site_op = site_op.mat if hasattr(site_op, 'mat') else site_op
        measurements[op_name] = measure_one_site(site_rdm, site_op)/site_norm
        measurements_ave[op_name] += measurements[op_name]
        measurement_str += f"\t{measurements[op_name].item():.8f}"
    tqdm.write(measurement_str)


def measure_bond_observables(bond, bond_rdm, bond_norm, model, measurements, measurements_ave):
    """
    Measure the observables for a bond in the iPEPS system and update the averages.

    Args:
        bond (list): A list representing the bond (e.g., [site1, site2, k]).
        bond_rdm (Tensor): The reduced density matrix for the bond.
        bond_norm (float): The normalization factor for the bond.
        model (Model): The model containing the two-site observables.
        measurements (dict): A dictionary to store the measurements for the bond.
        measurements_ave (dict): A dictionary to store the averaged measurements.
    """
    measurement_str = f"{bond[0]}\t{bond[1]}"
    for op_name,bond_op in model.two_site_observables(bond).items():
        bond_op = bond_op.mat if hasattr(bond_op, 'mat') else bond_op
        bond_op = bond_op.reshape((model.dim,)*4)
        measurements[op_name] = measure_two_site(bond_rdm, bond_op)/bond_norm
        measurements_ave[op_name] += measurements[op_name]
        measurement_str += f"\t{measurements[op_name].item():.8f}"
    tqdm.write(measurement_str)


def calculate_one_site_measurements(site_list, model, rdm, measurements, measurements_ave):
    """
    Calculate the measurements for each site in the iPEPS system, including observables and energy.

    Args:
        site_list (list): A list of site indices in the iPEPS system.
        model (Model): The model containing the one-site observables and Hamiltonian.
        rdm (RDM): The reduced density matrix for the iPEPS system.
        measurements (dict): A dictionary to store the measurements for each site.
        measurements_ave (dict): A dictionary to store the averaged measurements.
    """
    one_site_keys = model.one_site_observables((0,0)).keys()
    if one_site_keys:
        tqdm.write("Calculated one-site measurements:")
        for key in one_site_keys:
            measurements_ave[key] = 0
        tqdm.write("site\t" + "\t".join([key[:10].ljust(10) for key in one_site_keys]))

    for site in site_list:
        site_rdm = rdm[site]
        site_norm = einsum("pp->", site_rdm).real

        measurements_ave['Energy'] += measure_site_energy(site, site_rdm, site_norm, model)
        if one_site_keys:
            measure_site_observables(site, site_rdm, site_norm, model, measurements, measurements_ave)

    for key in one_site_keys:
        measurements_ave[key] /= len(site_list)


def calculate_two_site_measurements(bond_list, model, rdm, measurements, measurements_ave):
    """
    Calculate the measurements for each bond in the iPEPS system, including observables and energy.

    Args:
        bond_list (list): A list of bond indices in the iPEPS system.
        model (Model): The model containing the two-site observables and Hamiltonian.
        rdm (RDM): The reduced density matrix for the iPEPS system.
        measurements (dict): A dictionary to store the measurements for each bond.
        measurements_ave (dict): A dictionary to store the averaged measurements.
    """
    two_site_keys = model.two_site_observables(bond_list[0]).keys()
    if two_site_keys:
        tqdm.write("Calculated two-site measurements:")
        for key in two_site_keys:
            measurements_ave[key] = 0
        tqdm.write("site 1\t" + "site 2\t" + "\t".join([key[:10].ljust(10) for key in two_site_keys]))
    for bond in bond_list:
        bond_rdm = rdm[bond]
        bond_norm = einsum("pqpq->", bond_rdm).real

        measurements_ave['Energy'] += measure_bond_energy(bond, bond_rdm, bond_norm, model)
        if two_site_keys:
            measure_bond_observables(bond, bond_rdm, bond_norm, model, measurements, measurements_ave)

    for key in two_site_keys:
        measurements_ave[key] /= len(bond_list)


def measure_one_site(rdm, operator):
    """
    Measure the expectation value of a one-site operator with respect to a reduced density matrix.

    Args:
        rdm (Tensor): The reduced density matrix for the site.
        operator (Tensor): The one-site operator to measure.
    
    Returns:
        float: The expectation value of the operator.
    """
    return einsum("Pp,pP->", rdm, operator).real


def measure_two_site(rdm, operator):
    """
    Measure the expectation value of a two-site operator with respect to a reduced density matrix.

    Args:
        rdm (Tensor): The reduced density matrix for the two-site bond.
        operator (Tensor): The two-site operator to measure.

    Returns:
        float: The expectation value of the operator.
    """
    return einsum("PQpq,pqPQ->", rdm, operator).real
