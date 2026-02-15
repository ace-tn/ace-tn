import torch


class Gate:
    """
    A class representing quantum gates for one-site and two-site Hamiltonians, 
    used in time evolution algorithms.
    """
    _gate = {}
    def __init__(self, model, dtau, bond_list=None, site_list=None):
        """
        Initializes the Gate object, creating gates for the given model.

        Parameters:
        -----------
        model : object
            The model that defines the Hamiltonians and other parameters.
        dtau : float
            The imaginary-time step for the gate calculation.
        bond_list : list, optional
            A list of bonds for which two-site gates will be created (default is None).
        site_list : list, optional
            A list of sites for which one-site gates will be created (default is None).
        """
        self.dtype = model.dtype
        self.device = model.device
        self.wrap_one_site = True
        if site_list is not None:
            self.build_one_site_gates(model, dtau, site_list)
        if bond_list is not None:
            self.build_two_site_gates(model, dtau, bond_list)

    def __getitem__(self, key):
        """
        Retrieves the gate corresponding to a site or bond.

        Parameters:
        -----------
        key : tuple or Bond
            A tuple representing a site or a Bond representing a bond.

        Returns:
        --------
        torch.Tensor
            The gate matrix for the given key (site or bond).
        """
        return self._gate[key]

    def __setitem__(self, key, gate):
        """
        Sets the gate for a given site or bond.

        Parameters:
        -----------
        key : tuple or Bond
            A tuple representing a site or a Bond representing a bond.
        gate : torch.Tensor
            The gate matrix to store for the given site or bond.
        """
        self._gate[key] = gate

    def build_one_site_gates(self, model, dtau, site_list):
        """
        Builds the one-site gates for the given list of sites by calculating the exponential of
        the Hamiltonian for each site.

        Parameters:
        -----------
        model : object
            The model that defines the Hamiltonian for the site.
        dtau : float
            The imaginary-time step for the gate calculation.
        site_list : list
            A list of sites for which one-site gates will be created.
        """
        for site in site_list:
            site_ham = model.one_site_hamiltonian(site)
            if site_ham is not None:
                scale = 1. if self.wrap_one_site else 0.25
                self[site] = self.calculate_gate(site_ham.mat, scale*dtau)

    def build_two_site_gates(self, model, dtau, bond_list):
        """
        Builds the two-site gates for the given list of bonds by calculating the exponential of
        the Hamiltonian for each bond and adding contributions from one-site Hamiltonians if necessary.

        Parameters:
        -----------
        model : object
            The model that defines the Hamiltonians for the bonds and sites.
        dtau : float
            The imaginary-time step for the gate calculation.
        bond_list : list
            A list of bonds for which two-site gates will be created.
        """
        for bond in bond_list:
            pD = model.dim
            bond_ham = model.two_site_hamiltonian(bond).mat
            if self.wrap_one_site:
                id_mat = torch.eye(pD, dtype=self.dtype, device=self.device)
                site_ham_1 = model.one_site_hamiltonian(bond.s1)
                site_ham_2 = model.one_site_hamiltonian(bond.s2)
                if site_ham_1 is not None and site_ham_2 is not None:
                    site_ham_1 = torch.kron(site_ham_1.mat, id_mat)
                    site_ham_2 = torch.kron(id_mat, site_ham_2.mat)
                    bond_ham += 0.25*(site_ham_1 + site_ham_2)
            self[bond] = self.calculate_gate(bond_ham, dtau).reshape(pD,pD,pD,pD)

    def calculate_gate(self, ham, dtau):
        """
        Calculates the gate from the Hamiltonian using the matrix exponential and time step.

        Parameters:
        -----------
        ham : torch.Tensor
            The Hamiltonian matrix for which the gate will be calculated.
        dtau : float
            The imaginary-time step for the gate calculation.

        Returns:
        --------
        torch.Tensor
            The calculated gate for the given Hamiltonian.
        """
        return torch.matrix_exp(-dtau * ham).to(dtype=self.dtype, device=self.device)
