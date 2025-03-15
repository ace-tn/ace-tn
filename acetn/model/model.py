from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for quantum models.

    This class provides the basic structure for a quantum model, including methods
    to access one-site and two-site Hamiltonians, observables, and bond directions. 
    It must be extended by specific models like IsingModel or HeisenbergModel.

    Attributes:
    -----------
    config : dict
        A dictionary containing the model configuration (name, params, dtype, device, dim).
    name : str
        The name of the model (e.g., "Ising", "Heisenberg").
    params : dict
        A dictionary containing the model parameters (e.g., interaction strengths).
    dtype : torch.dtype
        The data type of the model (e.g., torch.float32).
    device : torch.device
        The device where the model will run (e.g., CPU or GPU).
    dim : dict
        A dictionary containing the model dimensions (physical dimension).

    Methods:
    --------
    bond_direction(bond):
        Returns the direction associated with a given bond.
    one_site_observables(site=None):
        Returns a dictionary of one-site observables for the given site.
    two_site_observables(bond=None):
        Returns a dictionary of two-site observables for the given bond.
    one_site_hamiltonian(site=None):
        Returns the one-site Hamiltonian for the given site.
    two_site_hamiltonian(bond=None):
        Abstract method for calculating the two-site Hamiltonian for the given bond.
    """
    def __init__(self, config):
        """
        Initializes a quantum model with the provided configuration.

        This constructor is used to initialize the model with its configuration, which
        includes the model's name, parameters, data type (dtype), device (e.g., CPU or GPU),
        and dimensions (dim). The model's attributes are set based on the values provided
        in the configuration dictionary.

        Parameters:
        -----------
        config : dict
            A dictionary containing the model configuration. It must contain the following keys:
            - 'name': The name of the model (e.g., 'Ising', 'Heisenberg').
            - 'params': A dictionary of model parameters (e.g., coupling constants).
            - 'dtype': The data type for the model (e.g., torch.float32).
            - 'device': The device where the model will be computed (e.g., 'cpu' or 'cuda').
            - 'dim': A dictionary containing the dimensions of the model (physical dimension).
        """
        self.config = config
        self.name   = config.name
        self.params = config.params
        self.dtype  = config.dtype
        self.device = config.device
        self.dim    = config.dim

    @staticmethod
    def bond_direction(bond):
        """
        Determines the direction of a given bond.

        Parameters:
        -----------
        bond : list
            A list representing the bond, where the third element indicates the bond direction.

        Returns:
        --------
        str
            The bond direction ('-x', '+y', '+x', '-y') or None if the bond direction is unrecognized.
        """
        _,_,k = bond
        directions = {0:'-x', 1:'+y', 2:'+x', 3:'-y'}
        return directions.get(k, None)

    def initial_site_state(self, site):
        """
        Determines the site tensor state used to initialize a tensor-network product state.

        Parameters:
        -----------
        site : tuple
            The site for which to return the one-site Hamiltonian (default: None).

        Returns:
        --------
        list
            The state of a site tensor.
        """
        return [1,] + [0,]*(self.dim - 1)

    def one_site_observables(self, site=None):
        """
        Returns a dictionary of one-site observables for a given site.

        This is a placeholder method intended to be overridden in subclasses.

        Parameters:
        -----------
        site : optional
            The site for which to return the one-site observables (default: None).

        Returns:
        --------
        dict
            A dictionary of one-site observables.
        """
        return {}

    def two_site_observables(self, bond=None):
        """
        Returns a dictionary of two-site observables for a given bond.

        This is a placeholder method intended to be overridden in subclasses.

        Parameters:
        -----------
        bond : optional
            The bond for which to return the two-site observables (default: None).

        Returns:
        --------
        dict
            A dictionary of two-site observables.
        """
        return {}

    def one_site_hamiltonian(self, site=None):
        """
        Returns the one-site Hamiltonian for a given site.

        This is a placeholder method intended to be overridden in subclasses.

        Parameters:
        -----------
        site : optional
            The site for which to return the one-site Hamiltonian (default: None).

        Returns:
        --------
        torch.Tensor or None
            The one-site Hamiltonian as a tensor, or None if not defined.
        """
        return None

    @abstractmethod
    def two_site_hamiltonian(self, bond=None):
        """
        Abstract method to compute the two-site Hamiltonian for a given bond.

        This method must be implemented by subclasses.

        Parameters:
        -----------
        bond : optional
            The bond for which to return the two-site Hamiltonian (default: None).

        Returns:
        --------
        torch.Tensor
            The two-site Hamiltonian as a tensor.
        """
        pass
