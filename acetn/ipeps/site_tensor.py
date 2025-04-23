import torch
from torch import conj

class SiteTensor:
    """
    A class that represents a site tensor with corner and edge tensors in a tensor network.

    The `SiteTensor` class provides methods to initialize, copy, and manipulate the site tensor, 
    corner tensors, and edge tensors. These tensors are used to form the iPEPS tensor network.
    """
    _site_tensor = None
    _corner_tensors = None
    _edge_tensors = None
    def __init__(self, dims, site_tensor=None, site_state=[1., 0.], dtype=torch.float64, device=torch.device("cpu"),):
        """
        Initializes a `SiteTensor` object.

        If `site_tensor` is provided, it copies its tensors with an optional noise factor. 
        Otherwise, it initializes the site tensor, corner tensors, and edge tensors with random values 
        and a given initial state.

        Args:
            dims (dict): A dictionary with dimensions 'bond', 'phys', and 'chi' for bond, physical, and auxiliary dimensions.
            site_tensor (SiteTensor, optional): An existing `SiteTensor` to copy.
            site_state (list, optional): A list specifying the initial state for the site tensor (default is [1., 0.]).
            dtype (torch.dtype, optional): The data type of the tensors (default is torch.float64).
            device (torch.device, optional): The device where the tensors are placed (default is 'cpu').
        """
        self.dims = dims
        self.dtype = dtype
        self.device = device
        self.initial_condition = None

        if site_tensor is None:
            self.initialize_site_tensor(site_state, noise=1e-2)
            self.initialize_corner_tensors()
            self.initialize_edge_tensors()
        else:
            self.copy_from(site_tensor)

    def __getitem__(self, key):
        """
        Accessor method for getting the site tensor, corner tensors, or edge tensors.

        Args:
            key (str): The key specifying which tensor to retrieve. 
                    ('A' for site tensor, 'C' for corner tensors, 'E' for edge tensors).

        Returns:
            torch.Tensor or list: The requested tensor(s).

        Raises:
            ValueError: If an invalid key is provided.
        """
        if key == 'A':
            return self._site_tensor
        elif key == 'C':
            return self._corner_tensors
        elif key == 'E':
            return self._edge_tensors
        else:
            raise ValueError(f"Invalid key: '{key}' provided.")

    def __setitem__(self, key, val):
        """
        Setter method for assigning a new value to the site tensor, corner tensors, or edge tensors.

        Args:
            key (str): The key specifying which tensor to set. 
                    ('A' for site tensor, 'C' for corner tensors, 'E' for edge tensors).
            val (torch.Tensor or list): The new tensor or list of tensors to assign.

        Raises:
            ValueError: If an invalid key is provided.
        """
        if key == 'A':
            self._site_tensor = torch.clone(val).detach().to(dtype=self.dtype, device=self.device)
        elif key == 'C':
            self._corner_tensors = [torch.clone(v).detach().to(dtype=self.dtype, device=self.device) for v in val]
        elif key == 'E':
            self._edge_tensors = [torch.clone(v).detach().to(dtype=self.dtype, device=self.device) for v in val]
        else:
            raise ValueError(f"Invalid key: '{key}' provided.")

    def copy_from(self, site_tensor):
        """
        Copies from the tensors from another `SiteTensor`.

        Args:
            site_tensor (SiteTensor): The `SiteTensor` to copy from.
        """
        self['A'] = site_tensor['A']
        self['C'] = site_tensor['C']
        self['E'] = site_tensor['E']

    def copy(self):
        """
        Returns a deep copy of the current `SiteTensor` instance.

        Returns:
            SiteTensor: A new `SiteTensor` with identical contents and metadata.
        """
        return SiteTensor(dims=self.dims, site_tensor=self, dtype=self.dtype, device=self.device)

    def to(self, dtype=None, device=None):
        """
        Sends the tensors to a device and/or changes the data type.

        Args:
            dtype (torch.dtype, optional): The data type of the tensors.
            device (torch.device, optional): The device where the tensors are placed.

        Returns:
            SiteTensor: The site tensor with updated device and/or dtype.
        """
        device = device or self.device
        dtype = dtype or self.dtype
        return SiteTensor(dims=self.dims, site_tensor=self, dtype=dtype, device=device)

    def initialize_site_tensor(self, site_state=[1.,], noise=0.0):
        """
        Initializes the site tensor with a given state and optional noise.

        The site tensor is initialized with random values, and the provided `site_state` is added
        to the first entry of the tensor. The tensor is then normalized.

        Args:
            site_state (list, optional): A list specifying the initial state for the site tensor (default is [1.]).
            noise (float, optional): The amount of noise to add to the site tensor (default is 0.0).
        """
        bD = self.dims['bond']
        pD = self.dims['phys']
        self['A'] = noise*torch.rand(bD,bD,bD,bD,pD).to(dtype=self.dtype, device=self.device)
        for n in range(len(site_state)):
            self['A'][0,0,0,0,n] += site_state[n]
        self['A'] = self['A']/self['A'].norm()

    def bond_permute(self, k):
        """
        Permutes the bond indices of the site tensor.

        This method returns a version of the site tensor with its bond dimensions permuted by an offset `k`.

        Args:
            k (int): The offset used to permute the bond dimensions.

        Returns:
            torch.Tensor: The permuted site tensor.
        """
        return self['A'].permute([(i+k)%4 for i in range(4)] + [4,])

    def initialize_corner_tensors(self):
        """
        Initializes the corner tensors for the site tensor.

        The corner tensors are computed from the bond-permuted site tensor using an einsum contraction.
        They are normalized, and stored in `self._corner_tensors`.

        If `initial_condition` is set to "random", the corner tensors are initialized randomly.

        Returns:
            None
        """
        bD = self.dims['bond']
        cD = self.dims['chi']
        if self.initial_condition == "random":
            self['C'] = [torch.rand(cD,cD) for _ in range(4)]
            return
        corner_tensors = []
        for k in range(4):
            ak = self.bond_permute(k)
            ck = torch.einsum("lurdp,luRDp->dDrR", ak, conj(ak))
            ck = ck.reshape(bD**2,bD**2)
            ck = ck/ck.norm()
            corner_tensors.append(ck)
        self['C'] = corner_tensors

    def initialize_edge_tensors(self):
        """
        Initializes the edge tensors for the site tensor.

        The edge tensors are computed from the bond-permuted site tensor using an einsum contraction.
        They are normalized, and stored in `self._edge_tensors`.

        If `initial_condition` is set to "random", the edge tensors are initialized randomly.

        Returns:
            None
        """
        bD = self.dims['bond']
        cD = self.dims['chi']
        if self.initial_condition == "random":
            self['E'] = [torch.rand(cD,cD,bD,bD) for _ in range(4)]
            return
        edge_tensors = []
        for k in range(4):
            ak = self.bond_permute(k)
            ek = torch.einsum("lurdp,LuRDp->lLrRdD", ak, conj(ak))
            ek = ek.reshape(bD**2,bD**2,bD,bD)
            ek = ek/ek.norm()
            edge_tensors.append(ek)
        self['E'] = edge_tensors
