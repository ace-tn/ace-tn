import torch

class PauliMatrix:
    """
    A class that represents a Pauli matrix or identity matrix.

    This class allows operations with Pauli matrices and supports multiplication, addition,
    subtraction, negation, and division. The matrices are represented as tensors, and 
    the operations are performed using PyTorch. The class can handle the four standard 
    Pauli matrices ('X', 'Y', 'Z', 'I') and allows scaling and Kronecker product operations.

    Attributes:
        key (str): A key representing the Pauli matrix ('X', 'Y', 'Z', 'I').
        dtype (torch.dtype): The data type for the matrix.
        device (torch.device): The device (e.g., CPU or CUDA) on which the matrix resides.
        mat (torch.Tensor): The Pauli matrix or identity matrix represented as a tensor.
    """
    mat = None
    def __init__(self, key, dtype, device, mat=None):
        """
        Initializes a Pauli matrix object.

        If a matrix (`mat`) is provided, it is used directly. Otherwise, a matrix corresponding 
        to the provided `key` is constructed (if the key is 'X', 'Y', 'Z', or 'I'). The matrix is
        then stored as a tensor on the specified device with the given data type.

        Args:
            key (str): A string representing the type of Pauli matrix ('X', 'Y', 'Z', 'I').
            dtype (torch.dtype): The data type of the matrix (e.g., torch.float32).
            device (torch.device): The device on which the matrix should be placed (e.g., 'cpu' or 'cuda').
            mat (torch.Tensor, optional): A custom matrix to initialize the object. If None, a default Pauli matrix is created.
        """
        self.key = key
        self.dtype = dtype
        self.device = device
        if mat is not None:
            self.mat = mat
        else:
            match key:
                case 'X':
                    self.mat = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype, device=device)
                case 'Y':
                    self.mat = torch.tensor([[0.,-1.], [1., 0.]], dtype=dtype, device=device)
                    if torch.is_complex(self.mat):
                        self.mat *= 1j
                case 'Z':
                    self.mat = torch.tensor([[1., 0.], [0.,-1.]], dtype=dtype, device=device)
                case 'I':
                    self.mat = torch.tensor([[1., 0.], [0., 1.]], dtype=dtype, device=device)

    def __mul__(self, other):
        """
        Multiplies the Pauli matrix by another matrix or scalar.

        If the other object is another instance of `PauliMatrix`, a Kronecker product of the two matrices is computed.
        If the other object is a scalar, the Pauli matrix is scaled by the scalar.

        Args:
            other (PauliMatrix or scalar): The object to multiply with.

        Returns:
            PauliMatrix: A new `PauliMatrix` object containing the result of the multiplication.
        """
        if isinstance(other, (PauliMatrix)):
            new_key = self.key + other.key
            phase = 1.
            if not torch.is_complex(self.mat):
                # flip sign for even product of real-valued pauli Y matrices
                phase = -1. if (new_key.count('Y') % 2 == 0) and (self.key.count('Y') % 2 != 0) else 1.
            return PauliMatrix(new_key, self.dtype, self.device, mat=phase*torch.kron(self.mat, other.mat))
        else:
            return PauliMatrix(self.key, self.dtype, self.device, mat=other*self.mat)

    def __imul__(self, other):
        """
        Performs an in-place Kronecker product with another Pauli matrix or scales the matrix by a scalar.

        If the other object is another instance of `PauliMatrix`, the Kronecker product is computed.
        If the other object is a scalar, the matrix is scaled by the scalar.

        Args:
            other (PauliMatrix or scalar): The object to multiply with.

        Returns:
            PauliMatrix or torch.Tensor: A new `PauliMatrix` object (in the case of scalar) or a tensor (in the case of matrix Kronecker product).
        """
        if isinstance(other, (PauliMatrix)):
            return torch.kron(self.mat, other.mat)
        else:
            return PauliMatrix(self.key, self.dtype, self.device, mat=other*self.mat)

    def __rmul__(self, other):
        """
        Multiplies a scalar by the Pauli matrix.

        This method is used when a scalar is multiplied on the left side of the matrix.

        Args:
            other (scalar): The scalar to multiply with.

        Returns:
            PauliMatrix: A new `PauliMatrix` object containing the scaled matrix.
        """
        return PauliMatrix(self.key, self.dtype, self.device, mat=other*self.mat)

    def __add__(self, other):
        """
        Adds two Pauli matrices.

        If the other object is an instance of `PauliMatrix`, the two matrices are added element-wise.

        Args:
            other (PauliMatrix): The `PauliMatrix` object to add.

        Returns:
            PauliMatrix: A new `PauliMatrix` object containing the result of the addition.
        """
        if isinstance(other, (PauliMatrix)):
            return PauliMatrix(self.key + other.key, self.dtype, self.device, mat=(self.mat + other.mat))

    def __sub__(self, other):
        """
        Subtracts another Pauli matrix from this Pauli matrix.

        If the other object is an instance of `PauliMatrix`, the two matrices are subtracted element-wise.

        Args:
            other (PauliMatrix): The `PauliMatrix` object to subtract.

        Returns:
            PauliMatrix: A new `PauliMatrix` object containing the result of the subtraction.
        """
        if isinstance(other, (PauliMatrix)):
            return PauliMatrix(self.key + other.key, self.dtype, self.device, mat=(self.mat - other.mat))

    def __neg__(self):
        """
        Negates the Pauli matrix.

        This method multiplies the matrix by -1, effectively negating all elements.

        Returns:
            PauliMatrix: A new `PauliMatrix` object containing the negated matrix.
        """
        return PauliMatrix(self.key, self.dtype, self.device, mat=-self.mat)

    def __truediv__(self, scalar):
        """
        Divides the Pauli matrix by a scalar.

        This method divides each element of the matrix by the given scalar.

        Args:
            scalar (float or int): The scalar by which to divide the matrix.

        Returns:
            PauliMatrix: A new `PauliMatrix` object containing the result of the division.
        """
        return PauliMatrix(self.key, self.dtype, self.device, mat=self.mat/scalar)


def pauli_matrices(dtype, device):
    """
    Creates and returns the four Pauli matrices (X, Y, Z, I) as instances of the PauliMatrix class.

    Parameters:
    -----------
    dtype : torch.dtype
        The data type of the Pauli matrices (e.g., torch.float32, torch.float64).
    device : torch.device
        The device on which the Pauli matrices will be allocated (e.g., CPU or GPU).

    Returns:
    --------
    tuple
        A tuple containing the four Pauli matrices (X, Y, Z, I) as instances of the PauliMatrix class.
    """
    X = PauliMatrix('X', dtype=dtype, device=device)
    Y = PauliMatrix('Y', dtype=dtype, device=device)
    Z = PauliMatrix('Z', dtype=dtype, device=device)
    I = PauliMatrix('I', dtype=dtype, device=device)
    return X,Y,Z,I
