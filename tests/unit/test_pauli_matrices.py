import pytest
import torch
from model.pauli_matrix import PauliMatrix, pauli_matrices

@pytest.fixture
def setup_matrices():
    dtype = torch.float32
    device = torch.device('cpu')
    X, Y, Z, I = pauli_matrices(dtype, device)
    return X, Y, Z, I, dtype, device

def test_initialization(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    # Test if Pauli matrices are created correctly
    assert X.mat.shape == (2, 2), "Pauli X matrix should be 2x2"
    assert Y.mat.shape == (2, 2), "Pauli Y matrix should be 2x2"
    assert Z.mat.shape == (2, 2), "Pauli Z matrix should be 2x2"
    assert I.mat.shape == (2, 2), "Pauli I matrix should be 2x2"

    assert X.dtype == dtype, "Pauli matrix dtype should match the input dtype"
    assert X.device == device, "Pauli matrix device should match the input device"

def test_addition(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    result = X + Y
    assert result.mat.shape == (2, 2), "Resulting matrix should have shape 2x2"
    assert torch.allclose(result.mat, X.mat + Y.mat), "Matrix addition did not work correctly"

def test_subtraction(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    result = X - Z
    assert result.mat.shape == (2, 2), "Resulting matrix should have shape 2x2"
    assert torch.allclose(result.mat, X.mat - Z.mat), "Matrix subtraction did not work correctly"

def test_negation(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    result = -X
    assert result.mat.shape == (2, 2), "Negated matrix should have shape 2x2"
    assert torch.allclose(result.mat, -X.mat), "Matrix negation did not work correctly"

def test_multiplication(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    result = X * 2
    assert result.mat.shape == (2, 2), "Scaled matrix should have shape 2x2"
    assert torch.allclose(result.mat, 2 * X.mat), "Matrix scaling did not work correctly"

    result = X * Y  # Kronecker product
    assert result.mat.shape == (4, 4), "Kronecker product matrix should have shape 4x4"
    
def test_inplace_multiplication(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    X_copy = X
    X_copy *= Y  # In-place Kronecker product
    result = X_copy  # Store the result of the in-place operation

    assert result.shape == (4, 4), "In-place Kronecker product result should have shape 4x4"

    # Check if original matrix has been modified correctly
    assert torch.allclose(result, torch.kron(X.mat, Y.mat)), "In-place multiplication failed"


def test_division(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    result = X / 2
    assert result.mat.shape == (2, 2), "Divided matrix should have shape 2x2"
    assert torch.allclose(result.mat, X.mat / 2), "Matrix division did not work correctly"

def test_pauli_matrices_function(setup_matrices):
    X, Y, Z, I, dtype, device = setup_matrices

    # Check if pauli_matrices returns correct instances
    assert isinstance(X, PauliMatrix), "X should be an instance of PauliMatrix"
    assert isinstance(Y, PauliMatrix), "Y should be an instance of PauliMatrix"
    assert isinstance(Z, PauliMatrix), "Z should be an instance of PauliMatrix"
    assert isinstance(I, PauliMatrix), "I should be an instance of PauliMatrix"

    # Check if matrices returned by pauli_matrices function have correct dtype and device
    assert X.dtype == dtype, "Pauli matrices should have correct dtype"
    assert X.device == device, "Pauli matrices should have correct device"

if __name__ == "__main__":
    pytest.main()
