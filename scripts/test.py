from triangular_lattice_hamitonian import hamiltonian
from scipy import sparse


Nx = 4
Ny = 3
H = hamiltonian(Nx, Ny)
E = sparse.linalg.eigsh(H, k=20, which='SA', return_eigenvectors=False)
print(E)
