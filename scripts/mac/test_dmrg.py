from spinsys import dmrg
from hamiltonians.triangular_lattice_model import DMRG_Hamiltonian


Nx = 3
Ny = 4
N = Nx * Ny
m = 50
H = DMRG_Hamiltonian(Nx, Ny)
dmrg.finite_system_dmrg(H, m, N, debug=True)
