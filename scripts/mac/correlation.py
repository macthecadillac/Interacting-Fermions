from spinsys import constructors, half
from scipy import sparse
import hamiltonians.triangular_lattice_model as t


def _gen_full_ops(N):
    σx = constructors.sigmax()
    σy = constructors.sigmay()
    σz = constructors.sigmaz()
    x_mats = [half.full_matrix(σx, k, N) for k in range(N)]
    y_mats = [half.full_matrix(σy, k, N) for k in range(N)]
    z_mats = [half.full_matrix(σz, k, N) for k in range(N)]
    return x_mats, y_mats, z_mats


def _gen_ops(N, bonds):
    """generate the H_z and H_pm components of the Hamiltonian"""
    ops = []
    x_mats, y_mats, z_mats = _gen_full_ops(N)
    for bond in bonds:
        site1, site2 = bond
        i, j = site1.lattice_index, site2.lattice_index
        ops.append(x_mats[i].dot(x_mats[j]) + y_mats[i].dot(y_mats[j]))
    return ops


Nx = 3
Ny = 6
J_z = 1
J_pm = 0.5
H = t.hamiltonian_dp(Nx, Ny, J_z=J_z, J_pm=J_pm)
E, V = sparse.linalg.eigsh(H, which='SA', k=1)
ψ = sparse.csc_matrix(V)
nearest, second, third = t._generate_bonds(Nx, Ny)
xxyy = _gen_ops(N, nearest) + _gen_ops(N, second) + _gen_ops(N, third)
