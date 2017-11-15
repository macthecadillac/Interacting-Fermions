import numpy as np
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


def gen_plot_xy(Nx, Ny, i, ψ):
    plot = np.empty(N)
    for j in range(N):
        op = x_mats[i].dot(x_mats[j]) + y_mats[i].dot(y_mats[j])
        plot[j] = ψ.T.conjugate().dot(op).dot(ψ)[0, 0].real
    return np.flip(plot.reshape(Ny, -1), axis=0)


def gen_plot_z(Nx, Ny, i, ψ):
    plot = np.empty(N)
    for j in range(N):
        op = z_mats[i].dot(z_mats[j])
        plot[j] = ψ.T.conjugate().dot(op).dot(ψ)[0, 0].real
    return np.flip(plot.reshape(Ny, -1), axis=0)


Nx = 3
Ny = 6
N = Nx * Ny
J_z = 1
J_pm = 0.5
H = t.hamiltonian_dp(Nx, Ny, J_z=J_z, J_pm=J_pm)
E, V = sparse.linalg.eigsh(H, which='SA', k=1)
ψ = sparse.csc_matrix(V)
x_mats, y_mats, z_mats = _gen_full_ops(N)
for i in range(N):
    plot = gen_plot_z(Nx, Ny, i, ψ)
    np.savetxt('/home/mac/plot_z_i={}.txt'.format(i), plot, fmt='%7.4f')
