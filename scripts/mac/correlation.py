import numpy as np
from spinsys import constructors, half, utils
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
    plot = np.empty((N, 3))
    for j in range(N):
        op = x_mats[i].dot(x_mats[j]) + y_mats[i].dot(y_mats[j])
        plot[j, 0] = j % Nx
        plot[j, 1] = j // Nx
        plot[j, 2] = ψ.T.conjugate().dot(op).dot(ψ)[0, 0].real
    return plot


def gen_plot_z(Nx, Ny, i, ψ):
    plot = np.empty((N, 3))
    for j in range(N):
        op = z_mats[i].dot(z_mats[j])
        plot[j, 0] = j % Nx
        plot[j, 1] = j // Nx
        plot[j, 2] = ψ.T.conjugate().dot(op).dot(ψ)[0, 0].real
    return plot


Nx = 3
Ny = 6
N = Nx * Ny
J_z = 1
J_pm = 0.5
i = 0
res = 21
coord = (i, i // Nx)
timer = utils.timer.Timer(res)
header = 'Jz={}, J+-={}, J++--={}\nix         iy         Sz corr'

x_mats, y_mats, z_mats = _gen_full_ops(N)
for J_ppmm in np.linspace(-1, 1, res):
    H = t.hamiltonian_dp(Nx, Ny, J_z=J_z, J_pm=J_pm, J_ppmm=J_ppmm)
    E, V = sparse.linalg.eigsh(H, which='SA', k=1)
    ψ = sparse.csc_matrix(V)
    plot1 = gen_plot_z(Nx, Ny, i, ψ)
    plot2 = gen_plot_xy(Nx, Ny, i, ψ)
    np.savetxt('/home/mac/plot_z_{}x{}_i={}_Jz={}_J+-={}_J++--={}.txt'
               .format(Nx, Ny, coord, J_z, J_pm, round(J_ppmm, 1)),
               plot1, fmt='%10.7f', header=header.format(J_z, J_pm, round(J_ppmm, 1)),
               footer=' ', comments=' ')
    np.savetxt('/home/mac/plot_xy_{}x{}_i={}_Jz={}_J+-={}_J++--={}.txt'
               .format(Nx, Ny, coord, J_z, J_pm, round(J_ppmm, 1)),
               plot2, fmt='%10.7f', header=header.format(J_z, J_pm, round(J_ppmm, 1)),
               footer=' ', comments=' ')
    timer.progress()
