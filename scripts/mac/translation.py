from scipy import sparse
import spinsys
from hamiltonians.triangular_lattice_model import hamiltonian


Nx = 4
Ny = 3
N = Nx * Ny
H = hamiltonian(Nx, Ny)
E, ψ = sparse.linalg.eigsh(H, k=1, which='SA')
ψ = sparse.csc_matrix(ψ)
T1 = spinsys.half.translation_operator(Nx, Ny, direction='x')
T2 = T1.dot(T1)
T3 = T2.dot(T1)
T4 = spinsys.half.translation_operator(Nx, Ny, direction='y')
T5 = T4.dot(T4)
T6 = T5.dot(T4)
T7 = T4.dot(sparse.linalg.inv(T1))
T8 = T7.dot(T7)
T9 = T8.dot(T7)
for T in [T1, T2, T3, T4, T5, T6, T7, T8, T9]:
    print(ψ.T.conjugate().dot(T).dot(ψ)[0, 0])
