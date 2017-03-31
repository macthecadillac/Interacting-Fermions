import math
from numpy.linalg import matrix_power, eig
from scipy.sparse.linalg import expm, expm_multiply
from scipy.sparse import csc_matrix
import numpy as np
import spinsys as s
import time
from spinsys.hamiltonians.ising.one_d_random import H
from spinsys.hamiltonians.spin_flip.x_field import flip_Sz
from tempfile import TemporaryFile

N = 4
H1 = flip_Sz(N, np.pi*.5, 0).tocsc()
H2 = H(N, 2*np.pi , .05).tocsc()
S = s.half.block_diagonalization_transformation(N).toarray()
psi = np.zeros(2**N)
psi[np.random.randint((2**N))] = 1
U1 = expm(-1j*H1)
U2 = expm(-1j*H2)
Uf = U2.dot(U1).toarray()
print(Uf)
A,B = eig(Uf)
print(A)
A = sorted((1j*np.log(A)).real)
print(A)
UfB = S.dot(Uf.dot(S.T))
print(UfB)
print(H2)

# psi_t = []
# psi_t.append(psi)
#
# for t in range(1,150):
#     psi = Uf.dot(psi)
#     psi_t.append(psi)
#
# np.savez('outfile.npz', psi_t_1=psi_t)
# npzfile = np.load('outfile.npz')
# print(npzfile.files)