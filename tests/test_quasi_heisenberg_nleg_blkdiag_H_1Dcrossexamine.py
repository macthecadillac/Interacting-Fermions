from spinsys.hamiltonians.quasi_heisenberg import nleg, nleg_blkdiag
import unittest
from numpy import testing
import numpy as np


class TestCrossexamineNLegH1D(unittest.TestCase):

    def test_N4_W11_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 4, 1, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1).toarray()

        H2 = nleg.H(N, W1, c1, phi1).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N5_W3_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 5, 3, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1).toarray()
        H2 = nleg.H(N, W1, c1, phi1).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N8_W6_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 8, 6, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1).toarray()
        H2 = nleg.H(N, W1, c1, phi1).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N4_W11_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 4, 1, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1, mode='periodic').toarray()

        H2 = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N5_W3_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 5, 3, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1, mode='periodic').toarray()
        H2 = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N8_W6_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 8, 6, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1, mode='periodic').toarray()
        H2 = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)


if __name__ == '__main__':
    unittest.main()
