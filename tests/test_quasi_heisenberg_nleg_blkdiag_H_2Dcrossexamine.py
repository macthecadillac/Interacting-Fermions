from spinsys.hamiltonians.quasi_heisenberg import nleg_blkdiag, nleg
import numpy as np
from numpy import testing
import unittest


class TestCrossexamineNLegH2D(unittest.TestCase):

    def test_N4_W11_csqrt2_phi1_2_2leg_open(self):
        N, W, c, phi, J = 4, 1, np.sqrt(2), 1, 1
        H1 = nleg_blkdiag.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                            phi2=phi, J2=J, nleg=2).toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                    phi2=phi, J2=J, nleg=2).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N6_W3_csqrt2_phi1_2_3leg_open(self):
        N, W, c, phi, J = 6, 3, np.sqrt(2), 1, 1
        H1 = nleg_blkdiag.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                            phi2=phi, J2=J, nleg=2).toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                    phi2=phi, J2=J, nleg=2).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N8_W6_csqrt2_phi1_2_4leg_open(self):
        N, W, c, phi, J = 8, 6, np.sqrt(2), 1, 1
        H1 = nleg_blkdiag.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                            phi2=phi, J2=J, nleg=2).toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                    phi2=phi, J2=J, nleg=2).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N12_W6_csqrt2_phi1_3_4leg_open(self):
        N, W, c, phi, J = 12, 6, np.sqrt(2), 1, 1
        H1 = nleg_blkdiag.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                            phi2=phi, J2=J, nleg=3).toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                    phi2=phi, J2=J, nleg=3).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N12_W0_csqrt2_phi1_3_4leg_open(self):
        N, W, c, phi, J = 12, 0, np.sqrt(2), 1, 1
        H1 = nleg_blkdiag.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                            phi2=phi, J2=J, nleg=3).toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                    phi2=phi, J2=J, nleg=3).toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N4_W11_csqrt2_phi1_2_2leg_periodic(self):
        N, W, c, phi, J = 4, 1, np.sqrt(2), 1, 1
        H1 = nleg_blkdiag.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                            phi2=phi, J2=J, nleg=2, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                    phi2=phi, J2=J, nleg=2, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N6_W3_csqrt2_phi1_2_3leg_periodic(self):
        N, W, c, phi, J = 6, 3, np.sqrt(2), 1, 1
        H1 = nleg_blkdiag.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                            phi2=phi, J2=J, nleg=2, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J, W2=W, c2=c,
                    phi2=phi, J2=J, nleg=2, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N8_W6_csqrt2_phi1_2_4leg_periodic(self):
        N, W1, c1, phi1 = 8, 6, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1, nleg=2, mode='periodic').toarray()
        H2 = nleg.H(N, W1, c1, phi1, nleg=2, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N12_W6_csqrt2_phi1_3_4leg_periodic(self):
        N, W1, c1, phi1 = 12, 6, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1, nleg=4, mode='periodic').toarray()
        H2 = nleg.H(N, W1, c1, phi1, nleg=4, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)

    def test_N12_W0_csqrt2_phi1_3_4leg_periodic(self):
        N, W1, c1, phi1 = 12, 0, np.sqrt(2), 1
        H1 = nleg_blkdiag.H(N, W1, c1, phi1, nleg=3, mode='periodic').toarray()
        H2 = nleg.H(N, W1, c1, phi1, nleg=3, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        for E in E1:
            self.assertLess(np.min(np.abs(E2 - E)), 1e-6)


if __name__ == '__main__':
    unittest.main()
