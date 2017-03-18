from spinsys.hamiltonians.quasi_heisenberg import nleg, oneleg
import unittest
from numpy import testing
import numpy as np


class TestCrossexamineNLegH1D(unittest.TestCase):

    def test_N4_W11_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 4, 1, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1).toarray()

        desired = oneleg.H(N, W1, c1, phi1).toarray()
        testing.assert_array_equal(output, desired)

    def test_N5_W3_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 5, 3, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1).toarray()
        desired = oneleg.H(N, W1, c1, phi1).toarray()
        testing.assert_array_equal(output, desired)

    def test_N8_W6_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 8, 6, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1).toarray()
        desired = oneleg.H(N, W1, c1, phi1).toarray()
        testing.assert_array_equal(output, desired)

    def test_N4_W11_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 4, 1, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()

        desired = oneleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        testing.assert_array_equal(output, desired)

    def test_N5_W3_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 5, 3, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        desired = oneleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        testing.assert_array_equal(output, desired)

    def test_N8_W6_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 8, 6, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        desired = oneleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        testing.assert_array_equal(output, desired)


class TestConsistencyNLegH2D(unittest.TestCase):

    def test_N6_W0_csqrt2_2_3leg_open(self):
        N, W, c, phi, J = 6, 0, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=2, mode='open').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=3, mode='open').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N8_W3_csqrt2_2_4leg_open(self):
        N, W, c, phi, J = 8, 3, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=2, mode='open').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=4, mode='open').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N12_W3_csqrt2_3_4leg_open(self):
        N, W, c, phi, J = 12, 3, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=3, mode='open').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=4, mode='open').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N6_W13_W21_csqrt2_2_3leg_open(self):
        N, W1, W2, c, phi, J = 6, 3, 1, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W1, c1=c, phi1=phi, J1=J,
                            W2=W2, c2=c, phi2=phi, J2=J,
                            nleg=2, mode='open').toarray()
        H2 = nleg.H(N, W1=W2, c1=c, phi1=phi, J1=J,
                            W2=W1, c2=c, phi2=phi, J2=J,
                            nleg=3, mode='open').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N6_W3_c1sqrt2_c2sqrt3_2_3leg_open(self):
        N, W, c1, c2, phi, J = 6, 0, np.sqrt(2), np.sqrt(3), 0, 1
        H1 = nleg.H(N, W1=W, c1=c1, phi1=phi, J1=J,
                            W2=W, c2=c2, phi2=phi, J2=J,
                            nleg=2, mode='open').toarray()
        H2 = nleg.H(N, W1=W, c1=c2, phi1=phi, J1=J,
                            W2=W, c2=c1, phi2=phi, J2=J,
                            nleg=3, mode='open').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N12_W13_W28_c1sqrt2_c2sqrt5_J11_J24_3_4leg_open(self):
        N, W1, W2, c1, c2 = 12, 3, 8, np.sqrt(2), np.sqrt(5)
        phi, J1, J2 = 0, 1, 4
        H1 = nleg.H(N, W1=W1, c1=c1, phi1=phi, J1=J1,
                            W2=W2, c2=c2, phi2=phi, J2=J2,
                            nleg=3, mode='open').toarray()
        H2 = nleg.H(N, W1=W2, c1=c2, phi1=phi, J1=J2,
                            W2=W1, c2=c1, phi2=phi, J2=J1,
                            nleg=4, mode='open').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N6_W0_csqrt2_2_3leg_periodic(self):
        N, W, c, phi, J = 6, 0, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=2, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=3, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N8_W3_csqrt2_2_4leg_periodic(self):
        N, W, c, phi, J = 8, 3, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=2, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=4, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N12_W3_csqrt2_3_4leg_periodic(self):
        N, W, c, phi, J = 12, 3, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=3, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W, c1=c, phi1=phi, J1=J,
                            W2=W, c2=c, phi2=phi, J2=J,
                            nleg=4, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N6_W13_W21_csqrt2_2_3leg_periodic(self):
        N, W1, W2, c, phi, J = 6, 3, 1, np.sqrt(2), 0, 1
        H1 = nleg.H(N, W1=W1, c1=c, phi1=phi, J1=J,
                            W2=W2, c2=c, phi2=phi, J2=J,
                            nleg=2, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W2, c1=c, phi1=phi, J1=J,
                            W2=W1, c2=c, phi2=phi, J2=J,
                            nleg=3, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N6_W3_c1sqrt2_c2sqrt3_2_3leg_periodic(self):
        N, W, c1, c2, phi, J = 6, 0, np.sqrt(2), np.sqrt(3), 0, 1
        H1 = nleg.H(N, W1=W, c1=c1, phi1=phi, J1=J,
                            W2=W, c2=c2, phi2=phi, J2=J,
                            nleg=2, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W, c1=c2, phi1=phi, J1=J,
                            W2=W, c2=c1, phi2=phi, J2=J,
                            nleg=3, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)

    def test_N12_W13_W28_c1sqrt2_c2sqrt5_J11_J24_3_4leg_periodic(self):
        N, W1, W2, c1, c2 = 12, 3, 8, np.sqrt(2), np.sqrt(5)
        phi, J1, J2 = 0, 1, 4
        H1 = nleg.H(N, W1=W1, c1=c1, phi1=phi, J1=J1,
                            W2=W2, c2=c2, phi2=phi, J2=J2,
                            nleg=3, mode='periodic').toarray()
        H2 = nleg.H(N, W1=W2, c1=c2, phi1=phi, J1=J2,
                            W2=W1, c2=c1, phi2=phi, J2=J1,
                            nleg=4, mode='periodic').toarray()
        E1 = np.linalg.eigvalsh(H1)
        E2 = np.linalg.eigvalsh(H2)
        testing.assert_array_almost_equal(E1, E2)


if __name__ == '__main__':
    unittest.main()
