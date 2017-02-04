import numpy as np


class TimeMachine():

    def __init__(self, eigvs, eigvecs, psi):
        """
        Time evolves a given vector to any point in the past or future.

        Args: "eigvs" eigenenergies of the Hamiltonian. Numpy 1D array
              "eigvecs" eigenvectors of the Hamiltonian. Numpy 2D square array
              "psi" initial state. Numpy 1D array
        """
        self.eigenenergies = eigvs
        self.back_transform_matrix = eigvecs
        self.initial_state = self._convert_to_eigenbasis(eigvecs, psi)
        self.curr_state = self.initial_state.copy()
        self.coeffs = 1    # the exponential coeffs for psi when time evolving
        self.last_dt = 0

    def _convert_to_eigenbasis(self, U, psi):
        return U.T.conjugate().dot(psi)

    def evolve_by_step(self, dt, basis='orig'):
        """Evolves the state by dt

        Args: "dt" time step, float
              "basis" "orig" or "energy". The basis of the returned state
        Returns: Numpy 1D array
        """
        if not dt == self.last_dt:
            self.coeffs = np.exp(-1j * self.eigenenergies * dt)
            self.last_dt = dt
        self.curr_state *= self.coeffs
        if basis == 'orig':
            return self.back_transform_matrix.dot(self.curr_state)
        elif basis == 'energy':
            return self.curr_state

    def evolve_to_time(self, t, basis='orig'):
        """Evolves the state by time "t"

        Args: "t" time, float
              "basis" "orig" or "energy". The basis of the returned state
        Returns: Numpy 1D array
        """
        self.coeffs = np.exp(-1j * self.eigenenergies * t)
        self.curr_state = self.coeffs * self.initial_state
        if basis == 'orig':
            return self.back_transform_matrix.dot(self.curr_state)
        elif basis == 'energy':
            return self.curr_state
