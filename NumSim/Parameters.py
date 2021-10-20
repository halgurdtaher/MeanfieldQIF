import numpy as np
from copy import copy


class Params:
    """
        Contains information for the solver. Inherited classes might provide parameters for the dynamics system.

        Attributes:
            t0 (float): Integration start time. Default is t0 = 0.
            t1 (float): Integration stop time
            x0 (np.ndarray): Array of any shape that defines the initial value x(t0)
            dt: Defines the constant timestep for the Euler solver

            rtol: Relative tolerance for the RK45 solver. Default is rtol = 1e-07.
            atol: Absolute tolerance for the RK45 solver. Default is atol = 1e-012.
            method: Defines the method used by scipy.integrate.ode(). Default is "dopri5" for the RK45 solver.
    """

    def __init__(self):
        self.t0 = 0
        self.t1 = None
        self.x0: np.ndarray = None
        self.dt = None

        self.rtol = 1e-7
        self.atol = 1e-12

        self.method = "dopri5"
