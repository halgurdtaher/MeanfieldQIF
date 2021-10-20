from NumSim.Parameters import Params
import numpy as np

import NumSim.Solvers as Solvers


class DSys:
    """
        A base class that contains the definition of the right hand side of

        dx/dt=f(t,x)

        Attributes:
            p (Parameter.Params): Provides integration and system parameters
            dxdt (np.ndarray): Has shape of p.x0 and stores the right hand side f(x,t). Should be returned by RHS(). This is used to not create a new array at every call of RHS().
    """

    def __init__(self, p: Params):
        self.p = p
        self.dxdt = np.zeros(p.x0.shape)

    def RHS(self, t, x: np.ndarray):
        """
            Defines the right hand side f(x,t)

            Parameters:
                t (float): Time variable

                x (np.ndarray): State variable

            Returns:
                dxdt (np.ndarray): The right hand side.
        """

        return self.dxdt

    def integrate(self, solver_class=Solvers.SolverRK45, observers=[], auto_observe=True, freq=30, verbose=True):
        solver = solver_class(self, auto_observe=auto_observe, freq=freq)
        solver.observers.extend(observers)
        solver.integrate(verbose=verbose)

        return solver
