"""
    Solves the initial value problem dx/dt=-a*x, where a and x are numpy arrays of the same shape.
    Plots the results using matplotlib.
"""

import numpy as np
from NumSim.DynamicalSystem import DSys
from NumSim.Parameters import Params
from NumSim.SimData import SData
from NumSim.Solvers import SolverRK45
from Examples.ExpDecay.Plot import plot_exp_decay


class ExpData(SData):
    base_path = "Data/"

    def __init__(self, *args):
        self.tv = None
        self.xv = None
        SData.__init__(self, *args)


class ExpParams(Params):
    """
       Parameters for integration and system

       Attributes:
            a (float or numpy array): Decay constant
    """

    def __init__(self):
        Params.__init__(self)

        self.a = np.linspace(1, 10, 20)  # the decay constant(s)
        self.x0 = np.zeros(self.a.size)

        self.x0[:] = 10

        self.t0 = 0
        self.t1 = 5


class ExpSys(DSys):
    """
        Defines the right hand side.
    """

    def RHS(self, t, x: np.ndarray):
        p = self.p
        self.dxdt[:] = -p.a * x

        return self.dxdt


if (__name__ == "__main__"):
    p = ExpParams()
    sys = ExpSys(p)

    solver = SolverRK45(sys, auto_observe=True)

    solver.integrate()

    data = ExpData()
    data.p = p
    data.tv = solver.tv
    data.xv = solver.xv
    data.save("exp_decay_example")

    plot_exp_decay(data)
