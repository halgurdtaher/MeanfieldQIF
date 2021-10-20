import numpy as np

from NumSim.DynamicalSystem import DSys
from QIFSystems import QIFParams
from NumSim.SimData import SData


class NeuralMassBase(DSys):
    def __init__(self, p: QIFParams):
        self.p = p
        self.dxdt = np.zeros(p.x0.shape)

    def RHS(self, t, x):
        p = self.p
        dxdt = self.dxdt

        Iext_v = 0
        if (p.Iext is not None):
            Iext_v = p.Iext(t)

        r = x[0]
        v = x[1]

        dxdt[0] = p.Delta / np.pi + 2 * r * v
        dxdt[1] = v * v + p.eta0 - (np.pi * r) ** 2 + p.J * r + Iext_v

        return dxdt


class NeuralMassSTP(DSys):
    def __init__(self, p: QIFParams):
        self.p = p
        self.dxdt = np.zeros(p.x0.shape)

    def RHS(self, t, x):
        p = self.p
        dxdt = self.dxdt

        Iext_v = 0
        if (p.Iext is not None):
            Iext_v = p.Iext(t)

        r = x[0]
        v = x[1]
        d = x[2]
        u = x[3]


        dxdt[0] = p.Delta / np.pi + 2 * r * v
        dxdt[1] = v * v + p.eta0 - (np.pi * r) ** 2 + p.J * u * d * r + Iext_v
        dxdt[2] = (1 - d) / p.taud - u * d * r
        dxdt[3] = (p.U0 - u) / p.tauf + p.U0 * (1 - u) * r


        return dxdt


class NMData(SData):
    """
         Class to store QIF network simulation data
     """

    def __init__(self, path=None, relative_path=True, rescale_r=False):
        self.p: QIFParams = None

        self.xv: np.ndarray = None
        self.tv: np.ndarray = None
        SData.__init__(self, path, relative_path)
