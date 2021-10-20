import numpy as np

from NumSim.DynamicalSystem import DSys
from QIFSystems import QIFParams, QIFObserverBase


class QIF_STP(DSys):
    """
        Defines the right hand side of the QIF network.
    """

    def __init__(self, p: QIFParams):
        self.p = p
        self.dxdt = np.zeros(p.x0.shape)

    def RHS(self, t, x):
        p = self.p
        dxdt = self.dxdt

        Iext_v = 0
        if (p.Iext is not None):
            Iext_v = p.Iext(t)

        v = x[:p.N]
        d = x[-2]
        u = x[-1]

        dxdt[:p.N] = v * v + p.eta + Iext_v
        dxdt[-2] = (1 - d) / p.taud
        dxdt[-1] = (p.U0 - u) / p.tauf

        return dxdt


class QIFObserverSTP(QIFObserverBase):
    """
        The QIF observer base class. Takes care of reset rule and spike handling,
        as well as calculating and storing microscopic quantities, like individual membrane potentials (xv) and
        mean quantities, like the mean membrane potential (avgvv).
    """

    def __init__(self, p: QIFParams):
        super().__init__(p)
        self.avgdv = []
        self.avguv = []

    def handle_firing(self, t, x: np.ndarray, nfire):
        """
            What to do when there is a  "delta - spike" in the network.
        """
        p = self.p

        r = nfire / p.N  # contribution to the mean firing rate by nfire spikes

        d = x[-2]
        u = x[-1]

        x[:p.N] += p.J * d * u * r  # instantaneous increase of all membrane potentials by p.J*r
        x[-2] -= d * u * r
        x[-1] += p.U0 * (1 - u) * r

    def observe(self, t, x: np.ndarray):
        p = self.p
        self.spike_check(t, x)  # check for spikes

        if (self.total_step % p.AvgSavestep == 0):
            num_refrac = self.refrac_idx.size
            avgv = x[:p.N].sum() - num_refrac * p.Vr
            avgv /= p.N - num_refrac

            self.nfirev.append(self.nfire)
            self.avgvv.append(avgv)
            self.avgdv.append(x[-2])
            self.avguv.append(x[-1])
            self.tv.append(t)
