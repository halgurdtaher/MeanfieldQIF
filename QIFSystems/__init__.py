"""
Contains base classes to perform numerical simulations of networks of Quadratic integrate & fire (QIF) neurons.
The right hand side reads

dV_i/dt=V_i^2 + eta_i + eta_t(t) + J*r .

When a voltage exceeds the thresholds Vr, it is being reset to Vp. We consider these values in the limit to infinity.
For this the integration is stopped at V=100 and V is set to V=-100. The dynamics of the neuron is stopped for a
refractory period TRefrac, to account for the evolution from 100 to infinity and from -infinity to -100.
In order to implement the coupling term J*r, a spike is registered at the middle of the refractory period:
the voltages of all other neurons are changed instantaneously by J/N.

eta_t(t) is a time dependent external current, common to all QIF neurons.

The model and simulation method are described in:
Ernest Montbrió, Diego Pazó, and Alex Roxin. “Macroscopic Description for Networks of Spiking Neurons”.
Phys. Rev. X 5 (2 June 2015), p. 021028.



"""

import numpy as np

from NumSim.DynamicalSystem import DSys
from NumSim.Parameters import Params
from NumSim.Observers import ObserverBase
from NumSim.SimData import SData


def deterministicLorentzian(size, loc, scale):
    jvpi = np.pi / 2 * (2 * np.arange(1, size + 1) - size - 1) / (size + 1)
    values = loc + scale * np.tan(jvpi)
    return values


class QIFParams(Params):
    """
         Parameters of the QIF network.

         Attributes:
             eta: Constant external current to each neuron, must have shape of V
             eta0: Center of the Lorentzian distribution of eta
             Delta: Width parameter of the Lorentzian distribution of eta
             Vp: Threshold voltage at which the reset to Vr takes place
             Vr: Reset voltage
             AvgSavestep: Defines how many integration steps to skip for storing the mean values of the microscopic quantities
             tauiChoice: Defines the neuron indices for which the firing times are stored.
             atol, rtol: Absolute and relative tolerances for the RK45 solver
     """

    def __init__(self):
        Params.__init__(self)
        self.AvgSavestep = 1

        self.eta = None

        self.eta0 = None
        self.Delta = None

        self.taum = None
        self.taud = None
        self.tauf = None
        self.U0 = 0

        self.AvgSavestep = 1
        self.tauiChoice = None

        self.atol = 1e-12
        self.rtol = 1e-7

        self.Iext = None
        self.Vp = 100
        self.Vr = -100


class QIFData(SData):
    """
         Class to store QIF network simulation data
     """

    def __init__(self, path=None, relative_path=True, rescale_r=False):
        self.atv: np.ndarray = None
        self.avgvv: np.ndarray = None
        self.avguv: np.ndarray = None
        self.avgdv: np.ndarray = None

        self.avgrv: np.ndarray = None
        self.rawrv: np.ndarray = None

        self.taui = None
        self.p: QIFParams = None

        self.xv: np.ndarray = None
        self.tv: np.ndarray = None
        SData.__init__(self, path, relative_path)


class QIFBase(DSys):
    """
        Defines the right hand side of the QIF network.
        The term J*r is missing, since it is handled by the QIFObserverBase during integration,
        via an instantaneous change of the state variable.
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

        dxdt[:] = x * x + p.eta + Iext_v

        return dxdt


class QIFObserverBase(ObserverBase):
    """
        The QIF observer base class. Takes care of reset rule and spike handling,
        as well as calculating and storing microscopic quantities, like individual membrane potentials (xv) and
        mean quantities, like the mean membrane potential (avgvv).
    """

    def __init__(self, p: QIFParams):

        self.t0 = None
        self.t1 = None

        self.p = p

        self.TRefrac = 2.0 / p.Vp
        self.TRefrac2 = self.TRefrac / 2

        self.taui = [[] for k in range(p.N)]

        self.tv = []
        self.avgvv = []

        self.refrac_done_times = np.zeros(p.N)
        self.fire_times = np.zeros(p.N)
        self.fire_done = np.zeros(p.N)
        self.refrac_idx = np.array([], dtype=np.int)

        self.total_step = 0

        self.store_taui = True

        self.nfire = 0
        self.nfirev = []

    def handle_firing(self, t, x: np.ndarray, nfire):
        """
            What to do when there is a  "delta - spike" in the network.
        """
        p = self.p

        r = nfire / p.N  # contribution to the mean firing rate by nfire spikes

        x[:] += p.J * r  # instantaneous increase of all membrane potentials by p.J*r

    def observe(self, t, x: np.ndarray):
        p = self.p
        self.spike_check(t, x)  # check for spikes

        if (self.total_step % p.AvgSavestep == 0):
            num_refrac = self.refrac_idx.size
            avgv = x.sum() - num_refrac * p.Vr
            avgv /= p.N - num_refrac

            self.nfirev.append(self.nfire)
            self.avgvv.append(avgv)
            self.tv.append(t)

    def spike_check(self, t, x: np.ndarray):
        """
            This function deals with the voltage reset and checks for spikes.
        """

        p = self.p
        refrac_change = False

        spj = np.where(x[:p.N] >= p.Vp)[0]  # check which neurons are above threshold
        if (spj.size > 0):
            refrac_change = True
            self.fire_times[spj] = t + self.TRefrac2  # define when the above threshold neurons will reach V->infinity
            self.refrac_done_times[
                spj] = t + self.TRefrac  # define when the neurons have reached p.Vr after surpassing p.Vr

        refrac_idx = self.refrac_idx  # just a shortcut for below
        if (len(refrac_idx) > 0):  # if there are neurons in the refractory period
            sub_fire_done = self.fire_done[refrac_idx]  # get those which have already spiked during the refrac. period

            sub_fire_idx = np.where(np.logical_and(t >= self.fire_times[refrac_idx], sub_fire_done == 0))[
                0]  # get those who will spike in this step

            firing_idx = refrac_idx[sub_fire_idx]  # indices of spiking neurons

            self.fire_done[firing_idx] = 1  # update who has spiked already
            self.fire_times[firing_idx] = 0  # reset spike times

            sub_refrac_done_idx = np.where(t >= self.refrac_done_times[refrac_idx])[
                0]  # indices of those who are now done with refractory period
            if (sub_refrac_done_idx.size > 0):
                top_refrac_done_idx = refrac_idx[sub_refrac_done_idx]
                self.refrac_done_times[top_refrac_done_idx] = 0
                self.fire_done[top_refrac_done_idx] = 0
                refrac_change = True

            nfire = len(firing_idx)
            if (nfire > 0):
                if (self.store_taui):  # if the spike times are supposed to be stored
                    for fidx in firing_idx:
                        if (p.tauiChoice is not None):
                            if (fidx in p.tauiChoice):
                                self.taui[fidx].append(t)  # store spike times
                        else:
                            self.taui[fidx].append(t)

                self.handle_firing(t, x, nfire)  # what to do if neurons spike.

                # keep track of the total number of spikes.
                # the array self.nfirev stores this values in time
                # it can be used to calculate the firing rate
                self.nfire += nfire

        if (refrac_change):  # if there are changes to the set of refractory neurons
            self.refrac_idx = np.where(self.refrac_done_times != 0)[0]  # check which ones
        x[self.refrac_idx] = p.Vr  # make sure that refractory neurons remain frozen at V = p.Vr
