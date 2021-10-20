"""
    Solves the QIF initial value problem dV/dt=V^2+eta + J*r, where V and eta are numpy arrays of the same shape.
    Here N = 10000 QIF neurons are simulated.
    J is the synaptic weigth. At each spike of one of the neurons, the membrane potential V of the other neuron is
    instantaneously increeased by J/N. The mean firing rate as a function of time is calculated by counting spikes
    in a running time window of length 1e-2.

    The model and simulation method are described in:
    Exact neural mass model for synaptic-based working memory
    Taher H, Torcini A, Olmi S (2020) Exact neural mass model for synaptic-based working memory.
    PLOS Computational Biology 16(12): e1008533. https://doi.org/10.1371/journal.pcbi.1008533
    This example reproduces Fig. 1B of that publication

    Plots the results using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
from NumSim.Solvers import SolveEuler, SolverRK45
from QIFSystems import QIFParams, QIFData
from QIFSystems import deterministicLorentzian
from QIFSystems.NeuralMass import NeuralMassSTP, NMData
from Examples.MesoscopicSTP.Plot import plot_dynamics
from QIFSystems.QIF_STP import QIF_STP, QIFObserverSTP


def run_QIF(p, save_name):
    sys = QIF_STP(p)
    qobs = QIFObserverSTP(p)

    solver = SolveEuler(sys, auto_observe=False)
    solver.observers.append(qobs)
    solver.integrate()

    data = QIFData()
    data.p = p

    tv = np.array(qobs.tv)

    data.avgvv = np.array(qobs.avgvv)
    data.avgdv = np.array(qobs.avgdv)
    data.avguv = np.array(qobs.avguv)
    data.atv = tv

    rv_dt = tv[1] - tv[0]

    dnfire = np.zeros(tv.size)
    dnfire[1:] = np.diff(qobs.nfirev, axis=0)

    data.rawrv = dnfire / rv_dt / p.N

    cur_Dt = rv_dt
    new_Dt = 1e-2
    n_avg = int(new_Dt / cur_Dt)
    data.avgrv = np.convolve(data.rawrv, np.ones(n_avg) / n_avg, mode="same")

    data.taui = qobs.taui

    data.save(f"Data/{save_name}_network")


def run_NM(p, save_name):
    sys = NeuralMassSTP(p)

    solver = SolverRK45(sys, auto_observe=True)
    solver.integrate()

    data = NMData()
    data.p = p

    data.tv = np.array(solver.tv)
    data.xv = np.array(solver.xv)

    data.save(f"Data/{save_name}_NM")


if (__name__ == "__main__"):
    p = QIFParams()

    p.taum = 0.015
    p.dt = 1e-4
    p.t0 = -5
    p.t1 = 1 / p.taum
    p.AvgSavestep = 10

    p.taud = 0.2 / p.taum
    p.tauf = 1.5 / p.taum
    p.U0 = 0.2


    def stim_step(t):
        ret = 0
        if (10 < t < 20 or 30 < t < 40):
            ret = 2

        return ret


    p.N = 10000
    p.x0 = np.zeros(p.N + 2)
    p.x0[-2] = 1
    p.x0[-1] = p.U0

    p.J = 15
    p.eta0 = -1
    p.Delta = 0.25

    p.eta = deterministicLorentzian(p.N, p.eta0, p.Delta)

    p.Iext = stim_step

    save_name = "MesoscopicSTP"

    run_QIF(p, save_name)

    p.x0 = np.zeros(4)
    p.x0[2] = 1.0
    p.x0[3] = p.U0
    run_NM(p, save_name)

    data_net = QIFData(f"Data/{save_name}_network.dill")
    data_NM = NMData(f"Data/{save_name}_NM.dill")

    plot_dynamics(data_net, data_NM)
    plt.show()
