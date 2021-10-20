"""
    Solves the QIF initial value problem dV/dt=V^2+eta + J*r, where V and eta are numpy arrays of the same shape.
    Here N = 10000 QIF neurons are simulated.
    J is the synaptic weigth. At each spike of one of the neurons, the membrane potential V of the other neuron is
    instantaneously increeased by J/N. The mean firing rate as a function of time is calculated by counting spikes
    in a running time window of length 1e-2.

    The model and simulation method are described in:
    Ernest Montbrió, Diego Pazó, and Alex Roxin. “Macroscopic Description for Networks of Spiking Neurons”.
    Phys. Rev. X 5 (2 June 2015), p. 021028.

    This example reproduces Fig. 2(a-g) of that publication

    Plots the results using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
from NumSim.Solvers import SolveEuler, SolverRK45
from QIFSystems import QIFBase, QIFParams, QIFObserverBase, QIFData
from QIFSystems import deterministicLorentzian
from QIFSystems.NeuralMass import NeuralMassBase, NMData
from Examples.MPR2015.Plot import plot_dynamics


def stim_step(t):
    ret = 0
    if (0 < t < 30):
        ret = 3

    return ret


def run_QIF(p, save_name):
    sys = QIFBase(p)
    qobs = QIFObserverBase(p)

    solver = SolveEuler(sys, auto_observe=False)
    solver.observers.append(qobs)
    solver.integrate()

    data = QIFData()
    data.p = p

    tv = np.array(qobs.tv)
    avgvv = np.array(qobs.avgvv)

    data.avgvv = avgvv
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
    sys = NeuralMassBase(p)

    solver = SolverRK45(sys, auto_observe=True)
    solver.integrate()

    data = NMData()
    data.p = p

    data.tv = np.array(solver.tv)
    data.xv = np.array(solver.xv)

    data.save(f"Data/{save_name}_NM")


if (__name__ == "__main__"):
    p = QIFParams()

    p.dt = 1e-4
    p.t0 = -40
    p.t1 = 40
    p.AvgSavestep = 10

    p.N = 10000
    p.x0 = np.zeros(p.N)

    p.J = 15
    p.eta0 = -5
    p.Delta = 1

    p.eta = deterministicLorentzian(p.N, p.eta0, p.Delta)

    p.Iext = stim_step

    save_name = "MPR2015_Fig2a"

    run_QIF(p, save_name)
    p.x0 = np.zeros(2)
    run_NM(p, save_name)

    data_net = QIFData(f"Data/{save_name}_network.dill")
    data_NM = NMData(f"Data/{save_name}_NM.dill")

    plot_dynamics(data_net, data_NM)
    plt.show()
