"""
    Solves the QIF initial value problem dV/dt=V^2+eta + J*r, where V and eta are numpy arrays of the same shape.
    Here two QIF neurons are simulated.
    J is the synaptic weigth. At each spike of one of the neurons, the membrane potential V of the other neuron is
    instantaneously increeased by J/N = 50/2 = 25

    Plots the results using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
from NumSim.Solvers import SolveEuler
from QIFSystems import QIFBase, QIFParams, QIFObserverBase, QIFData
from Examples.SimpleQIF.Plot import plot_potential

if (__name__ == "__main__"):
    p = QIFParams()
    p.x0 = np.zeros(2)
    p.eta = np.zeros(2)

    p.N = 2
    p.eta[0] = 20
    p.eta[1] = -4

    p.dt = 1e-4

    p.J = 50

    p.t0 = 0
    p.t1 = 1

    sys = QIFBase(p)
    qobs = QIFObserverBase(p)

    solver = SolveEuler(sys, auto_observe=True)
    solver.observers.append(qobs)
    solver.integrate()

    data = QIFData()
    data.p = p
    data.tv = solver.tv
    data.xv = solver.xv
    data.save("Data/QIF2Example")

    plot_potential(data)
    plt.show()
