"""
    Loads and plots the data generated by Simulation.py using matplotlib.
"""

import matplotlib.pyplot as plt
from QIFSystems import QIFData
from QIFSystems.NeuralMass import NMData

import numpy as np


def get_spike_scatter(data, idx=None):
    taui = data.taui
    if (idx is None):
        idx = range(data.p.N)

    xs = []
    ys = []
    for i, ni in enumerate(idx):
        tk = np.array(taui[ni])
        xs.extend(tk)
        ys.extend([i] * len(tk))

    return np.array(xs), np.array(ys)


def plot_dynamics(data_net: QIFData, data_NM: NMData):
    p = data_net.p
    tv = data_net.atv
    vv = data_net.avgvv
    rv = data_net.avgrv

    Iextv = np.array([p.Iext(t) for t in data_NM.tv])
    spike_tv, spike_idx = get_spike_scatter(data_net, np.random.choice(p.N, p.N, replace=False))

    net_kwargs = dict(lw=1, alpha=0.7, color="black")
    NM_kwargs = dict(lw=1, alpha=1, color="red")

    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(data_NM.tv, Iextv, color="black")

    ax[1].scatter(spike_tv, spike_idx, marker=".", color=net_kwargs["color"], alpha=0.5,
                  edgecolors="none", s=2)
    ax[1].set_ylim(0, p.N)
    ax[2].plot(tv, rv, **net_kwargs)
    ax[3].plot(tv, vv, **net_kwargs)

    for k in range(2):
        ax[k + 2].plot(data_NM.tv, data_NM.xv[:, k], **NM_kwargs)

    ax[0].set_xlim(-10, p.t1)
    ax[0].set_ylabel("$I(t)$")
    ax[1].set_ylabel("Neuron index $i$")
    ax[2].set_ylabel("Mean voltage $v(t)$")
    ax[3].set_ylabel("Mean firing rate $r(t)$")
    ax[-1].set_xlabel("Time $t$")

    # ax.legend()
    fig.align_labels()
    plt.tight_layout()


if (__name__ == "__main__"):
    save_name = "MPR2015_Fig2a"

    data_net = QIFData(f"Data/{save_name}_network.dill")
    data_NM = NMData(f"Data/{save_name}_NM.dill")
    plot_dynamics(data_net, data_NM)
    plt.savefig("Results/NetworkQIFExample.png", dpi=300)
    plt.show()