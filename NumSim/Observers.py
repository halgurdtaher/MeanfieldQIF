import psutil
from NumSim.Parameters import Params

from timeit import default_timer as timer
import numpy as np
import os

process = psutil.Process(os.getpid())


class ObserverBase:
    """
        Observer base class. During integration, observe() is called for t0<=t<=t1.
    """

    def __init__(self, p: Params, t0=None, t1=None):
        self.name = None

        self.p = p
        self.t0 = t0
        self.t1 = t1

    def observe(self, t, x: np.ndarray):
        pass


class Solution(ObserverBase):
    """
        Stores time values and state variables every save_step succesfull integration steps.
        idx defines the slices of the state variable to be stored.
    """

    def __init__(self, p: Params, t0=None, t_max=None, ):
        ObserverBase.__init__(self, p, t0=t0, t1=t_max)
        self.tv = []
        self.xv = []

        self.total_step = 0
        self.save_step = 1
        self.idx = slice(None)

    def observe(self, t, x):
        if (self.total_step % self.save_step == 0):
            xtmp = x[self.idx].copy()
            self.xv.append(xtmp)
            self.tv.append(t)

        self.total_step += 1

    def clear(self):
        self.xv = []
        self.tv = []

    def __del__(self):
        self.xv = None
        self.tv = None


class TerminalOutput(ObserverBase):
    """
        Prints text to the terminal with information on the integration progress and memory consumption.
        Output is written every 60/freq seconds.
    """

    def __init__(self, p: Params, freq=30, suppress=False, show_mem=True):
        ObserverBase.__init__(self, p)

        self._timer = -1e10

        self.ot = 0

        self.elapsed = 0

        self.mindt = 1e3
        self.avgdt = 0
        self.step = 0

        self.store = False

        self.freq = freq
        self.e_timer = None
        self.suppress = suppress

        self.show_mem = show_mem

        self.eta_sim_t1 = 0
        self.eta_real_t1 = 0
        self.rate = 1

        self.total_step = -1
        self.skip_steps = 20

        self.name_string = " "
        if (self.name is not None):
            self.name_string = self.name

    def observe(self, t, x):
        self.total_step += 1
        if (self.total_step % self.skip_steps != 0):
            return

        if (self.e_timer is None):
            self.e_timer = timer()
        nt = timer()
        self.elapsed = timer() - self.e_timer
        dt = t - self.ot
        p = self.p

        if (dt > 0):
            self.mindt = min(dt, self.mindt)
        if ((nt - self._timer) > 1.0 / self.freq * 60.0 or self._timer == -1e10 or self.p.t1 - t < 1e-6):

            T = p.t1 - p.t0
            remaining = (t - p.t0) / T
            if (remaining > 0):
                rate = remaining / self.elapsed
                t_done = 1.0 / rate
                eta = t_done - self.elapsed

                m, s = divmod(eta, 60)
                h, m = divmod(m, 60)
                eta_str = "%02d:%02d:%02d" % (h, m, s)

                mem_bytes = process.memory_info().rss
                mem_mega_bytes = mem_bytes / 1e6

                self.printt(
                    "%st=%g/%g | mindt=%1g | mem: %1g MB | elapsed: %0gs | eta: %s" % (
                        self.name_string, t, p.t1, self.mindt, mem_mega_bytes, self.elapsed,
                        eta_str))
                self._timer = nt
                self.mindt = 1000

        self.ot = t

    def printt(self, txt):
        if (not self.suppress):
            print(txt)
