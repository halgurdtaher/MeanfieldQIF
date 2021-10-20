from scipy.integrate import ode
import numpy as np
import NumSim.Observers as Observers


class SolverBase:
    """
        A base class for solving initial value problems of the form

        dx/dt=f(t,x)

        Attributes:
            p (Parameter.Params): Provides integration and system parameters
            sys (DynamicalSystem.DSys): The dynamical system to solve.
            observers (List of Observer.Obs): A list of observer of which the Obs.observe(t,x) method is called after each succesfull timestep.
    """

    def integrate(self):
        pass

    def __init__(self, sys):
        self.p = sys.p
        self.sys = sys
        self.observers = []

    def call_observers(self, t, x):
        """
             Called after each succesfull integration step. Iterates through all observers and calls Obs.Observe(t,x).
         """
        for obs in self.observers:
            if (obs.t0 is None or t >= obs.t0):
                if (obs.t1 is None or t <= obs.t1):
                    obs.observe(t, x)


class SolverRK45(SolverBase):
    """
        A class built around the RK45 solver from scipy.integrate.ode().
        Uses Params objects to define integration and system parameters.
        Reshapes the state variable, such that nd.ndarray of any shape can be used.


        Parameters:
            freq (bool): Frequency of terminal output. Only used of auto_observe=True.

        Attributes:
            p (Parameter.Params): Provides integration and system parameters
            sys (DynamicalSystem.DSys): The dynamical system to solve.
            auto_observe (bool): If true, automatically attaches solution and terminal output observers to the solver.
            tv (np.ndarray): An array of shape (nstep) that contains all time values for all nstep timesteps. Only used if auto_observe=True.
            xv (np.ndarray): An array of shape (nstep,shape(x0)) that contains all state variables for all nstep timesteps. Only used if auto_observe=True.
            verbosity (int): Defines the verbosity for the RK45 solver. Default is 0.
    """

    def __init__(self, sys, auto_observe=False, freq=30):
        SolverBase.__init__(self, sys)

        self.verbosity = 0

        self.xv = None
        self.tv = None

        self._obs_sol_ = None
        self._obs_out_ = None

        self.auto_observe = None

        if (freq > 0):
            self._obs_out_ = Observers.TerminalOutput(self.p, freq=freq)
            self.observers.append(self._obs_out_)
        if (auto_observe):
            self._obs_sol_ = Observers.Solution(self.p)
            self.observers.append(self._obs_sol_)

        self.auto_observe = auto_observe

    def integrate(self, verbose=False, **kwargs):
        """
             Integrates dx/dt = f(x,t) from p.t0 to p.t1 using x(p.t0)=p.x0 as inital condition.

             Parameters:
                 verbose (bool): If true, provides additional information during integration

                 kwargs: Passed to scipy.integrate.ode().set_integrator().

             Returns:
                 solver: The scipy solver.
         """
        p = self.p

        if (np.ndim(p.x0) == 1):
            x0 = p.x0.copy()
            RHS = self.sys.RHS
            call_observers = self.call_observers
        else:
            prod_shape = np.prod(p.x0.shape)
            x0 = p.x0.reshape(prod_shape)

            def RHS(t, x):
                return self.sys.RHS(t, x.reshape(p.x0.shape)).reshape(prod_shape)

            def call_observers(t, x):
                return self.call_observers(t, x.reshape(p.x0.shape))

        solver = ode(RHS)

        max_step = 1

        solver.set_integrator(p.method, nsteps=5e9, rtol=p.rtol, atol=p.atol, verbosity=self.verbosity,
                              max_step=max_step, **kwargs)

        if (verbose):
            print(f"Solving with rtol = {p.rtol}, atol = {self.p.atol}")

        if (p.t1 - p.t0 > 0):
            solver.set_solout(call_observers)
            solver.set_initial_value(x0, p.t0)
            solver.integrate(p.t1)
            if (verbose):
                print("Integration success post:%s" % solver.successful())

        if (self.auto_observe):
            self.xv = np.array(self._obs_sol_.xv)
            self.tv = np.array(self._obs_sol_.tv)

        return solver


class SolveEuler(SolverRK45):

    def __init__(self, sys, auto_observe=False, freq=30):
        SolverRK45.__init__(self, sys, auto_observe, freq)

        self.x = None
        self.t = None

    @staticmethod
    def get_stepper(dt: float, RHS):
        """
            Used to define the explicit Euler method

            Parameters:
                dt (float): The timestep

                RHS (float): The function that defines the right hand side of the initial value problem

            Returns:
                euler_step: A function that increments x by dt*RHS, given t and x (without making a copy).
        """

        def euler_step_func(t, x):
            x[:] += dt * RHS(t, x)

        return euler_step_func

    def integrate(self):
        p = self.p
        RHS = self.sys.RHS

        euler_step = SolveEuler.get_stepper(p.dt, RHS)

        self.t = p.t0
        self.x = p.x0.copy()
        tv = np.arange(p.t0, p.t1 + p.dt, p.dt)

        self.t = p.t0

        for t in tv:
            self.call_observers(t, self.x)
            euler_step(t, self.x)

        if (self.auto_observe):
            self.xv = np.array(self._obs_sol_.xv)
            self.tv = np.array(self._obs_sol_.tv)

        return self.tv, self.xv
