import numpy as np


class smc_param:
    def __init__(self,
                 dim: int = 6,
                 dt: float = 0.01,
                 k1: np.ndarray = np.zeros(6),
                 k2: np.ndarray = np.zeros(6)):
        self.dim = dim
        self.dt = dt
        self.k1 = k1
        self.k2 = k2


class smc:
    def __init__(self, param: smc_param):
        self.k1 = param.k1
        self.k2 = param.k2
        self.dt = param.dt
        self.dim = param.dim
        self.s = np.zeros(self.dim)
        self.ctrl = np.zeros(self.dim)
    
    @staticmethod
    def sig(x, a, kt=5):
        return np.fabs(x) ** a * np.tanh(kt * x)
    
    def control_update(self,
                       e_eta: np.ndarray,
                       dot_e_eta: np.ndarray,
                       dd_ref: np.ndarray,
                       A_eta: np.ndarray,
                       B_eta: np.ndarray,
                       obs: np.ndarray,
                       e_max: float = np.inf,
                       de_max: float = np.inf):
        e_eta = np.clip(-e_max, e_max, e_eta)
        dot_e_eta = np.clip(-de_max, de_max, dot_e_eta)
        self.s = dot_e_eta + self.k1 * e_eta
        ctrl1 = -B_eta - self.k1 * dot_e_eta + dd_ref
        ctrl2 = -self.k2 * self.sig(self.s, 1) - obs
        self.ctrl = np.dot(np.linalg.inv(A_eta), ctrl1 + ctrl2)
