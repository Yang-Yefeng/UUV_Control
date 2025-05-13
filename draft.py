import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma
from utils.utils import *
from uuv.bluerov2_heavy import bluerov2_heavy, bluerov2_heavy_param

def f(x, k):
    return np.sign(1 - x) * np.fabs(1 - x) ** k - 1 + x ** k

if __name__ == '__main__':
    # param = bluerov2_heavy_param()
    # np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
    # uuv = bluerov2_heavy(param)
    # print(uuv.T)
    # v1 = np.array([1,2,3,4])
    # v2 = np.array([1, 2, 3, 4])
    # print(np.concatenate((v1,v2)))
    xx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    eta = np.zeros(6)
    eta[:] = xx[[0,1,2,6,7,8]]
    print(eta)