import os
import sys
import datetime
import matplotlib.pyplot as plt
import platform
import pandas as pd

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from controller.smc import smc_param, smc
from uuv.bluerov2_heavy import bluerov2_heavy_param, bluerov2_heavy
from utils.ref_cmd import *
from utils.utils import *
from utils.collector import data_collector


IS_IDEAL = True
USE_OBS = True
SAVE = False
cur_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
cur_path = os.path.dirname(os.path.abspath(__file__))
windows = platform.system().lower() == 'windows'
if windows:
    new_path = cur_path + '\\..\\datasave\\smc_BlueROV2Heavy-' + cur_time + '/'
else:
    new_path = cur_path + '/../datasave/smc_BlueROV2Heavy-' + cur_time + '/'

DT = 0.01
uuv_param = bluerov2_heavy_param(time_max=40)
ctrl_param = smc_param(
    dim=6,
    dt=DT,
    k1=np.array([5, 5, 100, 0.1, 0.1, 0.1]).astype(float),
    k2=np.array([0.,0.,0.,0.,0.,0.,]).astype(float))


if __name__ == '__main__':
    uuv = bluerov2_heavy(uuv_param)
    smc_ctrl = smc(ctrl_param)
    data_record = data_collector(N=int(uuv.time_max / uuv.dt))

    ra = np.array([0, 0, 0, 0, 0, 0])  # 振幅 x y z phi theta psi
    rp = np.array([1, 1, 1, 1, 1, 1])  # 周期
    rba = np.array([0, 0, 10, 0, 0, 0])  # 初始位置偏移
    rbp = np.array([0, 0, 0, 0, 0, 0])  # 初始相位偏移

    while uuv.time < uuv.time_max - uuv.dt / 2:
        if uuv.n % int(1 / uuv.dt) == 0:
            print('time: %.2f s.' % (uuv.n / int(1 / uuv.dt)))

        dis = generate_uncertainty(uuv.time, is_ideal=IS_IDEAL)
        ref, d_ref, dd_ref = ref_uuv_circle(uuv.time, ra, rp, rba, rbp)

        e = uuv.eta - ref
        de = uuv.uuv_dot_eta_cb() - d_ref

        '''observer'''
        if USE_OBS:
            obs = np.zeros(6)
        else:
            obs = np.zeros(6)
        '''observer'''

        '''control'''
        smc_ctrl.control_update(e_eta=e,
                                dot_e_eta=de,
                                dd_ref=dd_ref,
                                A_eta=uuv.A_eta(),
                                B_eta=uuv.B_eta(),
                                obs=obs)
        '''control'''
        # smc_ctrl.ctrl = np.zeros(6)
        uuv.rk44(action=smc_ctrl.ctrl, dis=dis)

        data_block = {'time': uuv.time,
                      'control': uuv.tau,
                      'F_motor': uuv.F_motor,
                      'ref_pos': ref,
                      'ref_vel': d_ref,
                      'd': np.zeros(6),
                      'd_obs': np.zeros(6),
                      'state': uuv.uuv_state_cb()}
        data_record.record(data=data_block)

    '''datasave'''
    if SAVE:
        os.mkdir(new_path)
        data_record.package2file(new_path)
    '''datasave'''

    data_record.plot_pos()
    data_record.plot_att()

    plt.show()
