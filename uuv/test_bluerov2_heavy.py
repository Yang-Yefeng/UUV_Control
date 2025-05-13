import numpy as np
import os, sys, platform, datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from bluerov2_heavy import bluerov2_heavy, bluerov2_heavy_param
from utils.collector import data_collector
from utils.ref_cmd import *

SAVE = False
cur_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
cur_path = os.path.dirname(os.path.abspath(__file__))
windows = platform.system().lower() == 'windows'
if windows:
    new_path = cur_path + '\\..\\datasave\\uuv-' + cur_time + '/'
else:
    new_path = cur_path + '/../datasave/uuv-' + cur_time + '/'


if __name__ == '__main__':
    uuv_param = bluerov2_heavy_param(time_max=20.0)
    uuv_param.ignore_Coriolis = False

    uuv = bluerov2_heavy(uuv_param)
    data_record = data_collector(N=int(uuv.time_max / uuv.dt))

    ra = np.array([0, 0, 0, 0, 0, 0])   # 振幅 x y z phi theta psi
    rp = np.array([1, 1, 1, 1, 1, 1])   # 周期
    rba = np.array([0, 0, 0, 0, 0, 0])  # 初始位置偏移
    rbp = np.array([0, 0, 0, 0, 0, 0])  # 初始相位偏移

    # print(uuv.MRB)

    while uuv.time < uuv.time_max - uuv.dt / 2:
        if uuv.n % int(1 / uuv.dt) == 0:
            print('time: %.2f s.' % (uuv.n / int(1 / uuv.dt)))

        tau = np.array([2., 2., 0, 0, 0, 0]).astype(float)
        dis = generate_uncertainty(uuv.time, is_ideal=True)
        ref, d_ref, dd_ref = ref_uuv_circle(uuv.time, ra, rp, rba, rbp)

        uuv.rk44(action=tau, dis=dis)

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
