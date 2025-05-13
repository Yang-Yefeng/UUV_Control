import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class data_collector:
    def __init__(self, N: int):
        self.t = np.zeros((N, 1)).astype(float)             # 时间
        self.tau = np.zeros((N, 6)).astype(float)           # 三个力，三个力矩
        self.F_motor = np.zeros((N, 8)).astype(float)       # 8 个电机的力
        self.ref_pos = np.zeros((N, 6)).astype(float)       # 参考位置 包括三维参考位置 + 三维参考姿态角
        self.ref_vel = np.zeros((N, 6)).astype(float)       # 参考速度 包括三维参考速度 + 三维参考姿态角导数
        self.d = np.zeros((N, 6)).astype(float)             # 六轴干扰
        self.d_obs = np.zeros((N, 6)).astype(float)         # 六轴干扰的估计
        self.state = np.zeros((N, 12)).astype(float)        # 状态 x y z vx vy vz phi theta psi p q r
        self.index = 0
        self.name = ['uav_state.csv', 'ref_cmd.csv', 'control.csv', 'observe.csv']
        self.path = os.getcwd() + '/datasave/'
        self.N = N

    def record(self, data: dict):
        if self.index < self.N:
            self.t[self.index][0] = data['time']
            self.tau[self.index] = data['control']
            self.F_motor[self.index] - data['F_motor']
            self.ref_pos[self.index] = data['ref_pos']
            self.ref_vel[self.index] = data['ref_vel']
            self.d[self.index] = data['d']
            self.d_obs[self.index] = data['d_obs']
            self.state[self.index] = data['state']
            self.index += 1

    def package2file(self, path: str):
        pd.DataFrame(np.hstack((self.t, self.state)),
                     columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']). \
            to_csv(path + self.name[0], sep=',', index=False)

        pd.DataFrame(np.hstack((self.t, self.ref_pos, self.ref_vel)),
                     columns=['time', 'ref_x', 'ref_y', 'ref_z', 'ref_vx', 'ref_vy', 'ref_vz', 'ref_phi', 'ref_theta', 'ref_psi', 'ref_dphi', 'ref_dtheta', 'ref_dpsi']). \
            to_csv(path + self.name[1], sep=',', index=False)

        pd.DataFrame(np.hstack((self.t, self.tau, self.F_motor)),
                     columns=['time', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']). \
            to_csv(path + self.name[2], sep=',', index=False)

        pd.DataFrame(np.hstack((self.t, self.d, self.d_obs)),
                     columns=['time', 'dx', 'dy', 'dz', 'dphi', 'dtheta', 'dpsi', 'dx_obs', 'dy_obs', 'dz_obs', 'dphi_obs', 'dtheta_obs', 'dpsi_obs']). \
            to_csv(path + self.name[3], sep=',', index=False)

    def load_file(self, path: str):
        controlData = pd.read_csv(path + 'control.csv', header=0).to_numpy()
        observeData = pd.read_csv(path + 'observe.csv', header=0).to_numpy()
        ref_cmdData = pd.read_csv(path + 'ref_cmd.csv', header=0).to_numpy()
        uav_stateData = pd.read_csv(path + 'uav_state.csv', header=0).to_numpy()

        self.t = controlData[:, 0]
        self.tau, self.F_motor = controlData[:, 1: 7], controlData[:, 7: 15]
        self.ref_pos, self.ref_vel = ref_cmdData[:, 1: 7], ref_cmdData[:, 7: 13]
        self.d, self.d_obs = observeData[:, 1:7], observeData[:, 7:13]
        self.state = uav_stateData[:, 1: 13]

    def plot_pos(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.ref_pos[:, 0], 'red')
        plt.plot(self.t, self.state[:, 0], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('X')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.ref_pos[:, 1], 'red')
        plt.plot(self.t, self.state[:, 1], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('Y')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.ref_pos[:, 2], 'red')
        plt.plot(self.t, self.state[:, 2], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('Z')

    def plot_vel(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.ref_vel[:, 0], 'red')
        plt.plot(self.t, self.state[:, 3], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('vx')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.ref_vel[:, 1], 'red')
        plt.plot(self.t, self.state[:, 4], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('vy')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.ref_vel[:, 2], 'red')
        plt.plot(self.t, self.state[:, 5], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('vz')

    def plot_att(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.ref_pos[:, 3] * 180 / np.pi, 'red')
        plt.plot(self.t, self.state[:, 6] * 180 / np.pi, 'blue')
        plt.grid(True)
        plt.ylim((-90, 90))
        plt.yticks(np.arange(-90, 90, 10))
        plt.xlabel('time(s)')
        plt.title('roll-phi')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.ref_pos[:, 4] * 180 / np.pi, 'red')
        plt.plot(self.t, self.state[:, 7] * 180 / np.pi, 'blue')
        plt.grid(True)
        plt.ylim((-90, 90))
        plt.yticks(np.arange(-90, 90, 10))
        plt.xlabel('time(s)')
        plt.title('pitch-theta')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.ref_pos[:, 5] * 180 / np.pi, 'red')
        plt.plot(self.t, self.state[:, 8] * 180 / np.pi, 'blue')
        plt.grid(True)
        plt.ylim((-180, 180))
        plt.yticks(np.arange(-180, 210, 30))
        plt.xlabel('time(s)')
        plt.title('yaw-psi')

    def plot_force(self):
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.tau[:, 0], 'blue')  # Fx
        plt.grid(True)
        plt.xlabel('time(s)')
        plt.title('Fx')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.tau[:, 1], 'blue')  # Fy
        plt.grid(True)
        plt.xlabel('time(s)')
        plt.title('Fy')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.tau[:, 2], 'blue')  # Fz
        plt.grid(True)
        plt.xlabel('time(s)')
        plt.title('Fz')

    def plot_torque(self):
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.tau[:, 3], 'blue')  # Tx
        plt.grid(True)
        # plt.ylim((-0.3, 0.3))
        # plt.yticks(np.arange(-0.3, 0.3, 0.1))
        plt.xlabel('time(s)')
        plt.title('Tx')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.tau[:, 4], 'blue')  # Ty
        plt.grid(True)
        # plt.ylim((-0.3, 0.3))
        # plt.yticks(np.arange(-0.3, 0.3, 0.1))
        plt.xlabel('time(s)')
        plt.title('Ty')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.tau[:, 5], 'blue')  # Tz
        plt.grid(True)
        # plt.ylim((-0.3, 0.3))
        # plt.yticks(np.arange(-0.3, 0.3, 0.1))
        plt.xlabel('time(s)')
        plt.title('Tz')

    def plot_motor_force(self):
        plt.figure()

        plt.subplot(2, 4, 1)
        plt.plot(self.t, self.F_motor[:, 0], 'blue')  # M1
        plt.xlabel('time(s)')
        plt.title('M1')
        plt.grid(True)

        plt.plot(self.t, self.F_motor[:, 1], 'blue')  # M2
        plt.xlabel('time(s)')
        plt.title('M2')
        plt.grid(True)

        plt.plot(self.t, self.F_motor[:, 2], 'blue')  # M3
        plt.xlabel('time(s)')
        plt.title('M3')
        plt.grid(True)

        plt.plot(self.t, self.F_motor[:, 3], 'blue')  # M4
        plt.xlabel('time(s)')
        plt.title('M4')
        plt.grid(True)

        plt.plot(self.t, self.F_motor[:, 4], 'blue')  # M5
        plt.xlabel('time(s)')
        plt.title('M5')
        plt.grid(True)

        plt.plot(self.t, self.F_motor[:, 5], 'blue')  # M6
        plt.xlabel('time(s)')
        plt.title('M6')
        plt.grid(True)

        plt.plot(self.t, self.F_motor[:, 6], 'blue')  # M7
        plt.xlabel('time(s)')
        plt.title('M7')
        plt.grid(True)

        plt.plot(self.t, self.F_motor[:, 7], 'blue')  # M8
        plt.xlabel('time(s)')
        plt.title('M8')
        plt.grid(True)

    def plot_pos_obs(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.d[:, 0], 'red')
        plt.plot(self.t, self.d_obs[:, 0], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        # plt.ylim((-4, 4))
        plt.title('delta x')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.d[:, 1], 'red')
        plt.plot(self.t, self.d_obs[:, 1], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        # plt.ylim((-4, 4))
        plt.title('delta y')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.d[:, 2], 'red')
        plt.plot(self.t, self.d_obs[:, 2], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        # plt.ylim((-4, 4))
        plt.title('delta z')

    def plot_att_obs(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.d[:, 3], 'red')
        plt.plot(self.t, self.d_obs[:, 3], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        # plt.ylim((-4, 4))
        plt.title('delta phi')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.d[:, 4], 'red')
        plt.plot(self.t, self.d_obs[:, 4], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        # plt.ylim((-4, 4))
        plt.title('delta theta')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.d[:, 5], 'red')
        plt.plot(self.t, self.d_obs[:, 5], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        # plt.ylim((-4, 4))
        plt.title('delta psi')
