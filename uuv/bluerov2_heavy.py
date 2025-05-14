import numpy as np
from typing import Union
from utils.utils import *


class bluerov2_heavy_param:
    def __init__(self, time_max: float = 10.0):
        self.dt = 0.01
        self.time_max = time_max
        self.ignore_Coriolis = False

        self.m = 13.5  # 质量
        self.g = 9.82  # 重力加速度
        self.rho = 1026  # 水的密度
        self.delta = 0.0134  # uuv 排出的水的体积
        self.Ix = 0.26  # x 轴转动惯量
        self.Iy = 0.23  # y 轴转动惯量
        self.Iz = 0.37  # z 轴转动惯量
        self.g_c = np.array([0, 0, 0]).astype(float)  # uuv 的重心坐标
        self.b_c = np.array([0, 0, -0.01]).astype(float)  # uuv 的几何中心坐标

        '''MA 附加质量矩阵'''
        # self.Xdu = 6.36
        # self.Ydv = 7.12
        # self.Zdw = 18.68
        # self.Kdp = 0.189
        # self.Mdq = 0.135
        # self.Ndr = 0.222
        self.Xdu = 5.5
        self.Ydv = 12.7
        self.Zdw = 14.57
        self.Kdp = 0.12
        self.Mdq = 0.12
        self.Ndr = 0.12
        '''MA 附加质量矩阵'''

        '''D 和 Dn 矩阵中涉及的系数'''
        # self.Xu = 13.7
        # self.Xuu = 141.0
        # self.Yv = 0.
        # self.Yvv = 217.0
        # self.Zw = 33.0
        # self.Zww = 190.0
        # self.Kp = 0.
        # self.Kpp = 1.19
        # self.Mq = 0.8
        # self.Mqq = 0.47
        # self.Nr = 0.
        # self.Nrr = 1.5
        self.Xu = 4.03
        self.Yv = 6.22
        self.Zw = 5.18
        self.Kp = 0.07
        self.Mq = 0.07
        self.Nr = 0.07
        self.Xuu = 18.18
        self.Yvv = 21.66
        self.Zww = 36.99
        self.Kpp = 1.55
        self.Mqq = 1.55
        self.Nrr = 1.55
        '''D 和 Dn 矩阵中涉及的系数'''

        '''动力分配部分涉及的系数'''
        self.N_m = 8  # 电机数量
        '''
            r 代表 8 个电机的安装位置向量, 每一行代表一个向量, 第一个元素代表角度, 第二个元素代表安装平面的基准向量
            例如, r1: [0, 0.156, 0.111, 0.085] 就是 J(0) * [0.156, 0.111, 0.085].T, 其中
            J(a) = [cos(a), -sin(a), 0; sin(a), cos(a), 0; 0, 0, 1] 为旋转矩阵
        '''
        self.r = [[0, 0.156, 0.111, 0.085],
                  [5.05, 0.156, 0.111, 0.085],
                  [1.91, 0.156, 0.111, 0.085],
                  [np.pi, 0.156, 0.111, 0.085],
                  [0, 0.120, 0.218, 0],
                  [4.15, 0.120, 0.218, 0],
                  [1.01, 0.120, 0.218, 0],
                  [np.pi, 0.120, 0.218, 0]]
        '''
            eps 代表 8 个电机的安装方向向量, 每一行代表一个向量, 第一个元素代表角度, 第二个元素代表安装平面的基准向量
            例如, r1: [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0] 就是 J(0) * [1 / np.sqrt(2), -1 / np.sqrt(2), 0].T, 其中
            J(a) = [cos(a), -sin(a), 0; sin(a), cos(a), 0; 0, 0, 1] 为旋转矩阵
        '''
        self.eps = [[0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0.5 * np.pi, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0.75 * np.pi, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [np.pi, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [0, 0, 0, -1],
                    [0, 0, 0, -1],
                    [0, 0, 0, -1],
                    [0, 0, 0, -1]]
        '''动力分配部分涉及的系数'''


class bluerov2_heavy:
    def __init__(self, param: bluerov2_heavy_param):
        """
        :param param:
        """
        '''状态变量'''
        self.eta = np.zeros(6).astype(float)  # x y z phi theta psi           位置 surge sway herve 角度 roll pitch yaw
        self.nu = np.zeros(6).astype(float)  # u v w p q r                   速度 角速度
        self.tau = np.zeros(6).astype(float)  # Fx Fy Fz tau_x tau_y tau_z    力 力矩
        '''状态变量'''

        self.dt = param.dt
        self.time_max = param.time_max
        self.ignore_Coriolis = param.ignore_Coriolis

        self.m = param.m
        self.g = param.g
        self.rho = param.rho
        self.delta = param.delta
        self.Ix = param.Ix
        self.Iy = param.Iy
        self.Iz = param.Iz
        self.g_c = param.g_c
        self.b_c = param.b_c
        self.W = self.m * self.g
        self.B = self.rho * self.g * self.delta

        '''MA 和 CA 矩阵中涉及的系数'''
        self.Xdu = param.Xdu
        self.Ydv = param.Ydv
        self.Zdw = param.Zdw
        self.Kdp = param.Kdp
        self.Mdq = param.Mdq
        self.Ndr = param.Ndr
        '''MA 矩阵'''

        '''D 和 Dn 矩阵中涉及的系数'''
        self.Xu = param.Xu
        self.Xuu = param.Xuu
        self.Yv = param.Yv
        self.Yvv = param.Yvv
        self.Zw = param.Zw
        self.Zww = param.Zww
        self.Kp = param.Kp
        self.Kpp = param.Kpp
        self.Mq = param.Mq
        self.Mqq = param.Mqq
        self.Nr = param.Nr
        self.Nrr = param.Nrr
        '''D 和 Dn 矩阵中涉及的系数'''

        '''动力分配部分涉及的系数'''
        self.N_m = param.N_m
        '''
            r 代表 8 个电机的安装位置向量, 每一行代表一个向量, 第一个元素代表角度, 第二个元素代表安装平面的基准向量
            例如, r1: [0, 0.156, 0.111, 0.085] 就是 J(0) * [0.156, 0.111, 0.085].T, 其中
            J(a) = [cos(a), -sin(a), 0; sin(a), cos(a), 0; 0, 0, 1] 为旋转矩阵
        '''
        self.r_raw = [[0, 0.156, 0.111, 0.085],
                      [5.0463, 0.156, 0.111, 0.085],
                      [1.9047, 0.156, 0.111, 0.085],
                      [np.pi, 0.156, 0.111, 0.085],
                      [0, 0.120, 0.218, 0],
                      [4.15, 0.120, 0.218, 0],
                      [1.01, 0.120, 0.218, 0],
                      [np.pi, 0.120, 0.218, 0]]
        self.r = np.zeros((3, self.N_m))

        '''
            eps 代表 8 个电机的安装方向向量, 每一行代表一个向量, 第一个元素代表角度, 第二个元素代表安装平面的基准向量
            例如, r1: [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0] 就是 J(0) * [1 / np.sqrt(2), -1 / np.sqrt(2), 0].T, 其中
            J(a) = [cos(a), -sin(a), 0; sin(a), cos(a), 0; 0, 0, 1] 为旋转矩阵
        '''
        self.eps_raw = [[0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                        [0.5 * np.pi, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                        [1.5 * np.pi, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                        [np.pi, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                        [0, 0, 0, -1],
                        [0, 0, 0, -1],
                        [0, 0, 0, -1],
                        [0, 0, 0, -1]]
        self.eps = np.zeros((3, self.N_m))

        for i in range(self.N_m):
            self.r[:, i][:] = np.dot(self.J3(self.r_raw[i][0]), self.r_raw[i][1:4])[:]
            self.eps[:, i][:] = np.dot(self.J3(self.eps_raw[i][0]), self.eps_raw[i][1:4])[:]
        '''动力分配部分涉及的系数'''

        self.J_eta = np.zeros((6, 6))  # 广义坐标变换矩阵
        self.Ic = np.diag([self.Ix, self.Iy, self.Iz])  # 惯性张量矩阵

        self.MRB = np.vstack((np.hstack((self.m * np.eye(3), np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), self.Ic))))  # constant
        self.MA = np.diag([self.Xdu, self.Ydv, self.Zdw, self.Kdp, self.Mdq, self.Ndr])  # constant
        self.M = self.MRB + self.MA  # 广义质量矩阵
        self.M_inv = np.linalg.inv(self.M)  # 广义质量矩阵的逆

        self.T = np.vstack((self.eps, matrix_cross_by_vec(self.r, self.eps, axis=0)))  # constant，动力分配矩阵
        self.perinv_T = np.dot(np.linalg.inv(np.dot(self.T.T, self.T)), self.T.T)

        self.CRB = np.zeros((6, 6))
        self.CA = np.zeros((6, 6))  #
        self.C = self.CRB + self.CA  # 广义科氏力矩阵

        self.D0 = np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])  # 线性阻尼矩阵系数
        self.Dn = np.zeros((6, 6))  # 非线性阻尼矩阵
        self.D = self.D0 + self.Dn

        self.g_eta = np.zeros(6)  # 重力与浮力组成的等效项
        self.F_motor = np.zeros(self.N_m)  # 8个电机的推力

        self.cal_J_eta(self.uuv_att_cb())
        self.cal_C(self.nu)
        self.cal_D(self.nu)
        self.cal_g_eta(self.uuv_att_cb())

        # self.F_max = self.cal_F(1.0)  # 电机最大推力 30.4
        # self.F_min = self.cal_F(-1.0)  # 电机最小推力 -30.4
        self.F_max = 52
        self.F_min = -52

        self.time = 0.
        self.n = 1
        
    def cal_Motor_F_with_sat(self, ctrl: np.ndarray, ideal: bool = False) -> np.ndarray:
        F = np.dot(self.perinv_T, ctrl)
        if ideal:
            self.F_motor = F
            return ctrl
        else:
            self.F_motor = np.clip(F, self.F_min, self.F_max)
            return np.dot(self.T, self.F_motor)
    
    @staticmethod
    def J1(att: Union[np.ndarray, list]) -> np.ndarray:
        return np.dot(R_z_psi(att[2]), np.dot(R_y_theta(att[1]), R_x_phi(att[0])))

    @staticmethod
    def J2(att: Union[np.ndarray, list]) -> np.ndarray:
        return np.array([[1, S(att[0]) * T(att[1]), C(att[0]) * T(att[1])],
                         [0, C(att[0]), -S(att[0])],
                         [0, S(att[0]) / C(att[1]), C(att[0]) / C(att[1])]])

    @staticmethod
    def dot_J1(att: Union[np.ndarray, list], dot_att: Union[np.ndarray, list]):
        res = np.zeros((3, 3))
        [phi, theta, psi] = att
        [dphi, dtheta, dpsi] = dot_att
        res[0, 0] = -S(psi) * dpsi * C(theta) - C(psi) * S(theta) * dtheta
        res[0, 1] = -C(psi) * dpsi * C(phi) + S(psi) * S(phi) * dphi - S(psi) * dpsi * S(theta) * S(phi) + C(psi) * C(theta) * dtheta * S(phi) + C(psi) * S(theta) * C(phi) * dphi
        res[0, 2] = C(psi) * dpsi * S(phi) + S(psi) * C(phi) * dphi - S(psi) * dpsi * C(phi) * S(theta) - C(psi) * S(phi) * dphi * S(theta) + C(psi) * C(phi) * C(theta) * dtheta
        res[1, 0] = C(psi) * dpsi * C(theta) - S(psi) * S(theta) * dtheta
        res[1, 1] = -S(psi) * dpsi * C(phi) - C(psi) * S(phi) * dphi + C(phi) * dphi * S(theta) * S(psi) + S(phi) * C(theta) * dtheta * S(psi) + S(phi) * S(theta) * C(psi) * dpsi
        res[1, 2] = S(psi) * dpsi * S(phi) - C(psi) * C(phi) * dphi + C(theta) * dtheta * S(psi) * C(phi) + S(theta) * C(psi) * dpsi * C(phi) - S(theta) * S(psi) * S(phi) * dphi
        res[2, 0] = -C(theta) * dtheta
        res[2, 1] = -S(theta) * dtheta * S(phi) + C(theta) * C(phi) * dphi
        res[2, 2] = -S(theta) * dtheta * C(phi) - C(theta) * S(phi) * dphi
        return res

    @staticmethod
    def dot_J2(att: Union[np.ndarray, list], dot_att: Union[np.ndarray, list]):
        res = np.zeros((3, 3))
        [phi, theta, _] = att
        [dphi, dtheta, _] = dot_att
        res[0, 0] = 1.
        res[0, 1] = C(phi) * dphi * T(theta) + S(phi) * dtheta / (C(theta) ** 2)
        res[0, 2] = -S(phi) * dphi * T(theta) + C(phi) * dtheta / (C(theta) ** 2)
        res[1, 0] = 0.
        res[1, 1] = -S(phi) * dphi
        res[1, 2] = -C(phi) * dphi
        res[2, 0] = 0.
        res[2, 1] = (C(phi) * dphi * C(theta) + S(phi) * S(theta) * dtheta) / (C(theta) ** 2)
        res[2, 2] = (-S(phi) * dphi * C(theta) + C(phi) * S(theta) * dtheta) / (C(theta) ** 2)
        return res

    def cal_dot_J_eta(self, att: Union[np.ndarray, list], dot_att: Union[np.ndarray, list]):
        return np.vstack((np.hstack((self.dot_J1(att, dot_att), np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), self.dot_J2(att, dot_att)))))

    @staticmethod
    def J3(alpha) -> np.ndarray:
        return R_z_psi(alpha)

    def cal_J_eta(self, att: Union[np.ndarray, list]):
        self.J_eta[0:3, 0:3] = self.J1(att)[:]
        self.J_eta[3:6, 3:6] = self.J2(att)[:]

    def cal_CRB(self, nu: np.ndarray):
        crb11 = np.zeros((3, 3))
        crb12 = -self.m * vec_2_antisym(nu[0:3])
        crb21 = crb12.copy()
        crb22 = -vec_2_antisym([self.Ix * nu[3], self.Iy * nu[4], self.Iz * nu[5]])
        self.CRB = np.vstack((np.hstack((crb11, crb12)), np.hstack((crb21, crb22))))

    def cal_CA(self, nu: np.ndarray):
        ca11 = np.zeros((3, 3))
        ca12 = vec_2_antisym(nu[0:3] * np.array([self.Xdu, self.Ydv, self.Zdw]))
        ca21 = ca12.copy()
        ca22 = vec_2_antisym(nu[3:6] * np.array([self.Kdp, self.Mdq, self.Ndr]))
        self.CA = np.vstack((np.hstack((ca11, ca12)), np.hstack((ca21, ca22))))

    def cal_C(self, nu: np.ndarray):
        if self.ignore_Coriolis:
            self.C = np.zeros((6, 6))
        else:
            self.cal_CRB(nu)
            self.cal_CA(nu)
            self.C = self.CRB + self.CA

    def cal_Dn(self, nu: np.ndarray):
        v = np.array([self.Xuu, self.Yvv, self.Zww, self.Kpp, self.Mqq, self.Nrr])
        self.Dn = np.diag(v * np.fabs(nu))

    def cal_D(self, nu: np.ndarray):
        self.cal_Dn(nu)
        self.D = self.D0 + self.Dn

    def cal_g_eta(self, att: Union[np.ndarray, list]):
        phi = att[0]
        theta = att[1]
        # self.W = self.B
        self.g_eta = np.array([(self.W - self.B) * S(theta),
                               -(self.W - self.B) * C(theta) * S(phi),
                               -(self.W - self.B) * C(theta) * C(phi),
                               self.b_c[1] * self.B * C(theta) * C(phi) - self.b_c[2] * C(theta) * S(phi),
                               -self.b_c[2] * self.B * S(theta) - self.b_c[0] * self.B * C(theta) * C(phi),
                               self.b_c[0] * self.B * C(theta) * S(phi) + self.b_c[1] * self.B * S(theta)])

    @staticmethod
    def cal_F(v):
        """
        :func:      通过 PWM 占空比计算推力
        :param v:   PWM 占空比 [-1, 1]
        :return:
        """
        return -140.3 * v ** 9 + 389.9 * v ** 7 - 404.1 * v ** 5 + 176.0 * v ** 3 + 8.9 * v

    def uuv_state_cb(self) -> np.ndarray:
        return np.concatenate((self.eta[0:3], self.nu[0:3], self.eta[3:6], self.nu[3:6]))  # x y z vx vy vz phi theta psi p q r

    def uuv_pos_vel_cb(self) -> np.ndarray:
        return np.concatenate((self.eta[0:3], self.nu[0:3]))

    def uuv_att_pqr_cb(self) -> np.ndarray:
        return np.concatenate((self.eta[3:6], self.nu[3:6]))

    def uuv_pos_cb(self) -> np.ndarray:
        return self.eta[0:3]

    def uuv_vel_cb(self):
        return self.nu[0:3]

    def uuv_att_cb(self):
        return self.eta[3:6]

    def uuv_dot_eta_cb(self):
        self.cal_J_eta(self.uuv_att_cb())
        return np.dot(self.J_eta, self.nu)

    def uuv_dot_att_cb(self):
        self.cal_J_eta(self.uuv_att_cb())
        dot_eta = np.dot(self.J_eta, self.nu)
        return dot_eta[3:6]

    def uuv_pqr_cb(self):
        return self.nu[3:6]

    def set_state(self, xx: np.ndarray):
        self.eta[:] = xx[[0, 1, 2, 6, 7, 8]]
        self.nu[:] = xx[[3, 4, 5, 9, 10, 11]]

    def A_eta(self):
        self.cal_J_eta(self.uuv_att_cb())
        return np.dot(self.J_eta, self.M_inv)

    def B_eta(self):
        dJ = self.cal_dot_J_eta(self.uuv_att_cb(), self.uuv_dot_att_cb())
        self.cal_C(self.nu)
        self.cal_D(self.nu)
        self.cal_g_eta(self.uuv_att_cb())
        # t1 = np.dot(self.J_eta, np.dot(self.M_inv, np.dot(self.C + self.D, self.nu) + self.g_eta))
        return np.dot(dJ, self.nu) - np.dot(self.J_eta, np.dot(self.M_inv, np.dot(self.C + self.D, self.nu) + self.g_eta))

    def ode(self, xx: np.ndarray, dis: np.ndarray):
        """
        :param xx:      进入微分方程的状态
        :param dis:     干扰
        :return:        状态的导数
        """
        [_x, _y, _z, _vx, _vy, _vz, _phi, _theta, _psi, _p, _q, _r] = xx[0:12]
        _nu = np.array([_vx, _vy, _vz, _p, _q, _r])
        self.cal_J_eta([_phi, _theta, _psi])  # 更新坐标变换矩阵
        self.cal_g_eta([_phi, _theta, _psi])  # 更新浮力向量
        self.cal_C(_nu)
        self.cal_D(_nu)

        [dx, dy, dz, dphi, dtheta, dpsi] = np.dot(self.J_eta, _nu)
        [dvx, dvy, dvz, dp, dq, dr] = np.dot(self.M_inv, self.tau + dis - self.g_eta - np.dot(self.D, _nu) - np.dot(self.C, _nu))
        return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])

    def rk44(self, action: np.ndarray, dis: np.ndarray):
        self.tau = action
        h = self.dt
        xx = self.uuv_state_cb()
        K1 = h * self.ode(xx, dis)
        K2 = h * self.ode(xx + K1 / 2, dis)
        K3 = h * self.ode(xx + K2 / 2, dis)
        K4 = h * self.ode(xx + K3, dis)
        xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        self.set_state(xx)
        self.time += self.dt
        if self.eta[5] > np.pi:  # 如果角度超过 180 度
            self.eta[5] -= 2 * np.pi
        if self.eta[5] < -np.pi:  # 如果角度小于 -180 度
            self.eta[5] += 2 * np.pi
        self.n += 1
