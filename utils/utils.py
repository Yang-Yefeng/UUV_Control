import numpy as np


def deg2rad(deg: float) -> float:
    """
    :brief:         omit
    :param deg:     degree
    :return:        radian
    """
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    :brief:         omit
    :param rad:     radian
    :return:        degree
    """
    return rad * 180.8 / np.pi

def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)

def T(x):
    return np.tan(x)

def R_x_phi(phi) -> np.ndarray:
    return np.array([[1, 0, 0], [0, C(phi), -S(phi)], [0, S(phi), C(phi)]])


def R_y_theta(theta) -> np.ndarray:
    return np.array([[C(theta), 0, S(theta)], [0, 1, 0], [-S(theta), 0, C(theta)]])


def R_z_psi(psi) -> np.ndarray:
    return np.array([[C(psi), -S(psi), 0], [S(psi), C(psi), 0], [0, 0, 1]])


def vec_2_antisym(v) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def vec_cross_3d(vec1:np.ndarray, vec2:np.ndarray) -> np.ndarray:
    return np.array([vec1[1] * vec2[2] - vec2[1] * vec1[2], vec2[0] * vec1[2] - vec2[2] * vec1[0], vec1[0] * vec2[1] - vec2[0] * vec1[1]])


def matrix_cross_by_vec(m1: np.ndarray, m2: np.ndarray, axis: int = 0) -> np.ndarray:
    res = np.zeros_like(m1)
    if m1.shape != m2.shape:
        print('Func matrix_cross_by_vec -- Wrong input!!!!')
    [row, column] = m1.shape
    if axis == 0:   # 一列一列来
        for i in range(column):
            res[:, i][:] = vec_cross_3d(m1[:, i], m2[:, i])
        return res
    elif axis == 1: # 一行一行来
        for i in range(row):
            res[i, :][:] = vec_cross_3d(m1[i, :], m2[i, :])
        return res
    else:
        return res


def uo_2_ref_angle_throttle(uo: np.ndarray, att: np.ndarray, m: float, g: float, att_max=np.pi/3, dot_att_max=np.pi/2):
    ux = uo[0]
    uy = uo[1]
    uz = uo[2]
    uf = (uz + g) * m / (C(att[0]) * C(att[1]))
    
    asin_phi_d = min(max((ux * np.sin(att[2]) - uy * np.cos(att[2])) * m / uf, -1), 1)
    phi_d = np.arcsin(asin_phi_d)
    
    if att_max is not None:
        phi_d = np.clip(phi_d, -att_max, att_max)
    
    asin_theta_d = min(max((ux * np.cos(att[2]) + uy * np.sin(att[2])) * m / (uf * np.cos(phi_d)), -1), 1)
    theta_d = np.arcsin(asin_theta_d)
    
    if att_max is not None:
        theta_d = np.clip(theta_d, -att_max, att_max)
    
    # print(phi_d * 180 / np.pi, theta_d * 180 / np.pi)
    return phi_d, theta_d, uf
