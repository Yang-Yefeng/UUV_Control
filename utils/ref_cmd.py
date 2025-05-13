import numpy as np


def ref_uuv_circle(time: float, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
    """
    :param time:        time
    :param amplitude:   amplitude
    :param period:      period
    :param bias_a:      amplitude bias
    :param bias_phase:  phase bias
    :return:            reference position and yaw angle and their 1st - 2nd derivatives
    """
    # _r = np.zeros_like(amplitude)
    # _dr = np.zeros_like(amplitude)
    # _ddr = np.zeros_like(amplitude)
    w = 2 * np.pi / period
    _r = amplitude * np.sin(w * time + bias_phase) + bias_a
    _dr = amplitude * w * np.cos(w * time + bias_phase)
    _ddr = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
    return _r, _dr, _ddr


def ref_uuv_Bernoulli_pos(time: float, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
    w = 2 * np.pi / period
    _r = np.zeros(3)
    _dr = np.zeros(3)
    _ddr = np.zeros(3)

    _r[0] = amplitude[0] * np.cos(w[0] * time + bias_phase[0]) + bias_a[0]
    _r[1] = amplitude[1] * np.sin(2 * w[1] * time + bias_phase[1]) / 2 + bias_a[1]
    _r[2] = amplitude[2] * np.sin(w[2] * time + bias_phase[2]) + bias_a[2]

    _dr[0] = -amplitude[0] * w[0] * np.sin(w[0] * time + bias_phase[0])
    _dr[1] = amplitude[1] * w[1] * np.cos(2 * w[1] * time + bias_phase[1])
    _dr[2] = amplitude[2] * w[2] * np.cos(w[2] * time + bias_phase[2])

    _ddr[0] = -amplitude[0] * w[0] ** 2 * np.cos(w[0] * time + bias_phase[0])
    _ddr[1] = - 2 * amplitude[1] * w[1] ** 2 * np.sin(2 * w[1] * time + bias_phase[1])
    _ddr[2] = -amplitude[2] * w[2] ** 2 * np.sin(w[2] * time + bias_phase[2])

    return _r, _dr, _ddr


def generate_uncertainty(time: float, is_ideal: bool = False) -> np.ndarray:
    """
    :param time:        time
    :param is_ideal:    ideal or not
    :return:            Fdx, Fdy, Fdz, dp, dq, dr
    """
    if is_ideal:
        return np.array([0, 0, 0, 0, 0, 0]).astype(float)
    else:
        T = 5
        w = 2 * np.pi / T
        phi0 = 0.
        if time < 10:
            Fdx = 0.9 * np.sin(w * time + phi0) - 1.5 * np.cos(2 * w * time + phi0)
            Fdy = 1.3 * np.sin(0.5 * w * time + phi0) + 0.7 * np.cos(w * time + phi0)
            Fdz = 1.5 * np.sin(w * time + phi0) - 1.5 * np.cos(w * time + phi0)
            dp = 2 * np.sin(w * time + phi0) + 2.5 * np.cos(0.5 * w * time + phi0)
            dq = 1 * np.sin(w * time + phi0) + 2.5 * np.cos(0.5 * w * time + phi0)
            dr = 1.5 * np.sin(w * time + phi0) - 2 * np.cos(0.5 * w * time + phi0)
        elif 10 <= time < 20:
            Fdx = -1.0
            Fdy = 1.0
            Fdz = 2.0
            dp = 1.5 * np.sin(np.sin(w * (time - 10) + phi0))
            dq = 1.7 * np.sin(np.cos(w * (time - 10) + phi0))
            dr = 2.5 * np.cos(np.sin(w * (time - 10) + phi0))
        elif 20 <= time < 30:
            Fdx = - 3.0 - np.tanh(0.1 * (25 - time))
            Fdy = - 2.0 - np.tanh(0.2 * (25 - time))
            Fdz = - 1.0 + np.tanh(0.25 * (25 - time))
            dp = 0.0
            dq = 1.5
            dr = -1.0
        elif 30 <= time < 40:
            # Fdx = np.sqrt(time - 30) + 1.5 * np.cos(np.sin(np.pi * (time - 30)))
            # Fdy = 0.5 * np.sqrt(time - 30) + 0.5 * np.cos(np.sin(np.pi * (time - 30)))
            # Fdz = 1.5 * np.sqrt(time - 30) - 1.0 * np.cos(np.sin(np.pi * (time - 30)))
            Fdx = 0.5 * np.cos(np.cos(np.pi * (time - 30)))
            Fdy = 0.8 * np.sin(np.sin(np.pi * (time - 30)))
            Fdz = -2.0 * np.cos(np.sin(np.pi * (time - 30)))
            dp = np.sqrt(time - 30) + 1.5 * np.cos(np.sin(np.pi * (time - 30)))
            dq = 0.5 * np.sqrt(time - 30) + 0.5 * np.cos(1.5 * np.sin(np.pi * (time - 30)))
            dr = 1.5 * np.sqrt(time - 30) - 1.0 * np.cos(0.5 * np.sin(np.pi * (time - 30)))
        else:
            Fdx = 0.
            Fdy = -1.
            Fdz = 2.
            dp = 0.5
            dq = 0.0
            dr = 1.0

        return np.array([Fdx, Fdy, Fdz, dp, dq, dr])
