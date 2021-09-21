from collections import deque


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=10, n=30, **kwargs):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._dt = 1.0 / fps
        self._n = n
        self._window = deque(maxlen=self._n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = sum(self._window) * self._dt
            derivative = (self._window[-1] - self._window[-2]) / self._dt
        else:
            integral = 0.0
            derivative = 0.0

        control = 0.0
        control += self._K_P * error
        control += self._K_I * integral
        control += self._K_D * derivative

        return control
