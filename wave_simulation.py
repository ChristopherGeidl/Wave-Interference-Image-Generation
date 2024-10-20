import cupy
import numpy as np
import cupy as cp
import cupyx.scipy.signal
from abc import ABC, abstractmethod


class SceneObject(ABC):
    @abstractmethod
    def render(self, wave_speed_field: cupy.ndarray, dampening_field: cupy.ndarray):
        pass

    @abstractmethod
    def update_field(self, field: cupy.ndarray, t):
        pass


class WaveSimulator2D:
    def __init__(self, w, h, scene_objects):
        self.global_dampening = 1.0
        self.c = cp.ones((h, w), dtype=cp.float32)        # wave speed field (from refractive indices)
        self.d = cp.ones((h, w), dtype=cp.float32)        # dampening field
        self.u = cp.zeros((h, w), dtype=cp.float32)       # field values
        self.u_prev = cp.zeros((h, w), dtype=cp.float32)  # field values of prev frame

        self.laplacian_kernel = cp.array([[0.066, 0.184, 0.066],[0.184, -1.0, 0.184],[0.066, 0.184, 0.066]])

        self.t = 0
        self.dt = 1.0

        self.scene_objects = scene_objects if scene_objects is not None else []

    def reset_time(self):
        self.t = 0.0

    def update_field(self):
        # calculate laplacian using convolution
        laplacian = cupyx.scipy.signal.convolve2d(self.u, self.laplacian_kernel, mode='same', boundary='fill')

        # update field
        v = (self.u - self.u_prev) * self.d * self.global_dampening
        r = (self.u + v + laplacian * (self.c * self.dt)**2)

        self.u_prev[:] = self.u
        self.u[:] = r

        self.t += self.dt

    def update_scene(self):
        # clear wave speed field and dampening field
        self.c.fill(1.0)
        self.d.fill(1.0)

        for obj in self.scene_objects:
            obj.render(self.c, self.d)

        for obj in self.scene_objects:
            obj.update_field(self.u, self.t)

    def get_field(self):
        return self.u