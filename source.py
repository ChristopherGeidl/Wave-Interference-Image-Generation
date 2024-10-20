import wave_simulation
import cupy

class PointSource(wave_simulation.SceneObject):
    def __init__(self, x, y, frequency, amplitude=1.0, phase=0):
        self.x = x
        self.y = y
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def render(self, wave_speed_field, dampening_field):
        pass

    def update_field(self, field, t):
        v = cupy.sin(self.phase + self.frequency * t) * self.amplitude
        field[self.y, self.x] = v