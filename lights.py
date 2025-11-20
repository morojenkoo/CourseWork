from dataclasses import dataclass
from ray import Vector3


@dataclass
class Light:
    position: Vector3
    color: Vector3 = None
    intensity: float = 1.0

    def __init__(self, position: Vector3, color: Vector3 = None, intensity: float = 1.0):
        self.position = position
        self.color = color if color is not None else Vector3(1.0, 1.0, 1.0)  # белый по умолчанию
        self.intensity = intensity