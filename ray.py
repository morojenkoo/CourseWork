import numpy as np
from dataclasses import dataclass

import math


class Vector3:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return Vector3(0, 0, 0)

    def to_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class Ray:
    origin: Vector3
    direction: Vector3

    def point_at_parameter(self, t: float) -> Vector3:
        return self.origin + self.direction * t


@dataclass
class Intersection:
    t: float  # расстояние до пересечения
    point: Vector3  # точка пересечения
    normal: Vector3  # нормаль в точке пересечения
    obj: 'Object'  # ссылка на объект
    material: 'Material'  # материал объекта

    @classmethod
    def none(cls):
        return cls(float('inf'), None, None, None, None)

    @property
    def happened(self):
        return self.t < float('inf')