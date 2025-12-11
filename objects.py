from abc import ABC, abstractmethod
from ray import Ray, Intersection, Vector3
from materials import Material
import math
import numpy as np


class Object(ABC):
    @abstractmethod
    def __init__(self, material: Material):
        self.material = material

    @abstractmethod
    def intersect(self, ray: Ray) -> Intersection:
        #Возвращает Intersection или Intersection.none()
        pass


class Sphere(Object):
    def __init__(self, center: Vector3, radius: float, material: Material):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray) -> Intersection:
        oc = ray.origin - self.center
        b = (oc.dot(ray.direction))
        c = oc.dot(oc) - self.radius * self.radius

        discriminant = b * b - c
        if discriminant < 0:
            return Intersection.none()

        sqrt_d = math.sqrt(discriminant)
        t = (-b - sqrt_d)
        if t <= 0.001:
            t = (-b + sqrt_d)
            if t <= 0.001:
                return Intersection.none()

        point = ray.point_at_parameter(t)
        normal = point - self.center
        normal = normal.normalize()
        return Intersection(t, point, normal, self, self.material)

