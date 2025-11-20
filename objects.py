from abc import ABC, abstractmethod
from ray import Ray, Intersection, Vector3
from materials import Material
import numpy as np

class Object(ABC):
    def __init__(self, material: Material):
        self.material = material

    @abstractmethod
    def intersect(self, ray: Ray) -> Intersection:
        #Возвращает Intersection или Intersection.none()#
        pass


class Sphere(Object):
    def __init__(self, center: Vector3, radius: float, material: Material):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray) -> Intersection:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return Intersection.none()

        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        t = None
        if t1 > 0.001:
            t = t1
        elif t2 > 0.001:
            t = t2
        else:
            return Intersection.none()

        point = ray.point_at_parameter(t)
        normal = (point - self.center).normalize()

        return Intersection(t, point, normal, self, self.material)

# Позже добавим другие объекты:
# class Triangle(Object):
# class Plane(Object):
# class Mesh(Object):