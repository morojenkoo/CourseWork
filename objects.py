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


class Cone(Object):
    def __init__(self, apex, height, radius, material):
        super().__init__(material)
        self.apex = apex
        self.height = height
        self.radius = radius

    def intersect(self, ray):
        a = self.apex
        h = self.height
        r = self.radius

        k = (r / h) ** 2

        dx = ray.direction.x
        dy = ray.direction.y
        dz = ray.direction.z

        ox = ray.origin.x - a.x
        oy = ray.origin.y - a.y
        oz = ray.origin.z - a.z

        A = dx * dx + dz * dz - k * dy * dy
        B = 2 * (ox * dx + oz * dz - k * oy * dy)
        C = ox * ox + oz * oz - k * oy * oy

        if abs(A) < 1e-8:
            if abs(B) < 1e-8:
                return Intersection.none()
            t = -C / (2 * B)
            if t <= 0.0001:
                return Intersection.none()
            point = ray.point_at_parameter(t)
            if not self.is_point_within_height(point):
                return Intersection.none()
            normal = self.compute_normal(point)
            return Intersection(t, point, normal, self, self.material)

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            return Intersection.none()

        sqrt_d = math.sqrt(discriminant)
        t1 = (-B - sqrt_d) / (2 * A)
        t2 = (-B + sqrt_d) / (2 * A)

        valid_intersections = []

        for t in [t1, t2]:
            if t <= 0.0001:
                continue
            point = ray.point_at_parameter(t)
            if not self.is_point_within_height(point):
                continue
            valid_intersections.append((t, point))

        if not valid_intersections:
            return Intersection.none()

        t, point = min(valid_intersections, key=lambda x: x[0])
        normal = self.compute_normal(point)

        return Intersection(t, point, normal, self, self.material)

    def is_point_within_height(self, point):
        y_rel = point.y - self.apex.y

        if self.height > 0:
            if y_rel > 0 or y_rel < -self.height:
                return False
        else:
            if y_rel < 0 or y_rel > -self.height:
                return False

        return True

    def compute_normal(self, point):
        y_rel = point.y - self.apex.y

        nx = point.x - self.apex.x
        ny = -((self.radius / self.height) ** 2) * y_rel
        nz = point.z - self.apex.z

        normal = Vector3(nx, ny, nz)
        return normal.normalize()


class Paraboloid(Object):
    def __init__(self, vertex, a, height, material):
        super().__init__(material)
        self.vertex = vertex
        self.a = a
        self.height = height

    def intersect(self, ray):
        vx, vy, vz = self.vertex.x, self.vertex.y, self.vertex.z
        ox, oy, oz = ray.origin.x, ray.origin.y, ray.origin.z
        dx, dy, dz = ray.direction.x, ray.direction.y, ray.direction.z

        A = dx * dx + dz * dz
        B = 2 * (dx * (ox - vx) + dz * (oz - vz)) - self.a * dy
        C = (ox - vx) * (ox - vx) + (oz - vz) * (oz - vz) - self.a * (oy - vy)

        if abs(A) < 1e-8:
            if abs(B) < 1e-8:
                return Intersection.none()
            t = -C / B
            if t <= 0.0001:
                return Intersection.none()
            point = ray.point_at_parameter(t)
            if point.y < vy - self.height or point.y > vy + self.height:
                return Intersection.none()
            normal = Vector3(
                2 * (point.x - vx),
                -self.a,
                2 * (point.z - vz)
            )
            normal = normal.normalize()
            return Intersection(t, point, normal, self, self.material)

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            return Intersection.none()

        sqrt_d = math.sqrt(discriminant)
        t1 = (-B - sqrt_d) / (2 * A)
        t2 = (-B + sqrt_d) / (2 * A)

        valid_intersections = []

        for t in [t1, t2]:
            if t <= 0.0001:
                continue
            point = ray.point_at_parameter(t)
            if point.y < vy - self.height or point.y > vy + self.height:
                continue
            valid_intersections.append((t, point))

        if not valid_intersections:
            return Intersection.none()

        t, point = min(valid_intersections, key=lambda x: x[0])
        normal = Vector3(
            2 * (point.x - vx),
            -self.a,
            2 * (point.z - vz)
        )
        normal = normal.normalize()

        return Intersection(t, point, normal, self, self.material)