from typing import List
from objects import Object
from lights import Light
from ray import Vector3, Ray, Intersection


class Scene:
    def __init__(self):
        self.objects: List[Object] = []
        self.lights: List[Light] = []
        self.background_color = Vector3(0.1, 0.1, 0.3)

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def add_light(self, light: Light):
        self.lights.append(light)

    def find_closest_intersection(self, ray: Ray) -> Intersection:
        closest_intersection = Intersection.none()

        for obj in self.objects:
            intersection = obj.intersect(ray)
            if intersection.happened and intersection.t < closest_intersection.t:
                closest_intersection = intersection

        return closest_intersection
