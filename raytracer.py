import numpy as np
from ray import Ray, Intersection, Vector3
from scene import Scene
import math


class RayTracer:
    def __init__(self, scene: Scene, max_depth: int = 3):
        self.scene = scene
        self.max_depth = max_depth

    def trace_ray(self, ray: Ray, depth: int = 0) -> Vector3:
        if depth >= self.max_depth:
            return self.scene.background_color

        intersection = self.scene.find_closest_intersection(ray)

        if not intersection.happened:
            return self.scene.background_color

        return self.shade(intersection, ray, depth)

    def shade(self, intersection: Intersection, ray: Ray, depth: int) -> Vector3:
        material = intersection.material

        # Локальное освещение (ambient + diffuse)
        local_color = self.compute_local_illumination(intersection)
        final_color = local_color

        # Отражение (рекурсия)
        if material.reflectivity > 0:
            reflection_color = self.compute_reflection(intersection, ray, depth)
            # Смешиваем с локальным цветом по коэффициенту отражения
            final_color = final_color + reflection_color * material.reflectivity

        if material.transparency > 0:
            refraction_color = self.compute_refraction(intersection, ray, depth)
            final_color = final_color + refraction_color * material.transparency

        # Ограничиваем цвет значениями [0, 1]
        return self.clamp_color(final_color)

    def compute_local_illumination(self, intersection: Intersection) -> Vector3:
        point = intersection.point
        normal = intersection.normal
        material = intersection.material

        color_r = material.color.x * material.ambient
        color_g = material.color.y * material.ambient
        color_b = material.color.z * material.ambient

        for light in self.scene.lights:
            # Вектор к свету (вычисляем вручную)
            light_dir = light.position - point

            # Длина и нормализация
            light_length = light_dir.length()
            light_dir = light_dir.normalize()

            # Теневой луч (смещение от самопересечения)
            shadow_origin = Vector3(point.x + normal.x * 0.001, point.y + normal.y * 0.001, point.z + normal.z * 0.001)

            shadow_ray = Ray(
                shadow_origin,
                light_dir
            )

            shadow_intersection = self.scene.find_closest_intersection(shadow_ray)

            # Проверяем тени
            if not shadow_intersection.happened or shadow_intersection.t > light_length:
                # Диффузная составляющая
                diffuse_intensity = max(0, light_dir.dot(normal))

                color_r += material.color.x * material.diffuse * diffuse_intensity * light.intensity
                color_g += material.color.y * material.diffuse * diffuse_intensity * light.intensity
                color_b += material.color.z * material.diffuse * diffuse_intensity * light.intensity

        # Ограничиваем цвет
        color_r = min(1.0, color_r)
        color_g = min(1.0, color_g)
        color_b = min(1.0, color_b)

        return Vector3(color_r, color_g, color_b)

    def compute_reflection(self, intersection: Intersection, ray: Ray, depth: int) -> Vector3:
        point = intersection.point
        normal = intersection.normal

        # Вектор отражения: R = I - 2*(I·N)*N
        incident = ray.direction
        reflection_dir = incident - normal * (2 * incident.dot(normal))
        reflection_dir = reflection_dir.normalize()

        # Создаем отраженный луч (с небольшим смещением чтобы избежать самопересечения)
        reflection_ray = Ray(point + normal * 0.001, reflection_dir)

        # Рекурсивно трассируем отраженный луч
        return self.trace_ray(reflection_ray, depth + 1)

    def compute_refraction(self, intersection: Intersection, ray: Ray, depth: int) -> Vector3:
        point = intersection.point
        normal = intersection.normal
        material = intersection.material

        # eta = in_IOR / out_IOR
        eta = 1.0 / material.ior

        cos_theta = -normal.dot(ray.direction)

        if cos_theta < 0:
            cos_theta *= -1.0
            normal = normal * -1.0
            eta = 1.0 / eta

        k = 1.0 - eta * eta * (1.0 - cos_theta * cos_theta)

        if k >= 0.0:
            # Вычисляем направление преломленного луча
            refraction_dir = (ray.direction * eta) + normal * (eta * cos_theta - math.sqrt(k))
            refraction_dir = refraction_dir.normalize()

            # Создаем преломленный луч
            refraction_ray = Ray(point + normal * -0.001, refraction_dir)

            # Рекурсивно трассируем преломленный луч
            return self.trace_ray(refraction_ray, depth + 1)
        else:
            # Полное внутреннее отражение
            return self.compute_reflection(intersection, ray, depth)

    def clamp_color(self, color: Vector3) -> Vector3:

        r = max(0, min(1, color.x))
        g = max(0, min(1, color.y))
        b = max(0, min(1, color.z))
        return Vector3(r, g, b)
