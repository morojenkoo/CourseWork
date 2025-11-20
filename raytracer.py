import numpy as np
from ray import Ray, Intersection, Vector3
from scene import Scene


class RayTracer:
    def __init__(self, scene: Scene, max_depth: int = 3):
        self.scene = scene
        self.max_depth = max_depth

    def trace_ray(self, ray: Ray, depth: int = 0) -> Vector3:
        """
        Рекурсивная трассировка луча
        Returns: цвет в направлении луча
        """
        # Базовый случай рекурсии - достигли максимальной глубины
        if depth >= self.max_depth:
            return self.scene.background_color

        # Ищем ближайшее пересечение
        intersection = self.scene.find_closest_intersection(ray)

        # Если пересечения нет - возвращаем фоновый цвет
        if not intersection.happened:
            return self.scene.background_color

        # Вычисляем цвет в точке пересечения
        return self.shade(intersection, ray, depth)

    def shade(self, intersection: Intersection, ray: Ray, depth: int) -> Vector3:
        """Вычисление финального цвета с учетом освещения и рекурсивных эффектов"""
        material = intersection.material

        # Локальное освещение (ambient + diffuse)
        local_color = self.compute_local_illumination(intersection)
        final_color = local_color

        # Отражение (рекурсия)
        if material.reflectivity > 0:
            reflection_color = self.compute_reflection(intersection, ray, depth)
            # Смешиваем с локальным цветом по коэффициенту отражения
            final_color = final_color + reflection_color * material.reflectivity

        # Преломление (можно добавить позже)
        # if material.transparency > 0:
        #     refraction_color = self.compute_refraction(intersection, ray, depth)
        #     final_color = final_color + refraction_color * material.transparency

        # Ограничиваем цвет значениями [0, 1]
        return self.clamp_color(final_color)

    def compute_local_illumination(self, intersection: Intersection) -> Vector3:
        """Вычисление локального освещения (ambient + diffuse)"""
        point = intersection.point
        normal = intersection.normal
        material = intersection.material

        # Начинаем с ambient компонента
        color = material.color * material.ambient

        # Добавляем вклад от каждого источника света
        for light in self.scene.lights:
            # Направление к источнику света
            light_dir = (light.position - point).normalize()

            # Проверяем тени - испускаем теневой луч
            shadow_ray = Ray(point + normal * 0.001, light_dir)
            shadow_intersection = self.scene.find_closest_intersection(shadow_ray)

            # Если нет объектов между точкой и светом (или объект дальше света)
            light_distance = (light.position - point).length()
            if not shadow_intersection.happened or shadow_intersection.t > light_distance:
                # Диффузная составляющая (зависит от угла между нормалью и светом)
                diffuse_intensity = max(0, normal.dot(light_dir))

                # Добавляем диффузный цвет
                diffuse_color = material.color * material.diffuse * diffuse_intensity
                color = color + diffuse_color * light.intensity

        return color

    def compute_reflection(self, intersection: Intersection, ray: Ray, depth: int) -> Vector3:
        """Вычисление отраженного цвета"""
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
        """Вычисление преломленного цвета (заглушка для будущей реализации)"""
        # TODO: реализовать закон Снеллиуса для преломления
        # Пока возвращаем черный цвет
        return Vector3(0, 0, 0)

    def clamp_color(self, color: Vector3) -> Vector3:
        """Ограничивает компоненты цвета значениями от 0 до 1"""
        r = max(0, min(1, color.x))
        g = max(0, min(1, color.y))
        b = max(0, min(1, color.z))
        return Vector3(r, g, b)