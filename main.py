import glfw
from OpenGL.GL import *
import numpy as np
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ray import Vector3, Ray
from scene import Scene
from objects import Sphere
from materials import Material
from lights import Light
from raytracer import RayTracer


def create_test_scene():
    scene = Scene()

    # Создаем материалы с разной отражающей способностью
    red_material = Material(
        color=Vector3(1.0, 0.2, 0.2),  # красный цвет
        ambient=0.1,
        diffuse=0.7,
        specular=0.5,
        reflectivity=0.4  # 40% отражения
    )

    blue_material = Material(
        color=Vector3(0.2, 0.2, 1.0),  # синий цвет
        ambient=0.1,
        diffuse=0.7,
        specular=0.5,
        reflectivity=0.2  # 20% отражения
    )

    mirror_material = Material(
        color=Vector3(1.0, 1.0, 1.0),  # белый цвет
        ambient=0.0,
        diffuse=0.0,
        specular=1.0,
        reflectivity=0.8  # 80% отражения (почти зеркало)
    )

    green_material = Material(
        color=Vector3(0.2, 1.0, 0.2),  # зеленый цвет
        ambient=0.1,
        diffuse=0.8,
        specular=0.3,
        reflectivity=0.1  # 10% отражения
    )

    # Создаем сферы
    sphere1 = Sphere(Vector3(0, 0, -5), 1.0, red_material)  # центральная красная
    sphere2 = Sphere(Vector3(2, 0, -6), 1.0, blue_material)  # правая синяя
    sphere3 = Sphere(Vector3(-2, 0, -4), 0.8, mirror_material)  # левая зеркальная
    sphere4 = Sphere(Vector3(0, -2, -5), 0.6, green_material)  # нижняя зеленая

    # Создаем источники света
    light1 = Light(
        position=Vector3(2, 5, -3),
        color=Vector3(1.0, 1.0, 1.0),  # белый свет
        intensity=1.0
    )

    light2 = Light(
        position=Vector3(-3, 3, -2),
        color=Vector3(0.8, 0.8, 1.0),  # голубоватый свет
        intensity=0.5
    )

    # Добавляем объекты в сцену
    scene.add_object(sphere1)
    scene.add_object(sphere2)
    scene.add_object(sphere3)
    scene.add_object(sphere4)

    # Добавляем источники света
    scene.add_light(light1)
    scene.add_light(light2)

    return scene


class RayTracingWindow:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.window = None

        # Создаем сцену и трассировщик
        self.scene = create_test_scene()
        self.raytracer = RayTracer(self.scene, max_depth=3)  # макс глубина рекурсии = 3
        self.camera_position = Vector3(0, 0, 0)  # камера в начале координат

        self.init_glfw()

    def init_glfw(self):
        """Инициализация GLFW и окна"""
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        # Создаем окно
        self.window = glfw.create_window(
            self.width,
            self.height,
            "Ray Tracing",
            None,
            None
        )

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        # Устанавливаем контекст
        glfw.make_context_current(self.window)

        # Устанавливаем callback'и
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)

        # Настраиваем OpenGL
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.1, 0.1, 0.1, 1.0)  # темно-серый цвет очистки

        print("Ray Tracing Window initialized")
        print(f"Scene contains: {len(self.scene.objects)} objects, {len(self.scene.lights)} lights")

    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        print(f"Window resized to: {width}x{height}")

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            print("ESC pressed - closing window")
            glfw.set_window_should_close(window, True)

    def pixel_to_ndc(self, x, y):
        # Преобразование пиксельных координат в Normalized Device Coordinates
        ndc_x = (x / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - (y / self.height) * 2.0
        return ndc_x, ndc_y

    def compute_pixel_color(self, x, y):
        # Получаем нормализованные координаты
        nx, ny = self.pixel_to_ndc(x, y)

        # Учитываем соотношение сторон
        aspect_ratio = self.width / self.height

        # Создаем луч
        ray_direction = Vector3(nx * aspect_ratio, ny, -1).normalize()
        ray = Ray(self.camera_position, ray_direction)

        # Трассируем
        color = self.raytracer.trace_ray(ray)
        return color.to_array()

    def display(self):
        glBegin(GL_POINTS)
        for y in range(self.height):
            for x in range(self.width):
                color = self.compute_pixel_color(x, y)
                gl_x, gl_y = self.pixel_to_ndc(x, y)
                glColor3f(color[0], color[1], color[2])
                glVertex2f(gl_x, gl_y)
        glEnd()

    def run(self):
        print("Starting ray tracing")

        while not glfw.window_should_close(self.window):
            # Очищаем экран
            glClear(GL_COLOR_BUFFER_BIT)

            # Отрисовываем сцену
            self.display()

            # Обновляем дисплей
            glfw.swap_buffers(self.window)

            # Обрабатываем события
            glfw.poll_events()

        # Завершаем работу
        glfw.terminate()
        print("Ray tracing finished.")


def main():
    try:
        # Создаем и запускаем приложение
        app = RayTracingWindow(800, 600)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()