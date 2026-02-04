import glfw
from OpenGL.GL import *
import numpy as np
import sys
import os

from ray import Vector3, Ray
from scene import Scene
from objects import Sphere, Cone, Paraboloid
from materials import Material
from lights import Light
from raytracer import RayTracer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_test_scene():
    scene = Scene()

    red_material = Material(
        color=Vector3(1.0, 0.2, 0.2),
        ambient=0.1,
        diffuse=0.7,
        specular=0.5,
        reflectivity=0.4
    )

    blue_material = Material(
        color=Vector3(0.2, 0.2, 1.0),
        ambient=0.1,
        diffuse=0.7,
        specular=0.5,
        reflectivity=0.2
    )

    mirror_material = Material(
        color=Vector3(1.0, 1.0, 1.0),
        diffuse=0.0,
        specular=1.0,
        reflectivity=1.0
    )

    green_material = Material(
        color=Vector3(0.2, 1.0, 0.2),
        ambient=0.1,
        diffuse=0.8,
        specular=0.3,
        reflectivity=0.1
    )
    glass_material = Material(
        color=Vector3(1.0, 1.0, 1.0),
        ambient=0.0,
        diffuse=0.0,
        specular=0.9,
        reflectivity=0.1,
        transparency=0.9,
        ior=1.5
    )

    sphere1 = Sphere(Vector3(0, 0, -5), 1.0, red_material)
    # sphere2 = Sphere(Vector3(0, 2, -5), 2.0, blue_material)

    cone1 = Cone(
        apex=Vector3(-3, -2, -5),
        height=-3.0,
        radius=1.4,
        material=green_material
    )
    paraboloid1 = Paraboloid(
        vertex=Vector3(3, -3, -4),
        a=0.5,
        height=4,
        material=blue_material
    )

    paraboloid2 = Paraboloid(
        vertex=Vector3(-2, 2, -6),
        a=-0.2,
        height=6,
        material=mirror_material
    )
    light1 = Light(
        position=Vector3(2, 5, -3),
        color=Vector3(1.0, 1.0, 1.0),
        intensity=1.0
    )
    light2 = Light(
        position=Vector3(-1, 4, 2),
        color=Vector3(1.0, 1.0, 1.0),
        intensity=1.0
    )
    light3 = Light(
        position=Vector3(-3, 2, -5),
        color=Vector3(1.0, 1.0, 1.0),
        intensity=1.0
    )

    scene.add_object(sphere1)
    # scene.add_object(sphere2)
    scene.add_object(cone1)
    scene.add_object(paraboloid1)
    scene.add_object(paraboloid2)
    scene.add_light(light1)
    scene.add_light(light2)
    scene.add_light(light3)

    return scene


class RayTracingWindow:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.window = None

        self.scene = create_test_scene()
        self.raytracer = RayTracer(self.scene, max_depth=3)
        self.camera_position = Vector3(0, 0, 0)

        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.texture_id = None
        self.raytrace_texture_data = None

        self.init_glfw()
        self.setup_shaders()
        self.setup_buffers()

    def load_shader_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading shader from {file_path}: {e}")
            return None

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            glDeleteShader(shader)
            raise Exception(f"Shader compilation error: {error}")

        return shader

    def setup_shaders(self):
        vertex_shader_source = self.load_shader_from_file("vertex_shader.glsl")
        fragment_shader_source = self.load_shader_from_file("fragment_shader.glsl")

        vertex_shader = self.compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)

        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            raise Exception(f"Shader program linking error: {error}")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def setup_buffers(self):
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 1.0,
        ], dtype=np.float32)

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)

        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        empty_data = np.zeros((self.height, self.width, 3), dtype=np.float32)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, empty_data)

    def init_glfw(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

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

        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)

        glViewport(0, 0, self.width, self.height)
        glClearColor(0.1, 0.1, 0.1, 1.0)

    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def pixel_to_ndc(self, x, y):
        ndc_x = (x / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - (y / self.height) * 2.0
        return ndc_x, ndc_y

    def compute_pixel_color(self, x, y):
        nx, ny = self.pixel_to_ndc(x, y)
        aspect_ratio = self.width / self.height

        ray_direction = Vector3(nx * aspect_ratio, ny, -1)
        ray_direction = ray_direction.normalize()
        ray = Ray(self.camera_position, ray_direction)

        color = self.raytracer.trace_ray(ray)
        return color.to_array()

    def render_to_texture(self):
        print("Rendering scene...")
        import time
        start_time = time.time()

        texture_data = np.zeros((self.height, self.width, 3), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                color = self.compute_pixel_color(x, y)
                texture_data[y, x] = color

        self.raytrace_texture_data = texture_data

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, texture_data)

        total_time = time.time() - start_time
        print(f"Ray tracing completed in {total_time:.2f} seconds")

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        texture_loc = glGetUniformLocation(self.shader_program, "rayTexture")
        glUniform1i(texture_loc, 0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    def run(self):
        print("Starting ray tracing")

        self.render_to_texture()
        print("Rendering complete")

        while not glfw.window_should_close(self.window):
            self.display()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])
        glDeleteTextures(1, [self.texture_id])
        glDeleteProgram(self.shader_program)

        glfw.terminate()
        print("Ray tracing finished")


def main():
    try:
        app = RayTracingWindow(800, 600)
        app.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()