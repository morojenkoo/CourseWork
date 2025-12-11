from dataclasses import dataclass
from ray import Vector3


@dataclass
class Material:
    color: Vector3
    ambient: float = 0.1
    diffuse: float = 0.9
    specular: float = 0.5
    shininess: float = 32.0
    reflectivity: float = 0.0
    transparency: float = 0.0
    ior: float = 1.0  # index of refraction (для преломления)
