from aurora_engine.rendering.mesh import MeshRenderer, create_sphere_mesh
from aurora_engine.scene.transform import Transform
from aurora_engine.ecs.world import World
from aurora_engine.rendering.renderer import Renderer
from aurora_engine.rendering.light import DirectionalLight, AmbientLight
from game.systems.day_night_cycle import DayNightCycle
from game.systems.fade_in_system import FadeInSystem
from game.managers.world_manager import WorldManager
from panda3d.core import Fog
import numpy as np

class EnvironmentManager:
    """
    Manages environmental effects, celestial bodies, and day/night cycle.
    """
    def __init__(self, world: World, renderer: Renderer, world_manager: WorldManager):
        self.world = world
        self.renderer = renderer
        self.world_manager = world_manager

    def setup(self, player_transform: Transform):
        """Initialize environment systems and entities."""
        self._setup_fog()
        sun, moon, ambient = self._create_celestial_bodies()
        self._setup_systems(sun, moon, ambient, player_transform)

    def _setup_fog(self):
        """Setup distance fog for performance and atmosphere."""
        # Fog disabled by user request
        if hasattr(self.renderer.backend, 'scene_graph'):
            self.renderer.backend.scene_graph.clearFog()

    def _create_celestial_bodies(self):
        """Create visual entities for Sun and Moon."""
        # Sun (Sphere + Light)
        sun = self.world.create_entity()
        sun.add_component(Transform())
        
        sun_renderer = MeshRenderer(
            mesh=create_sphere_mesh(radius=10.0, segments=32, rings=16), 
            color=(1.0, 1.0, 0.8, 1.0),
            texture_path="assets/textures/sun.png",
            shading_model="world",
        )
        sun.add_component(sun_renderer)
        
        sun_light = DirectionalLight(color=(1.0, 1.0, 0.8), intensity=1.0)
        sun_light.cast_shadows = True
        sun_light.shadow_map_size = 4096
        # Increase film size to cover more area
        sun_light.shadow_film_size = 1000.0 
        # Adjust near/far to ensure coverage
        sun_light.shadow_near_far = (1.0, 2000.0)
        sun.add_component(sun_light)
        
        # Moon (Sphere + Light)
        moon = self.world.create_entity()
        moon.add_component(Transform())
        
        moon_renderer = MeshRenderer(
            mesh=create_sphere_mesh(radius=8.0, segments=32, rings=16), 
            color=(0.8, 0.8, 1.0, 1.0),
            texture_path="assets/textures/moon.png",
            shading_model="world",
        )
        moon.add_component(moon_renderer)
        
        moon_light = DirectionalLight(color=(0.2, 0.2, 0.3), intensity=1.0)
        # Keep a single shadow map for the global sun to avoid shadow-map conflicts
        moon_light.cast_shadows = False
        moon_light.shadow_map_size = 2048
        moon_light.shadow_film_size = 1000.0
        moon_light.shadow_near_far = (1.0, 2000.0)
        moon.add_component(moon_light)
        
        # Ambient Light Entity
        ambient = self.world.create_entity()
        ambient.add_component(AmbientLight(color=(0.2, 0.2, 0.2), intensity=1.0))
        
        return sun, moon, ambient

    def _setup_systems(self, sun, moon, ambient, player_transform):
        self.world.add_system(FadeInSystem())
        
        day_night = DayNightCycle(self.renderer)
        day_night.target = player_transform
        day_night.sun_entity = sun
        day_night.moon_entity = moon
        day_night.ambient_entity = ambient
        day_night.orbit_radius = 500.0
        self.world.add_system(day_night)
