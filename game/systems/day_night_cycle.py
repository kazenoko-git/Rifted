# game/systems/day_night_cycle.py

from aurora_engine.ecs.system import System
from aurora_engine.scene.transform import Transform
from aurora_engine.rendering.light import DirectionalLight, AmbientLight
import numpy as np
import math
from aurora_engine.core.logging import get_logger

logger = get_logger()

class DayNightCycle(System):
    """
    Manages the day/night cycle, including sun/moon movement and lighting/sky color changes.
    """

    def __init__(self, renderer, day_duration: float = 240.0):
        super().__init__()
        self.renderer = renderer
        self.day_duration = day_duration  # Seconds for a full day
        self.time = 0.0  # 0.0 to 1.0 (0 = noon, 0.5 = midnight)
        self.paused = False
        
        self.sun_entity = None
        self.moon_entity = None
        self.ambient_entity = None
        
        self.target = None  # Player transform to follow
        self.orbit_radius = 500.0  # Distance from player
        
        self._setup_color_gradient()
        
        # Start in bright daytime by default.
        self.time = 0.10
        logger.info("DayNightCycle initialized (single directional + ambient mode).")

    def _setup_color_gradient(self):
        # Time -> Color mapping.
        # Keep a minimum ambient floor to avoid near-black world lighting.
        self.sky_colors = {
            0.0: (0.53, 0.8, 0.92),   # Noon
            0.20: (0.95, 0.64, 0.38), # Sunset
            0.30: (0.25, 0.24, 0.36), # Twilight
            0.5: (0.07, 0.09, 0.16),  # Midnight
            0.70: (0.25, 0.24, 0.36), # Twilight
            0.80: (0.95, 0.64, 0.38), # Sunrise
            1.0: (0.53, 0.8, 0.92),   # Noon
        }
        self.sun_colors = {
            0.0: (1.0, 0.98, 0.92),
            0.20: (1.0, 0.80, 0.62),
            0.30: (0.55, 0.58, 0.72),
            0.5: (0.32, 0.40, 0.58),
            0.70: (0.55, 0.58, 0.72),
            0.80: (1.0, 0.80, 0.62),
            1.0: (1.0, 0.98, 0.92),
        }
        self.ambient_colors = {
            0.0: (0.34, 0.35, 0.39),
            0.20: (0.30, 0.26, 0.28),
            0.30: (0.22, 0.22, 0.28),
            0.5: (0.18, 0.19, 0.25),  # Never drop to near-black.
            0.70: (0.22, 0.22, 0.28),
            0.80: (0.30, 0.26, 0.28),
            1.0: (0.34, 0.35, 0.39),
        }

    def get_required_components(self):
        return [] # Global system

    def update(self, entities, dt):
        if not self.paused:
            self.time = (self.time + dt / self.day_duration) % 1.0
        
        # Calculate Sun/Moon Position relative to Player
        center_pos = np.array([0, 0, 0], dtype=np.float32)
        if self.target:
            center_pos = self.target.get_world_position()

        # Full sky orbit:
        # - Azimuth rotates around the world (X/Y), so shadow direction clearly changes.
        # - Elevation controls day vs night (Z).
        angle = self.time * 2.0 * math.pi
        horiz_radius = self.orbit_radius * 0.85

        sun_x = math.cos(angle) * horiz_radius
        sun_y = math.sin(angle) * horiz_radius
        sun_z = math.cos(angle) * self.orbit_radius

        sun_pos = np.array([
            center_pos[0] + sun_x,
            center_pos[1] + sun_y,
            center_pos[2] + sun_z,
        ], dtype=np.float32)

        moon_pos = np.array([
            center_pos[0] - sun_x,
            center_pos[1] - sun_y,
            center_pos[2] - sun_z,
        ], dtype=np.float32)
        
        # Update Entities
        if self.sun_entity:
            t = self.sun_entity.get_component(Transform)
            t.set_world_position(sun_pos)
            self._look_at(t, center_pos)
            
        if self.moon_entity:
            t = self.moon_entity.get_component(Transform)
            t.set_world_position(moon_pos)
            self._look_at(t, center_pos)
            
        self._update_colors(sun_z)

    def _look_at(self, transform, target_pos):
        origin = transform.get_world_position()
        direction = target_pos - origin
        if np.linalg.norm(direction) < 0.001: return
        direction /= np.linalg.norm(direction)
        
        up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(direction, up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        up /= np.linalg.norm(up)
        
        rot_mat = np.eye(3, dtype=np.float32)
        rot_mat[:, 0] = right
        rot_mat[:, 1] = direction # Forward
        rot_mat[:, 2] = up
        
        from aurora_engine.utils.math import matrix_to_quaternion
        transform.set_world_rotation(matrix_to_quaternion(rot_mat))

    def _interpolate_color(self, gradient, time):
        keys = sorted(gradient.keys())
        key1 = keys[0]
        key2 = keys[-1]
        for k in keys:
            if k <= time:
                key1 = k
            if k >= time:
                key2 = k
                break
        if key1 == key2:
            return np.array(gradient[key1], dtype=np.float32)
            
        t = (time - key1) / (key2 - key1)
        c1 = np.array(gradient[key1], dtype=np.float32)
        c2 = np.array(gradient[key2], dtype=np.float32)
        return c1 * (1 - t) + c2 * t

    def _update_colors(self, sun_height):
        sky_color = self._interpolate_color(self.sky_colors, self.time)

        # Day factor from sun elevation. Keeps ambient stable at night.
        day_factor = max(0.0, min(1.0, (sun_height / self.orbit_radius + 1.0) * 0.5))
        # Robust physically-inspired day/night blend.
        sun_day = np.array([1.0, 0.98, 0.92], dtype=np.float32)
        sun_night = np.array([0.35, 0.42, 0.58], dtype=np.float32)
        ambient_day = np.array([0.34, 0.35, 0.39], dtype=np.float32)
        ambient_night = np.array([0.14, 0.16, 0.22], dtype=np.float32)

        sun_color = sun_night * (1.0 - day_factor) + sun_day * day_factor
        ambient_color = ambient_night * (1.0 - day_factor) + ambient_day * day_factor
        
        # Apply colors
        if hasattr(self.renderer.backend, 'base'):
            self.renderer.backend.base.setBackgroundColor(sky_color[0], sky_color[1], sky_color[2], 1)
        
        # Update Fog
        if hasattr(self.renderer.backend.scene_graph, 'getFog'):
            fog = self.renderer.backend.scene_graph.getFog()
            if fog:
                fog.setColor(sky_color[0], sky_color[1], sky_color[2])
            
        # Update Light Components
        if self.sun_entity:
            light = self.sun_entity.get_component(DirectionalLight)
            if light:
                light.color = sun_color
                # Make day/night change clearly visible on terrain.
                light.intensity = 0.10 + 1.50 * day_factor

        if self.moon_entity:
            light = self.moon_entity.get_component(DirectionalLight)
            if light:
                light.color = np.array([0.30, 0.36, 0.50], dtype=np.float32)
                light.intensity = 0.10 + 0.25 * (1.0 - day_factor)

        if self.ambient_entity:
            light = self.ambient_entity.get_component(AmbientLight)
            if light:
                light.color = ambient_color
                light.intensity = 1.00
