# game/managers/debug_manager.py

import math
import numpy as np
from typing import Optional
from aurora_engine.core.logging import get_logger
from aurora_engine.input.input_manager import InputManager
from aurora_engine.rendering.renderer import Renderer
from aurora_engine.physics.physics_world import PhysicsWorld
from aurora_engine.ecs.world import World
from aurora_engine.ui.widget import Label
from aurora_engine.scene.transform import Transform
from aurora_engine.rendering.light import DirectionalLight, AmbientLight
from game.systems.day_night_cycle import DayNightCycle

logger = get_logger()

class DebugManager:
    """
    Manages debug features like overlays, wireframe toggles, and secondary cameras.
    """
    
    def __init__(self, world: World, renderer: Renderer, input_manager: InputManager, physics_world: PhysicsWorld, ui_manager):
        self.world = world
        self.renderer = renderer
        self.input = input_manager
        self.physics = physics_world
        self.ui = ui_manager
        
        # State
        self.debug_panda_visible = False
        self.debug_node_path = None
        self.f3_pressed = False
        self.f4_pressed = False
        self.f5_pressed = False
        self.f6_pressed = False
        self.show_lighting_debug = False
        self.show_shadow_buffer = False
        
        # Time Control State
        self.bracket_left_pressed = False
        self.bracket_right_pressed = False
        self.p_pressed = False
        
        # Secondary Camera
        self.secondary_window = None
        self.secondary_cam_np = None
        self.debug_camera_active = False
        self.debug_cam_angle = 0.0
        self.debug_cam_distance = 150.0
        self.debug_cam_height = 100.0
        
        # UI
        self.debug_label = Label("DebugInfo", "FPS: 60")
        self.debug_label.position = np.array([10, 10], dtype=np.float32)
        self.debug_label.color = (0, 0, 0, 1) # Black text
        self.ui.add_widget(self.debug_label, layer='overlay')
        
        self.lighting_label = Label("LightingInfo", "Lighting Debug Off")
        self.lighting_label.position = np.array([10, 50], dtype=np.float32)
        self.lighting_label.color = (1, 0, 0, 1) # Red text
        self.lighting_label.visible = False
        self.ui.add_widget(self.lighting_label, layer='overlay')

    def update(self, dt: float, player_pos: np.ndarray):
        """Update debug logic."""
        self._handle_input()
        self._update_ui(dt, player_pos)
        
        if self.secondary_window:
            self._update_secondary_camera(dt, player_pos)

    def _handle_input(self):
        """Handle debug key toggles."""
        # F3: Toggle Wireframe / Physics Debug / FrameRate
        if self.input.is_key_down('f3'):
            if not self.f3_pressed:
                self.f3_pressed = True
                self._toggle_debug_view()
        else:
            self.f3_pressed = False
            
        # F4: Toggle Secondary Camera
        if self.input.is_key_down('f4'):
            if not self.f4_pressed:
                self.f4_pressed = True
                if not self.debug_camera_active:
                    self.debug_camera_active = True
                    self._open_secondary_window()
        else:
            self.f4_pressed = False
            
        # F5: Toggle Lighting Debug
        if self.input.is_key_down('f5'):
            if not self.f5_pressed:
                self.f5_pressed = True
                self.show_lighting_debug = not self.show_lighting_debug
                self.lighting_label.visible = self.show_lighting_debug
                logger.info(f"Toggled Lighting Debug: {self.show_lighting_debug}")
        else:
            self.f5_pressed = False
            
        # F6: Toggle Shadow Buffer View
        if self.input.is_key_down('f6'):
            if not self.f6_pressed:
                self.f6_pressed = True
                self._toggle_shadow_buffer()
        else:
            self.f6_pressed = False
            
        # Time Controls
        day_night = self._get_day_night_system()
        # Require Shift modifier for time controls to avoid accidental pause.
        shift_down = self.input.is_key_down('lshift') or self.input.is_key_down('rshift')
        if day_night and shift_down:
            # [: Time - 0.1
            if self.input.is_key_down('['):
                if not self.bracket_left_pressed:
                    self.bracket_left_pressed = True
                    day_night.time = (day_night.time - 0.1) % 1.0
                    logger.info(f"Time set to: {day_night.time:.2f}")
            else:
                self.bracket_left_pressed = False
                
            # ]: Time + 0.1
            if self.input.is_key_down(']'):
                if not self.bracket_right_pressed:
                    self.bracket_right_pressed = True
                    day_night.time = (day_night.time + 0.1) % 1.0
                    logger.info(f"Time set to: {day_night.time:.2f}")
            else:
                self.bracket_right_pressed = False
                
            # P: Pause/Resume
            if self.input.is_key_down('p'):
                if not self.p_pressed:
                    self.p_pressed = True
                    day_night.paused = not day_night.paused
                    logger.info(f"Time Paused: {day_night.paused}")
            else:
                self.p_pressed = False
        else:
            self.bracket_left_pressed = False
            self.bracket_right_pressed = False
            self.p_pressed = False

    def _toggle_debug_view(self):
        """Toggle visual debug modes."""
        self.debug_panda_visible = not self.debug_panda_visible
        
        if hasattr(self.renderer.backend, 'base'):
            base = self.renderer.backend.base
            base.setFrameRateMeter(self.debug_panda_visible)
            
            if self.debug_panda_visible:
                base.render.setRenderModeWireframe()
                if not self.debug_node_path:
                    self.debug_node_path = self.physics.attach_debug_node(self.renderer.backend.scene_graph)
                if self.debug_node_path:
                    self.debug_node_path.show()
            else:
                base.render.clearRenderMode()
                if self.debug_node_path:
                    self.debug_node_path.hide()

    def _toggle_shadow_buffer(self):
        """Toggle shadow buffer visualization."""
        self.show_shadow_buffer = not self.show_shadow_buffer
        if hasattr(self.renderer.backend, 'base'):
            if self.show_shadow_buffer:
                self.renderer.backend.base.bufferViewer.toggleEnable()
            else:
                self.renderer.backend.base.bufferViewer.toggleEnable()
        logger.info(f"Shadow Buffer View: {self.show_shadow_buffer}")

    def _update_ui(self, dt: float, player_pos: np.ndarray):
        """Update debug overlay text."""
        if self.debug_label.visible:
            fps = 1.0 / dt if dt > 0 else 60.0
            self.debug_label.text = f"FPS: {fps:.0f} | Pos: {player_pos[0]:.1f}, {player_pos[1]:.1f}, {player_pos[2]:.1f}"
            
        if self.show_lighting_debug:
            self.lighting_label.text = self._get_lighting_info()

    def _get_day_night_system(self) -> Optional[DayNightCycle]:
        for sys in self.world.systems:
            if isinstance(sys, DayNightCycle):
                return sys
        return None

    def _get_lighting_info(self) -> str:
        """Retrieve current lighting stats."""
        info = "Lighting Info:\n"
        
        day_night = self._get_day_night_system()
        
        if day_night:
            status = "PAUSED" if day_night.paused else "RUNNING"
            info += f"Time: {day_night.time:.2f} ({status})\n"
            
            if day_night.sun_entity:
                light = day_night.sun_entity.get_component(DirectionalLight)
                if light:
                    c = light.color
                    info += f"Sun: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) Int: {light.intensity:.2f}\n"
                    t = day_night.sun_entity.get_component(Transform)
                    if t:
                        f = t.forward
                        info += f"SunDir(Fwd): ({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f})\n"
            
            if day_night.moon_entity:
                light = day_night.moon_entity.get_component(DirectionalLight)
                if light:
                    c = light.color
                    info += f"Moon: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) Int: {light.intensity:.2f}\n"
                    
            if day_night.ambient_entity:
                light = day_night.ambient_entity.get_component(AmbientLight)
                if light:
                    c = light.color
                    info += f"Amb: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) Int: {light.intensity:.2f}\n"
        else:
            info += "DayNightCycle System not found."
            
        return info

    def _open_secondary_window(self):
        """Open a secondary window for global observation."""
        if self.secondary_window:
            return
            
        from panda3d.core import WindowProperties, Camera as PandaCamera
        
        base = self.renderer.backend.base
        props = WindowProperties()
        props.setTitle("Debug View - Global Observer")
        props.setSize(640, 480)
        
        self.secondary_window = base.openWindow(props=props, makeCamera=False)
        cam_node = PandaCamera('secondary_cam')
        lens = cam_node.getLens()
        lens.setFov(90)
        lens.setNear(1.0)
        lens.setFar(5000.0)
        
        self.secondary_cam_np = base.render.attachNewNode(cam_node)
        dr = self.secondary_window.makeDisplayRegion()
        dr.setCamera(self.secondary_cam_np)
        
        logger.info("Secondary debug window opened")

    def _update_secondary_camera(self, dt: float, target_pos: np.ndarray):
        """Orbit secondary camera around target."""
        self.debug_cam_angle += dt * 0.2
        
        x = target_pos[0] + math.cos(self.debug_cam_angle) * self.debug_cam_distance
        y = target_pos[1] + math.sin(self.debug_cam_angle) * self.debug_cam_distance
        z = target_pos[2] + self.debug_cam_height
        
        if self.secondary_cam_np:
            self.secondary_cam_np.setPos(x, y, z)
            self.secondary_cam_np.lookAt(target_pos[0], target_pos[1], target_pos[2])
