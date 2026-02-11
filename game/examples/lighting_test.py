# game/examples/lighting_test.py

import sys
import os
import numpy as np
import math
import json

# Add project root to path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from aurora_engine.core.application import Application
from aurora_engine.scene.transform import Transform
from aurora_engine.rendering.mesh import MeshRenderer, create_cube_mesh, create_sphere_mesh, create_plane_mesh
from aurora_engine.rendering.light import DirectionalLight, AmbientLight, PointLight
from aurora_engine.camera.camera import Camera
from aurora_engine.camera.free_fly import FreeFlyController
from aurora_engine.core.logging import get_logger
from aurora_engine.utils.math import quaternion_from_euler
from panda3d.core import Shader as PandaShader, loadPrcFileData, Vec4, Vec3, Quat, LMatrix4f, CullFaceAttrib, NodePath, DepthTestAttrib, StencilAttrib, RenderState, ColorWriteAttrib

logger = get_logger()

# --- Configuration for Shadow Map Debugging ---
loadPrcFileData("", "show-buffers 0") # Set to 1 to see all buffers by default

class LightingTest(Application):
    """
    Standalone test for Toon Shading and Real-time Shadows.
    """

    def initialize_game(self):
        logger.info("Initializing Toon Lighting Test...")
        
        # 1. Setup Camera
        self.camera = Camera()
        self.camera.transform.set_world_position(np.array([0, -25, 15], dtype=np.float32))
        self.renderer.register_camera(self.camera)
        
        self.cam_controller = FreeFlyController(self.camera)
        self.cam_controller.move_speed = 15.0
        
        # 2. Create Scene using the ECS
        self._create_scene()
        
        # 3. Create Lights using the ECS
        self._create_lights()
        
        # 4. Apply Outline Shells (Characters Only)
        self._apply_outlines()

        # 6. Input state for single-press toggle
        self._v_key_pressed = False
        
        logger.info("Lighting Test Initialized.")
        logger.info("Controls: WASD to Move, Hold Right Click + Mouse to Look.")
        logger.info("Debug: 'V' to toggle Shadow Map View, 'L' to rotate Sun.")

    def _load_outline_shader(self):
        """Load the outline shader for inverted hull rendering."""
        shader_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shaders"))
        try:
            self.outline_shader = PandaShader.load(
                PandaShader.SL_GLSL,
                vertex=os.path.join(shader_dir, "outline.vert"),
                fragment=os.path.join(shader_dir, "outline.frag")
            )
        except Exception as e:
            logger.error(f"Failed to load outline shader: {e}")
            self.outline_shader = None

    def _create_scene(self):
        """Creates the scene geometry using the engine's ECS."""
        self.scene_entities = []

        # Ground (Replaced Plane with Cube as requested)
        ground = self.world.create_entity()
        ground.add_component(Transform())
        ground.get_component(Transform).set_world_position(np.array([0, 0, -1.0], dtype=np.float32)) # Lowered slightly to keep surface at 0
        ground.get_component(Transform).set_local_scale(np.array([40, 40, 2], dtype=np.float32)) # Thickness of 2
        
        # Re-enable shadow casting for the ground if using complexpbr, as it handles self-shadowing better
        # But let's keep it False for now to be safe, unless complexpbr needs it.
        # complexpbr usually handles shadows well.
        ground_renderer = MeshRenderer(mesh=create_cube_mesh(), color=(0.8, 0.8, 0.8, 1.0), shading_model="world")
        ground_renderer.cast_shadows = True # Re-enabled for complexpbr
        ground.add_component(ground_renderer)
        
        ground.tag = "world"
        self.scene_entities.append(ground)

        # Red Sphere (Character)
        sphere = self.world.create_entity()
        sphere_transform = Transform()
        sphere_transform.set_local_position(np.array([0, 0, 2], dtype=np.float32))
        sphere_transform.set_local_scale(np.array([2, 2, 2], dtype=np.float32))
        sphere.add_component(sphere_transform)
        sphere.add_component(MeshRenderer(mesh=create_sphere_mesh(), color=(1.0, 0.2, 0.2, 1.0), shading_model="character"))
        sphere.tag = "character"
        self.scene_entities.append(sphere)

        # Green Cube (Character)
        cube = self.world.create_entity()
        cube_transform = Transform()
        cube_transform.set_local_position(np.array([8, 5, 1.5], dtype=np.float32))
        cube_transform.set_local_rotation(quaternion_from_euler(np.radians(np.array([0.0, 0.0, 35.0], dtype=np.float32))))
        cube_transform.set_local_scale(np.array([3, 3, 3], dtype=np.float32))
        cube.add_component(cube_transform)
        cube.add_component(MeshRenderer(mesh=create_cube_mesh(), color=(0.2, 1.0, 0.2, 1.0), shading_model="character"))
        cube.tag = "character" # Changed to character to get Toon Shader + Outline
        self.scene_entities.append(cube)

        # Blue Pillar (Character)
        pillar = self.world.create_entity()
        pillar_transform = Transform()
        pillar_transform.set_local_position(np.array([-8, 5, 4], dtype=np.float32))
        pillar_transform.set_local_scale(np.array([1, 1, 8.0], dtype=np.float32))
        pillar.add_component(pillar_transform)
        pillar.add_component(MeshRenderer(mesh=create_cube_mesh(), color=(0.2, 0.2, 1.0, 1.0), shading_model="character"))
        pillar.tag = "character"
        self.scene_entities.append(pillar)

    def _create_lights(self):
        # Ambient Light (Base fill)
        amb = self.world.create_entity()
        amb.add_component(AmbientLight(color=(0.3, 0.3, 0.4), intensity=0.4))
        
        # Directional Light (Sun)
        self.sun = self.world.create_entity()
        self.sun.add_component(Transform())
        self.sun.get_component(Transform).set_world_position(np.array([0, -20, 20], dtype=np.float32))
        
        # Initial Rotation
        self.sun_pitch = -60.0
        self.sun_yaw = -30.0
        self._update_sun_rotation()
        
        dlight = DirectionalLight(color=(1.0, 0.95, 0.8), intensity=2.0)
        dlight.cast_shadows = True
        dlight.shadow_map_size = 2048 # 4096 might be heavy for M1 8GB if unoptimized
        dlight.shadow_film_size = 100.0 # Increased to ensure ground coverage
        self.sun.add_component(dlight)

    def _update_sun_rotation(self):
        q_sun = quaternion_from_euler(np.radians(np.array([self.sun_pitch, self.sun_yaw, 0.0], dtype=np.float32)))
        self.sun.get_component(Transform).set_world_rotation(q_sun)
        
        # Calculate Light Direction Vector (Pointing TO the light)
        # Standard Forward is Y+. We rotate (0,1,0).
        p = math.radians(self.sun_pitch)
        y = math.radians(self.sun_yaw)
        
        # Direction vector for shader (To Light)
        # Inverse of the forward vector
        dir_x = -math.sin(y) * math.cos(p)
        dir_y = math.cos(y) * math.cos(p)
        dir_z = math.sin(p)
        
        # We want vector TO light, so negate
        self.sun_direction = Vec3(-dir_x, -dir_y, -dir_z)
        
        # Force update shader input immediately
        if hasattr(self, 'renderer') and self.renderer.backend.scene_graph:
            self.renderer.backend.scene_graph.setShaderInput("u_sun_direction", self.sun_direction)

    def _apply_outlines(self):
        """Apply outline shells to character meshes (non-world)."""
        self._load_outline_shader()
        if not self.outline_shader:
            return
        self.renderer.backend.base.taskMgr.doMethodLater(0.1, self._apply_outline_task, "ApplyOutlineTask")

    def _apply_outline_task(self, task):
        for entity in self.scene_entities:
            mesh_renderer = entity.get_component(MeshRenderer)
            if mesh_renderer and mesh_renderer._node_path:
                np = mesh_renderer._node_path
                
                # Apply Outline (Inverted Hull) only to characters
                if hasattr(entity, 'tag') and entity.tag == "character":
                    # --- 1. Setup Stencil on the Main Object ---
                    # We want the main object to WRITE a reference value (e.g., 1) to the stencil buffer.
                    # This marks the pixels occupied by the object.
                    
                    # Create a StencilAttrib
                    # stencil_op: (stencil_fail, stencil_pass_z_fail, stencil_pass_z_pass)
                    # We want to replace the stencil value with our reference (1) when the depth test passes.
                    stencil_attrib = StencilAttrib.make(
                        1,                          # enabled
                        StencilAttrib.SCFAlways,    # comparison function (always pass)
                        StencilAttrib.SOKeep,       # stencil fail op
                        StencilAttrib.SOKeep,       # z fail op
                        StencilAttrib.SOReplace,    # z pass op (write ref value)
                        1,                          # reference value
                        0xFF,                       # read mask
                        0xFF                        # write mask
                    )
                    np.setAttrib(stencil_attrib)
                    
                    # --- 2. Create the Outline Shell ---
                    outline_name = f"outline_shell_{entity.id}"
                    old_outline = np.getParent().find(f"**/{outline_name}")
                    if not old_outline.isEmpty():
                        old_outline.removeNode()
                        
                    outline = np.copyTo(np.getParent())
                    outline.setName(outline_name)
                    outline.setMat(np.getMat(np.getParent()))
                    
                    outline.setShader(self.outline_shader)
                    # Increased outline width as requested (0.03 -> 0.06)
                    outline.setShaderInput("u_outline_width", 0.06)
                    outline.setShaderInput("u_outline_color", Vec4(0.02, 0.02, 0.03, 1))
                    
                    # --- 3. Configure Outline Rendering ---
                    
                    # A. Inverted Hull: Cull Front Faces
                    # This means we only see the back faces of the expanded shell.
                    outline.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
                    
                    # B. Stencil Masking: Only draw where Stencil != 1
                    # This prevents the outline from drawing ON TOP of the object's faces.
                    # It effectively "cuts out" the object from the outline.
                    outline_stencil = StencilAttrib.make(
                        1,                          # enabled
                        StencilAttrib.SCFNotEqual,  # comparison function (pass if stencil != ref)
                        StencilAttrib.SOKeep,       # stencil fail op
                        StencilAttrib.SOKeep,       # z fail op
                        StencilAttrib.SOKeep,       # z pass op
                        1,                          # reference value (same as object)
                        0xFF,                       # read mask
                        0x00                        # write mask (don't write)
                    )
                    outline.setAttrib(outline_stencil)
                    
                    # C. Depth Settings
                    # We want the outline to be depth tested against the world, but we don't necessarily
                    # need it to write to depth (optional, but safer to turn off for outlines).
                    # Crucially, since we use stencil to mask the object, we don't need to worry about
                    # depth fighting with the object itself as much.
                    outline.setDepthWrite(False)
                    outline.setDepthTest(True)
                    
                    # D. Render Order
                    # Draw outline AFTER the object so the stencil buffer is populated.
                    outline.setBin("fixed", 40)
                    
                    # Disable lighting on the outline itself (it's a solid color)
                    outline.setLightOff(1)

                    outline.setTwoSided(False)
                
        return task.done

    def update(self, dt: float, alpha: float):
        super().update(dt, alpha)
        
        # --- Input & Camera ---
        move = np.zeros(3)
        if self.input.is_key_down('w'): move[1] += 1
        if self.input.is_key_down('s'): move[1] -= 1
        if self.input.is_key_down('a'): move[0] -= 1
        if self.input.is_key_down('d'): move[0] += 1
        if self.input.is_key_down('e'): move[2] += 1
        if self.input.is_key_down('q'): move[2] -= 1
        self.cam_controller.move(move)
        
        md = self.input.get_mouse_delta()
        if self.input.is_key_down('mouse3') or self.input.mouse_locked:
            self.cam_controller.rotate(md[0] * -50, md[1] * 50)
        self.cam_controller.update(dt)

        # --- Debug Controls ---
        # Toggle Shadow Map Debug
        if self.input.is_key_down('v'):
            if not self._v_key_pressed:
                self.renderer.backend.base.bufferViewer.toggleEnable()
                self._v_key_pressed = True
        else:
            self._v_key_pressed = False

        # Rotate Sun
        if self.input.is_key_down('l'):
            # Rotate Yaw
            self.sun_yaw += dt * 20.0
            self._update_sun_rotation()

        # Toggle mouse lock
        if self.input.is_key_down('escape'):
            self.input.set_mouse_lock(False)
        elif self.input.is_key_down('mouse1'):
            self.input.set_mouse_lock(True)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "test_config.json")
    # Clean up old config if it exists
    if os.path.exists(config_path):
        os.remove(config_path)
        
    with open(config_path, "w") as f:
        json.dump({
            'rendering': {
                'width': 1280, 
                'height': 720, 
                'title': 'Toon Lighting Test'
            }, 
            'database': {'database': 'test.db'}
        }, f, indent=4)

    app = LightingTest(config_path)
    try:
        app.run()
    finally:
        # Clean up config file to keep the directory clean
        if os.path.exists(config_path):
            pass # Keep it for inspection if needed, or os.remove(config_path)
