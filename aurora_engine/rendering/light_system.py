# aurora_engine/rendering/light_system.py

from typing import List, Type
from aurora_engine.ecs.system import System
from aurora_engine.ecs.component import Component
from aurora_engine.scene.transform import Transform
from aurora_engine.rendering.light import Light, AmbientLight, DirectionalLight, PointLight
from aurora_engine.core.logging import get_logger
from panda3d.core import AmbientLight as PandaAmbientLight
from panda3d.core import DirectionalLight as PandaDirectionalLight
from panda3d.core import PointLight as PandaPointLight
from panda3d.core import Vec4, Vec3, NodePath, BitMask32, LMatrix4f, Quat

logger = get_logger()

class LightSystem(System):
    """
    System to manage light components and sync them with the rendering backend.
    """
    
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer
        self.priority = 90 # Run before rendering
        self._debug_log_timer = 0.0
        self._initialized_lights = set()

    def get_required_components(self) -> List[Type[Component]]:
        return [Light]

    def update(self, entities: List, dt: float):
        self._debug_log_timer += dt
        should_log = self._debug_log_timer > 5.0 # Log every 5 seconds
        if should_log:
            self._debug_log_timer = 0.0

        ambient_color = Vec3(0.0, 0.0, 0.0)
        directional_color = Vec3(0.0, 0.0, 0.0)
        sun_dir = Vec3(0.0, 1.0, -1.0)
        have_sun = False
        
        sun_light_node = None
        sun_lens = None
        
        for entity in entities:
            light = entity.get_component(Light)
            
            # Initialize backend light if needed
            if entity.id not in self._initialized_lights:
                self._initialize_light(entity, light)
                self._initialized_lights.add(entity.id)
                
            if light._backend_handle:
                self._update_light(entity, light, should_log)

            # Collect global lighting for shaders
            if isinstance(light, AmbientLight):
                c = light.color * light.intensity
                ambient_color += Vec3(c[0], c[1], c[2])

            if isinstance(light, DirectionalLight):
                # Prefer the first shadow-casting directional as the sun
                if (not have_sun) or light.cast_shadows:
                    c = light.color * light.intensity
                    directional_color = Vec3(c[0], c[1], c[2])
                    sun_dir = self._get_directional_light_vector(entity, light)
                    
                    if light.cast_shadows and light._backend_handle:
                        sun_light_node = light._backend_handle
                        sun_lens = light._backend_handle.node().getLens()
                        
                    have_sun = True

        self._apply_global_shader_inputs(ambient_color, directional_color, sun_dir, sun_light_node, sun_lens)

    def on_entity_removed(self, entity):
        """Clean up light when entity is removed."""
        light = entity.get_component(Light)
        if light and light._backend_handle:
            # The light is already cleared from scene graph, just remove the node
            if hasattr(self.renderer.backend, 'scene_graph'):
                self.renderer.backend.scene_graph.clearLight(light._backend_handle)
            light._backend_handle.removeNode()
            light._backend_handle = None
        if entity.id in self._initialized_lights:
            self._initialized_lights.remove(entity.id)

    def _initialize_light(self, entity, light: Light):
        """Create the Panda3D light object."""
        panda_light = None
        name = f"Light_{entity.id}"
        
        if isinstance(light, AmbientLight):
            panda_light = PandaAmbientLight(name)
            
        elif isinstance(light, DirectionalLight):
            panda_light = PandaDirectionalLight(name)
            if light.cast_shadows:
                panda_light.setShadowCaster(True, light.shadow_map_size, light.shadow_map_size)
                lens = panda_light.getLens()
                # Ensure film size is large enough to cover the view
                lens.setFilmSize(light.shadow_film_size, light.shadow_film_size)
                lens.setNearFar(*light.shadow_near_far)
                
                # Visualize Shadow Volume (Enabled for debugging)
                # panda_light.showFrustum()
                logger.info(f"  -> Shadows Enabled: Map={light.shadow_map_size}, Film={light.shadow_film_size}")
                
        elif isinstance(light, PointLight):
            panda_light = PandaPointLight(name)
            panda_light.setAttenuation(light.attenuation)
            
        if panda_light:
            # Attach to scene graph
            light_np = self.renderer.backend.scene_graph.attachNewNode(panda_light)
            self.renderer.backend.scene_graph.setLight(light_np)
            light._backend_handle = light_np
            
            # Force Shadow Bitmasks
            if isinstance(light, DirectionalLight) and light.cast_shadows:
                # Ensure everything is visible to the shadow camera
                # BitMask32.allOn() might be too aggressive if we use masks, but good for debugging
                panda_light.setCameraMask(BitMask32.allOn())
            
            logger.info(f"Initialized light: {name} ({type(light).__name__})")

    def _update_light(self, entity, light: Light, log_debug: bool = False):
        """Update light properties."""
        light_np = light._backend_handle
        panda_light = light_np.node()
        
        # Update Color
        color = Vec4(light.color[0], light.color[1], light.color[2], 1.0) * light.intensity
        panda_light.setColor(color)
        
        if log_debug:
            logger.info(f"Light {entity.id} Color: {color}")
        
        # Update Transform (if not Ambient)
        if not isinstance(light, AmbientLight):
            transform = entity.get_component(Transform)
            if transform:
                # Force update of world transform to ensure we have the latest data
                # This is critical if the transform was modified in the same frame (e.g. by input)
                transform._update_world_transform()

                pos = transform.get_world_position()
                rot = transform.get_world_rotation()
                
                # Update position
                light_np.setPos(pos[0], pos[1], pos[2])
                
                # Update rotation (Panda uses HPR or Quat)
                from panda3d.core import Quat
                light_np.setQuat(Quat(rot[3], rot[0], rot[1], rot[2]))
                
        # Update specific properties
        if isinstance(light, PointLight):
            panda_light.setAttenuation(light.attenuation)
            
        # Update shadow properties dynamically if needed
        if isinstance(light, DirectionalLight) and light.cast_shadows:
             lens = panda_light.getLens()
             if lens.getFilmSize().getX() != light.shadow_film_size:
                 lens.setFilmSize(light.shadow_film_size, light.shadow_film_size)
             if lens.getNear() != light.shadow_near_far[0] or lens.getFar() != light.shadow_near_far[1]:
                 lens.setNearFar(*light.shadow_near_far)

    def _get_directional_light_vector(self, entity, light: DirectionalLight) -> Vec3:
        """Return world-space vector pointing TO the light (for shaders)."""
        # Prefer ECS transform rotation so day/night updates always drive shader light direction,
        # even if backend light node state is delayed or stale.
        transform = entity.get_component(Transform)
        if transform:
            rot = transform.get_world_rotation()
            q = Quat(rot[3], rot[0], rot[1], rot[2])
            forward = q.xform(Vec3(0, 1, 0))
            if forward.length() > 0.0001:
                forward.normalize()
            return -forward

        if light._backend_handle:
            # The node's local +Y transformed to world.
            forward = light._backend_handle.getQuat().xform(Vec3(0, 1, 0))
            if forward.length() > 0.0001:
                forward.normalize()
            return -forward

        return Vec3(0.0, 1.0, -1.0)

    def _apply_global_shader_inputs(self, ambient_color: Vec3, directional_color: Vec3, sun_dir: Vec3, sun_light_node: NodePath = None, sun_lens = None):
        """Apply shared lighting inputs to the scene graph so all shaders see them."""
        if not hasattr(self.renderer.backend, 'scene_graph'):
            return

        # Safety floors to prevent accidental all-black output.
        if ambient_color.length() < 0.01:
            ambient_color = Vec3(0.22, 0.22, 0.26)
        if directional_color.length() < 0.01:
            directional_color = Vec3(0.85, 0.85, 0.80)
        if sun_dir.length() < 0.001:
            sun_dir = Vec3(-0.4, -0.4, 0.8)
        else:
            sun_dir.normalize()

        sg = self.renderer.backend.scene_graph

        # Legacy names used by older shaders still in the repo.
        sg.setShaderInput("u_ambient_color", Vec4(ambient_color[0], ambient_color[1], ambient_color[2], 1.0))
        sg.setShaderInput("u_sun_color", Vec4(directional_color[0], directional_color[1], directional_color[2], 1.0))
        sg.setShaderInput("u_sun_direction", sun_dir)

        # Unified forward pipeline names (world_pbr + character_toon).
        sg.setShaderInput("u_ambientColor", Vec3(ambient_color[0], ambient_color[1], ambient_color[2]))
        sg.setShaderInput("u_lightDirection", sun_dir)
        sg.setShaderInput("u_lightColor", Vec3(directional_color[0], directional_color[1], directional_color[2]))
        sg.setShaderInput("u_shadowBias", 0.0015)
        sg.setShaderInput("u_shadowPcfRadius", 1.0)

        # Default PBR parameters for world shader (can be overridden per-object)
        sg.setShaderInput("u_metallic", 0.0)
        sg.setShaderInput("u_roughness", 0.8)
        sg.setShaderInput("u_ao", 1.0)
        sg.setShaderInput("u_normal_scale", 1.0)
        sg.setShaderInput("u_emissive_color", Vec3(0.0, 0.0, 0.0))
        sg.setShaderInput("u_emissive_strength", 0.0)
        
        # Calculate and set Shadow MVP if we have a sun with shadows
        use_shadows = False
        if sun_light_node and sun_lens:
            # View Matrix: World -> Light
            # light_node.getMat(sg) gives Light -> World
            # We want World -> Light, so invert it.
            view_mat = sun_light_node.getMat(sg)
            view_mat.invertInPlace()
            
            # Projection Matrix
            proj_mat = sun_lens.getProjectionMat()
            
            # Bias Matrix (Map [-1, 1] to [0, 1])
            bias_mat = LMatrix4f(
                0.5, 0.0, 0.0, 0.0,
                0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.5, 0.5, 0.5, 1.0
            )
            
            # MVP = View * Proj * Bias
            # Panda matrices are Row-Major, so we multiply in order: v * View * Proj * Bias
            mvp = view_mat * proj_mat * bias_mat
            
            # IMPORTANT: Transpose for GLSL if Panda doesn't do it automatically for shader inputs
            # Panda usually handles this, but let's be safe if we are manually constructing it.
            # Actually, Panda's setShaderInput for matrix expects row-major, and GLSL expects column-major.
            # Panda automatically transposes when sending to GLSL.
            
            sg.setShaderInput("u_light_mvp", mvp)
            use_shadows = True
        else:
            # Provide a dummy matrix if no shadows are active to prevent shader errors
            sg.setShaderInput("u_light_mvp", LMatrix4f.identMat())
            
        sg.setShaderInput("u_use_shadows", 1 if use_shadows else 0)
