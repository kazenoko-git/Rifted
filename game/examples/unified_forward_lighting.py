"""
Minimal Panda3D forward lighting setup:
- one directional light with built-in Panda shadow map
- one shared shadow include used by world + character shaders
- no auto-shader pipeline and no deferred/multi-pass setup
"""

import math
import os

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    DirectionalLight,
    Filename,
    PNMImage,
    Shader,
    Texture,
    Vec3,
    Vec4,
    getModelPath,
    loadPrcFileData,
)

loadPrcFileData("", "window-title Unified Forward Lighting")
loadPrcFileData("", "gl-version 3 3")
loadPrcFileData("", "framebuffer-srgb true")
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")


class UnifiedForwardLightingDemo(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.clock = ClockObject.getGlobalClock()

        self.cam.setPos(0.0, -18.0, 8.0)
        self.cam.lookAt(0.0, 0.0, 1.5)

        shader_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shaders"))
        getModelPath().appendDirectory(Filename.fromOsSpecific(shader_dir))

        self.world_shader = Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(shader_dir, "world_pbr.vert"),
            fragment=os.path.join(shader_dir, "world_pbr.frag"),
        )
        self.character_shader = Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(shader_dir, "character_toon.vert"),
            fragment=os.path.join(shader_dir, "character_toon.frag"),
        )

        # Tiny default textures keep the demo self-contained.
        self.tex_white = self._solid_texture("white", 1.0, 1.0, 1.0)
        self.tex_normal = self._solid_texture("normal", 0.5, 0.5, 1.0)
        self.tex_roughness = self._solid_texture("roughness", 0.75, 0.75, 0.75)

        self.world_np = self._create_world()
        self.character_np = self._create_character()
        self._setup_lights()
        self._setup_controls()
        self._bind_shaders()

        self.taskMgr.add(self._update_shader_inputs, "update-shader-inputs")

    def _solid_texture(self, name: str, r: float, g: float, b: float) -> Texture:
        image = PNMImage(1, 1, 4)
        image.fill(r, g, b)
        image.alphaFill(1.0)
        texture = Texture(name)
        texture.load(image)
        return texture

    def _create_world(self):
        # World geometry: a simple ground plane.
        cm = CardMaker("ground")
        cm.setFrame(-20.0, 20.0, -20.0, 20.0)
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90.0)
        ground.setPos(0.0, 0.0, 0.0)
        return ground

    def _create_character(self):
        # Character geometry: loaded model.
        character = self.loader.loadModel("models/smiley")
        character.reparentTo(self.render)
        character.setScale(2.0)
        character.setPos(0.0, 0.0, 2.0)
        return character

    def _setup_lights(self):
        # Ambient fill (not a shadow caster).
        ambient = AmbientLight("ambient-fill")
        ambient.setColor((0.15, 0.18, 0.22, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        # Single directional light with Panda3D built-in shadow map.
        sun = DirectionalLight("sun")
        sun.setColor((1.0, 0.96, 0.90, 1.0))
        sun.setShadowCaster(True, 2048, 2048)
        sun_lens = sun.getLens()
        sun_lens.setFilmSize(40.0, 40.0)
        sun_lens.setNearFar(1.0, 100.0)

        self.sun_np = self.render.attachNewNode(sun)
        self.sun_np.setPos(16.0, -20.0, 28.0)
        self.sun_np.lookAt(0.0, 0.0, 0.0)
        self.render.setLight(self.sun_np)

    def _setup_controls(self):
        # Light controls:
        # - WASD: move light rig in X/Y world plane
        # - L: toggle orbit/rotation around scene center
        self.light_move_speed = 18.0
        self.light_orbit_speed_deg = 30.0
        self.light_orbit_enabled = False
        self.light_target = Vec3(0.0, 0.0, 0.0)
        self.light_move_keys = {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
        }

        self.accept("l", self._toggle_light_orbit)

        for key in self.light_move_keys:
            self.accept(key, self._set_light_move_key, [key, True])
            self.accept(f"{key}-up", self._set_light_move_key, [key, False])

    def _set_light_move_key(self, key: str, is_down: bool):
        self.light_move_keys[key] = is_down

    def _toggle_light_orbit(self):
        self.light_orbit_enabled = not self.light_orbit_enabled
        print(f"[Light] Orbit {'ON' if self.light_orbit_enabled else 'OFF'}")

    def _update_light_transform(self, dt: float):
        pos = self.sun_np.getPos(self.render)
        move = Vec3(0.0, 0.0, 0.0)

        if self.light_move_keys["w"]:
            move.y += 1.0
        if self.light_move_keys["s"]:
            move.y -= 1.0
        if self.light_move_keys["a"]:
            move.x -= 1.0
        if self.light_move_keys["d"]:
            move.x += 1.0

        if move.lengthSquared() > 0.0:
            move.normalize()
            pos += move * self.light_move_speed * dt
            self.sun_np.setPos(self.render, pos)

        if self.light_orbit_enabled:
            pivot = self.light_target
            relative = self.sun_np.getPos(self.render) - pivot
            angle = math.radians(self.light_orbit_speed_deg * dt)
            c = math.cos(angle)
            s = math.sin(angle)
            rotated = Vec3(
                relative.x * c - relative.y * s,
                relative.x * s + relative.y * c,
                relative.z,
            )
            self.sun_np.setPos(self.render, pivot + rotated)

        # Keep directional light aimed at the scene center for intuitive controls.
        self.sun_np.lookAt(self.light_target)

    def _vector_to_light(self) -> Vec3:
        # Panda directional light shines down local +Y.
        # We need a vector from shaded point TO light source, so negate forward.
        forward = self.sun_np.getQuat(self.render).xform(Vec3(0.0, 1.0, 0.0))
        if forward.lengthSquared() > 0.0:
            forward.normalize()
        return -forward

    def _set_common_inputs(self, node):
        node.setShaderInput("u_lightDirection", self._vector_to_light())
        node.setShaderInput("u_shadowBias", 0.0015)
        node.setShaderInput("u_shadowPcfRadius", 1.0)
        node.setShaderInput("u_ambientColor", Vec3(0.15, 0.18, 0.22))
        node.setShaderInput("u_cameraWorldPos", self.cam.getPos(self.render))

    def _bind_shaders(self):
        # Separate shaders for world and character.
        self.world_np.setShader(self.world_shader)
        self.character_np.setShader(self.character_shader)

        self._set_common_inputs(self.world_np)
        self._set_common_inputs(self.character_np)

        # World PBR inputs.
        self.world_np.setShaderInput("u_albedoMap", self.tex_white)
        self.world_np.setShaderInput("u_normalMap", self.tex_normal)
        self.world_np.setShaderInput("u_roughnessMap", self.tex_roughness)
        self.world_np.setShaderInput("u_baseColor", Vec4(0.62, 0.67, 0.58, 1.0))
        self.world_np.setShaderInput("u_roughnessScale", 1.0)

        # Character toon inputs.
        self.character_np.setShaderInput("u_albedoMap", self.tex_white)
        self.character_np.setShaderInput("u_baseColor", Vec4(0.90, 0.65, 0.45, 1.0))
        self.character_np.setShaderInput("u_shadowTint", Vec3(0.20, 0.22, 0.28))
        self.character_np.setShaderInput("u_toonThreshold1", 0.35)
        self.character_np.setShaderInput("u_toonThreshold2", 0.70)

    def _update_shader_inputs(self, task):
        # Keep camera/light-dependent uniforms current.
        dt = self.clock.getDt()
        self._update_light_transform(dt)

        cam_pos = self.cam.getPos(self.render)
        light_dir = self._vector_to_light()

        self.world_np.setShaderInput("u_cameraWorldPos", cam_pos)
        self.character_np.setShaderInput("u_cameraWorldPos", cam_pos)

        self.world_np.setShaderInput("u_lightDirection", light_dir)
        self.character_np.setShaderInput("u_lightDirection", light_dir)
        return task.cont


if __name__ == "__main__":
    app = UnifiedForwardLightingDemo()
    app.run()
