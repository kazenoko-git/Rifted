# game/main.py

import os
import sys

# Add the project root to the Python path to enable imports from 'aurora_engine'.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np

from aurora_engine.core.application import Application
from aurora_engine.scene.transform import Transform
from aurora_engine.core.logging import get_logger
from aurora_engine.database.db_manager import DatabaseManager
from aurora_engine.database.schema import DatabaseSchema

from game.systems.dialogue_system import DialogueSystem
from game.systems.world_generator import WorldGenerator
from game.ai.ai_generator import AIContentGenerator

from game.managers.world_manager import WorldManager
from game.managers.player_manager import PlayerManager
from game.managers.ai_manager import AIManager
from game.managers.debug_manager import DebugManager
from game.managers.game_ui_manager import GameUIManager
from panda3d.core import Filename, getModelPath

logger = get_logger()

class Eternae(Application):
    """
    The Main RPG Game Application.
    """

    def initialize_game(self):
        """Game-specific initialization."""
        logger.info("Initializing Eternae Game...")
        
        # Initialize Database
        self._setup_database()
        
        # Initialize AI & World Gen
        self.ai_generator = AIContentGenerator(self.db_manager)
        self.world_generator = WorldGenerator(self.db_manager, self.ai_generator)
        
        # Initialize Managers
        self.world_manager = WorldManager(self.world, self.db_manager, self.world_generator)
        self.player_manager = PlayerManager(self.world, self.input, self.physics, self.renderer)
        self.ai_manager = AIManager(self.db_manager, self.ai_generator)

        self.debug_manager = DebugManager(self.world, self.renderer, self.input, self.physics, self.ui)
        self.game_ui_manager = GameUIManager(self.ui, self.config)

        # Unified forward pipeline: shared shadow include + world/character shader pair.
        self._configure_unified_forward_pipeline()
        
        # Setup World
        self.world_manager.initialize_world()
        
        # Create Player
        # Initial position will be adjusted after terrain load
        initial_pos = np.array([0, 0, 10.0], dtype=np.float32)
        self.player = self.player_manager.create_player(initial_pos)
        
        # Load Initial Area
        self.world_manager.load_initial_area(initial_pos)
        
        # Adjust Player Height
        h = self.world_manager.get_ground_height(0, 0)
        self.player.get_component(Transform).set_world_position(np.array([0, 0, h + 5.0], dtype=np.float32))

        # Unified environment: one directional light (with shadows) + one ambient fill.
        # Intentionally not using the old day/night stack to keep a single-light pipeline.
        self._setup_unified_lighting()
        
        # Setup UI
        self.game_ui_manager.setup_ui()
        
        # Add Game Systems
        dialogue_system = DialogueSystem(self.ui)
        dialogue_system.ai_manager = self.ai_manager
        self.world.add_system(dialogue_system)

    def _setup_database(self):
        """Initialize database connection."""
        db_config = self.config.get('database', {})
        if not db_config:
             db_config = {
                'database': 'eternae.db'
            }
            
        self.db_manager = DatabaseManager(db_config)
        self.db_manager.connect()
        DatabaseSchema.create_tables(self.db_manager)

    def _configure_unified_forward_pipeline(self):
        """Load and activate unified forward shaders for world and characters."""
        shader_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shaders"))
        getModelPath().appendDirectory(Filename.fromOsSpecific(shader_dir))

        world_shader = self.renderer._load_glsl_shader_with_includes(
            os.path.join(shader_dir, "world_pbr.vert"),
            os.path.join(shader_dir, "world_pbr.frag"),
        )
        toon_shader = self.renderer._load_glsl_shader_with_includes(
            os.path.join(shader_dir, "character_toon.vert"),
            os.path.join(shader_dir, "character_toon.frag"),
        )

        if world_shader:
            self.renderer._shader_world = world_shader
        if toon_shader:
            self.renderer._shader_toon = toon_shader

        if not world_shader or not toon_shader:
            missing = []
            if not world_shader:
                missing.append("world_pbr")
            if not toon_shader:
                missing.append("character_toon")
            raise RuntimeError(f"Unified shader compile failed: {', '.join(missing)}")

        # Keep the renderer on built-in shaders and disable complexpbr fallback.
        self.renderer.force_builtin_world_shader = True
        logger.info("Unified forward shaders configured (world_pbr + character_toon).")

    def _setup_unified_lighting(self):
        """Create a single shadow-casting directional light and one ambient fill."""
        from aurora_engine.rendering.light import DirectionalLight, AmbientLight
        from game.systems.day_night_cycle import DayNightCycle

        # Single directional light for the entire forward pipeline.
        self.sun = self.world.create_entity()
        self.sun.add_component(Transform())
        self.sun.get_component(Transform).set_world_position(np.array([160.0, -80.0, 220.0], dtype=np.float32))

        dlight = DirectionalLight(color=(1.0, 0.96, 0.90), intensity=1.35)
        dlight.cast_shadows = True
        dlight.shadow_map_size = 2048
        dlight.shadow_film_size = 900.0
        dlight.shadow_near_far = (1.0, 2000.0)
        self.sun.add_component(dlight)

        # Ambient fill (non-shadowing).
        self.ambient = self.world.create_entity()
        self.ambient.add_component(AmbientLight(color=(0.26, 0.27, 0.32), intensity=0.95))

        # Re-enable day/night movement with a single directional light.
        # This keeps lighting alive over time without adding moon/multi-shadow lights.
        day_night = DayNightCycle(self.renderer, day_duration=120.0)
        day_night.target = self.player.get_component(Transform)
        day_night.sun_entity = self.sun
        day_night.ambient_entity = self.ambient
        day_night.moon_entity = None
        day_night.orbit_radius = 520.0
        self.world.add_system(day_night)

        if hasattr(self.renderer.backend, "scene_graph"):
            sg = self.renderer.backend.scene_graph
            sg.setShaderInput("u_shadowBias", 0.0015)
            sg.setShaderInput("u_shadowPcfRadius", 1.0)

        logger.info("Unified lighting active: 1 directional light (shadow map 2048) + ambient fill.")

    def update(self, dt: float, alpha: float):
        """Override update to update managers."""
        super().update(dt, alpha)
        
        # Update Managers
        player_pos = self.player_manager.get_position()
        cam_transform = self.player_manager.get_camera_transform()
        
        if cam_transform:
            self.world_manager.update_chunks(dt, player_pos, cam_transform)
        
        self.player_manager.update(dt, alpha)
        self.ai_manager.update_emotions(dt)
        self.debug_manager.update(dt, player_pos)
        
        # Toggle mouse lock with Escape
        if self.input.is_key_down('escape'):
            self.input.set_mouse_lock(False)
        elif self.input.is_key_down('mouse1') and not self.input.mouse_locked:
            self.input.set_mouse_lock(True)
        
    def late_update(self, dt: float, alpha: float):
        """Update camera after physics."""
        # Player manager handles camera update in its update()
        pass

    def shutdown(self):
        """Cleanup."""
        super().shutdown()
        if hasattr(self, 'db_manager'):
            self.db_manager.disconnect()


if __name__ == "__main__":
    config_path = "config.json"
    if not os.path.exists(config_path):
        config_data = {
            'rendering': {
                'width': 1280,
                'height': 720,
                'title': 'Eternae',
            },
            'database': {
                'database': 'eternae.db'
            }
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    game = Eternae(config_path)
    game.run()
