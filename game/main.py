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
from game.managers.environment_manager import EnvironmentManager

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
        self.environment_manager = EnvironmentManager(self.world, self.renderer, self.world_manager)
        
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

        # Setup Environment (Day/Night, Fog, etc)
        self.environment_manager.setup(self.player.get_component(Transform))
        
        # Setup UI
        self.game_ui_manager.setup_ui()
        
        # Add Game Systems
        dialogue_system = DialogueSystem(self.ui)
        dialogue_system.ai_manager = self.ai_manager
        self.world.add_system(dialogue_system)
        
        # --- DIAGNOSTIC: LIGHTING TEST ---
        self._setup_diagnostic_lighting()

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

    def _setup_diagnostic_lighting(self):
        """Force a simple test case for lighting diagnostics."""
        logger.info("=== DIAGNOSTIC LIGHTING SETUP ===")
        from aurora_engine.rendering.light import DirectionalLight, AmbientLight
        from aurora_engine.rendering.mesh import MeshRenderer, create_cube_mesh, create_plane_mesh
        from aurora_engine.scene.transform import Transform
        from aurora_engine.utils.math import quaternion_from_euler
        from aurora_engine.physics.collider import Collider, BoxCollider
        from aurora_engine.physics.rigidbody import StaticBody
        import numpy as np

        # 1. Ground Plane (Receiver)
        ground = self.world.create_entity()
        ground.add_component(Transform())
        ground.get_component(Transform).set_world_position(np.array([0, 0, 0], dtype=np.float32))
        ground.get_component(Transform).set_local_scale(np.array([20, 20, 1], dtype=np.float32))
        ground.add_component(MeshRenderer(mesh=create_plane_mesh(), color=(0.5, 0.5, 0.5, 1.0), shading_model="world"))
        
        # Add Physics so player stands on it (Size 20x20x0.1)
        ground.add_component(Collider(BoxCollider(np.array([20.0, 20.0, 0.1], dtype=np.float32))))
        ground.add_component(StaticBody())

        # 2. Floating Cube (Caster)
        cube = self.world.create_entity()
        cube.add_component(Transform())
        cube.get_component(Transform).set_world_position(np.array([2.0, 0.0, 2.0], dtype=np.float32))
        # Rotate slightly to see shading
        q_cube = quaternion_from_euler(np.radians(np.array([45.0, 45.0, 0.0], dtype=np.float32)))
        cube.get_component(Transform).set_world_rotation(q_cube)
        cube.add_component(MeshRenderer(mesh=create_cube_mesh(), color=(1.0, 0.2, 0.2, 1.0), shading_model="character"))

        # 3. Directional Light (Sun)
        sun = self.world.create_entity()
        sun.add_component(Transform())
        sun.get_component(Transform).set_world_position(np.array([0, -10, 20], dtype=np.float32))
        # Look down-forward (Pitch -60)
        q_sun = quaternion_from_euler(np.radians(np.array([-60.0, 0.0, 0.0], dtype=np.float32)))
        sun.get_component(Transform).set_world_rotation(q_sun)
        
        dlight = DirectionalLight(color=(1.0, 0.95, 0.8), intensity=1.5)
        dlight.cast_shadows = True
        dlight.shadow_map_size = 2048
        dlight.shadow_film_size = 50.0 # Ensure it covers the scene
        sun.add_component(dlight)

        # 4. Weak Ambient Light
        amb = self.world.create_entity()
        amb.add_component(AmbientLight(color=(0.1, 0.1, 0.2), intensity=0.3))
        
        logger.info("Diagnostic Scene Created: Ground Plane, Red Cube, Sun (Shadows ON), Weak Ambient.")

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
