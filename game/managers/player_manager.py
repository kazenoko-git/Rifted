# game/managers/player_manager.py

import numpy as np
import json
import os
from typing import Dict, Optional

from aurora_engine.core.logging import get_logger
from aurora_engine.ecs.world import World
from aurora_engine.ecs.entity import Entity
from aurora_engine.scene.transform import Transform
from aurora_engine.rendering.mesh import MeshRenderer
from aurora_engine.rendering.animator import Animator
from aurora_engine.physics.collider import CapsuleCollider, Collider
from aurora_engine.physics.rigidbody import RigidBody
from aurora_engine.input.input_manager import InputManager
from aurora_engine.camera.camera import Camera
from aurora_engine.camera.third_person import ThirdPersonController

from game.components.player import PlayerController
from game.systems.player_system import PlayerSystem
from game.systems.player_action_system import PlayerActionSystem

logger = get_logger()

class PlayerManager:
    """
    Manages the player entity, including creation, character switching,
    and camera setup.
    """
    
    def __init__(self, world: World, input_manager: InputManager, physics_world, renderer):
        self.world = world
        self.input = input_manager
        self.physics = physics_world
        self.renderer = renderer
        
        self.player_entity: Optional[Entity] = None
        self.camera_controller: Optional[ThirdPersonController] = None
        self.main_camera: Optional[Camera] = None
        
        # Default configuration
        self.current_character_config = self._load_config("game/config/male_mc.json")

    def _load_config(self, path: str) -> Dict:
        """Loads character configuration from a JSON file."""
        if not os.path.exists(path):
            logger.error(f"Character config file not found: {path}")
            # Fallback hardcoded config to prevent crash if file missing
            return {
                "model_path": "assets/characters/playable/maleMC/maleMC.glb",
                "animations": {
                    "Idle": {"path": "assets/characters/playable/maleMC/idle.glb", "speed": 1.0},
                    "Walk": {"path": "assets/characters/playable/maleMC/walk.glb", "speed": 1.2},
                    "Run": {"path": "assets/characters/playable/maleMC/run.glb", "speed": 1.0}
                },
                "collider": {
                    "radius": 0.25,
                    "height": 1.9,
                    "offset": [0.0, 0.0, 0.9]
                },
                "mass": 80.0
            }
            
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load character config {path}: {e}")
            return {}

    def create_player(self, position: np.ndarray) -> Entity:
        """Creates the player entity at the specified position."""
        if self.player_entity:
            self.world.destroy_entity(self.player_entity)
            
        self.player_entity = self.world.create_entity()
        transform = self.player_entity.add_component(Transform())
        transform.set_world_position(position)
        
        self._apply_character_config(self.player_entity, self.current_character_config)
        
        # Add Logic Components
        self.player_entity.add_component(PlayerController())
        
        # Setup Camera
        self._setup_camera(transform)
        
        # Register Systems
        self._register_systems()
        
        return self.player_entity

    def switch_character(self, config_path: str):
        """Switches the player character to a new configuration."""
        new_config = self._load_config(config_path)
        if not new_config:
            return

        self.current_character_config = new_config
        
        if self.player_entity:
            # Preserve position
            transform = self.player_entity.get_component(Transform)
            pos = transform.get_world_position()
            rot = transform.rotation
            
            # Recreate player
            self.create_player(pos)
            self.player_entity.get_component(Transform).rotation = rot

    def get_position(self) -> np.ndarray:
        """Returns the current player position."""
        if self.player_entity:
            return self.player_entity.get_component(Transform).get_world_position()
        return np.zeros(3, dtype=np.float32)

    def get_camera_transform(self) -> Optional[Transform]:
        """Returns the main camera transform."""
        if self.main_camera:
            return self.main_camera.transform
        return None

    def _apply_character_config(self, entity: Entity, config: Dict):
        """Applies visual and physical properties from config."""
        # Visuals
        if "model_path" in config:
            entity.add_component(MeshRenderer(model_path=config["model_path"], shading_model="character"))
        
        # Animation
        if "animations" in config:
            animator = entity.add_component(Animator())
            for name, data in config["animations"].items():
                animator.add_clip(name, path=data["path"], speed=data.get("speed", 1.0))
            animator.play("Idle")
        
        # Physics
        if "collider" in config:
            col_cfg = config["collider"]
            collider = Collider(CapsuleCollider(radius=col_cfg["radius"], height=col_cfg["height"]))
            collider.offset = np.array(col_cfg["offset"], dtype=np.float32)
            entity.add_component(collider)
        
        rb = entity.add_component(RigidBody())
        rb.mass = config.get("mass", 80.0)
        rb.use_gravity = True
        rb.lock_rotation = True

    def _setup_camera(self, target_transform: Transform):
        """Sets up the third-person camera."""
        if not self.main_camera:
            self.main_camera = Camera()
            self.renderer.register_camera(self.main_camera)
            
        self.camera_controller = ThirdPersonController(self.main_camera, target_transform, self.input)
        self.camera_controller.physics_world = self.physics
        
        # Lock mouse
        self.input.set_mouse_lock(True)

    def _register_systems(self):
        """Ensures player systems are registered and updated."""
        # Check if systems exist, if not add them
        
        # Find PlayerSystem
        player_system = None
        for sys in self.world.systems:
            if isinstance(sys, PlayerSystem):
                player_system = sys
                break
        
        if not player_system:
            player_system = PlayerSystem(self.input)
            self.world.add_system(player_system)
            
        # Update camera reference in PlayerSystem
        if self.main_camera:
            player_system.camera_transform = self.main_camera.transform
            
        # Ensure PlayerActionSystem exists
        has_action_system = any(isinstance(s, PlayerActionSystem) for s in self.world.systems)
        if not has_action_system:
            self.world.add_system(PlayerActionSystem(self.input))

    def update(self, dt: float, alpha: float):
        """Update camera controller."""
        if self.camera_controller:
            self.camera_controller.update(dt, alpha)
