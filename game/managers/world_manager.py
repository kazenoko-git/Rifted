# game/managers/world_manager.py

import random
import time
import json
import numpy as np
from typing import Dict, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor

from aurora_engine.core.logging import get_logger
from aurora_engine.database.db_manager import DatabaseManager
from aurora_engine.ecs.world import World
from aurora_engine.ecs.entity import Entity
from aurora_engine.scene.transform import Transform
from aurora_engine.rendering.mesh import MeshRenderer, create_plane_mesh
from aurora_engine.physics.collider import MeshCollider, BoxCollider, Collider
from aurora_engine.physics.rigidbody import StaticBody
from aurora_engine.utils.profiler import profile_section
from panda3d.core import Material

from game.systems.world_generator import WorldGenerator
from game.components.fade_in import FadeInEffect
from game.utils.chunk_worker import generate_chunk_meshes
from game.utils.terrain import get_height_at_world_pos

logger = get_logger()

class WorldManager:
    """
    Manages the game world, including chunk loading/unloading,
    dimension management, and world generation coordination.
    """
    
    def __init__(self, world: World, db_manager: DatabaseManager, world_generator: WorldGenerator):
        self.world = world
        self.db_manager = db_manager
        self.world_generator = world_generator
        
        # Chunk Management
        self.loaded_chunks: Dict[Tuple[int, int], list] = {} 
        self.pending_data: Dict[Tuple[int, int], object] = {} 
        self.pending_meshes: Dict[Tuple[int, int], object] = {} 
        
        # Configuration
        self.chunk_size = 100.0
        self.render_radius_chunks = 5
        self.fog_radius = (self.render_radius_chunks - 1) * self.chunk_size
        
        # State
        self.current_dimension_id = None
        self.last_chunk_check = 0.0
        
        # Executors
        self.mesh_executor = ThreadPoolExecutor(max_workers=4)

    def initialize_world(self):
        """Initializes the main game world."""
        seed = random.randint(0, 2**32 - 1)
        self.current_dimension_id = f"dim_main_{seed}"
        dim = self.world_generator.get_or_create_dimension(self.current_dimension_id, seed)
        logger.info(f"Entered Dimension: {dim['name']} (Seed: {seed})")

    def load_initial_area(self, center_pos: np.ndarray):
        """Synchronously loads the area around the given position."""
        logger.info("Loading initial area...")
        
        current_chunk_x = int(center_pos[0] / self.chunk_size)
        current_chunk_y = int(center_pos[1] / self.chunk_size)
        
        radius = self.render_radius_chunks
        for x in range(current_chunk_x - radius, current_chunk_x + radius + 1):
            for y in range(current_chunk_y - radius, current_chunk_y + radius + 1):
                if (x - current_chunk_x)**2 + (y - current_chunk_y)**2 <= radius**2:
                    region = self.world_generator.generate_region(self.current_dimension_id, x, y)
                    meshes = generate_chunk_meshes(region)
                    self._instantiate_chunk(region, meshes, fade_in=False)

    def update_chunks(self, dt: float, player_pos: np.ndarray, camera_transform: Transform):
        """Updates chunk loading based on player position and camera."""
        self.last_chunk_check += dt
        if self.last_chunk_check > 0.5:
            self._manage_chunks(player_pos, camera_transform)
            self.last_chunk_check = 0.0
            
        self._process_futures()

    def get_ground_height(self, x: float, y: float) -> float:
        """Gets the ground height at a specific world position."""
        # This is a simplified check. Ideally we query the loaded chunks or physics.
        # For initialization, we might need to generate the region data on fly if not loaded.
        chunk_x = int(x / self.chunk_size)
        chunk_y = int(y / self.chunk_size)
        
        # Try to get from world generator cache first
        region_id = f"{self.current_dimension_id}_{chunk_x}_{chunk_y}"
        if region_id in self.world_generator.known_regions:
            region = self.world_generator.known_regions[region_id]
            return get_height_at_world_pos(x, y, region, cell_size=self.chunk_size/20.0)
            
        # Fallback: Generate region synchronously (might be slow)
        region = self.world_generator.generate_region(self.current_dimension_id, chunk_x, chunk_y)
        if region:
            return get_height_at_world_pos(x, y, region, cell_size=self.chunk_size/20.0)
            
        return 0.0

    def _manage_chunks(self, player_pos: np.ndarray, camera_transform: Transform):
        """Handle loading and unloading of chunks."""
        current_chunk_x = int(player_pos[0] / self.chunk_size)
        current_chunk_y = int(player_pos[1] / self.chunk_size)
        
        cam_pos = camera_transform.get_world_position()
        cam_fwd = camera_transform.forward
        cam_fwd_3d = np.array([cam_fwd[0], cam_fwd[1], cam_fwd[2]], dtype=np.float32)
        if np.linalg.norm(cam_fwd_3d) > 0.01:
            cam_fwd_3d = cam_fwd_3d / np.linalg.norm(cam_fwd_3d)
        else:
            cam_fwd_3d = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        needed_chunks = set()
        keep_loaded_chunks = set()
        
        check_radius = self.render_radius_chunks + 1
        fov_half_rad = np.radians(60)
        chunk_radius = 100.0
        
        for x in range(current_chunk_x - check_radius, current_chunk_x + check_radius + 1):
            for y in range(current_chunk_y - check_radius, current_chunk_y + check_radius + 1):
                chunk_center_x = x * self.chunk_size + self.chunk_size / 2
                chunk_center_y = y * self.chunk_size + self.chunk_size / 2
                chunk_center_z = player_pos[2]
                
                to_chunk = np.array([
                    chunk_center_x - cam_pos[0], 
                    chunk_center_y - cam_pos[1],
                    chunk_center_z - cam_pos[2]
                ], dtype=np.float32)
                
                dist = np.linalg.norm(to_chunk)
                dist_2d = np.sqrt(to_chunk[0]**2 + to_chunk[1]**2)
                chunk_dist = dist_2d / self.chunk_size
                
                if chunk_dist <= 2.0:
                    needed_chunks.add((x, y))
                    keep_loaded_chunks.add((x, y))
                    continue
                    
                if chunk_dist <= self.render_radius_chunks:
                    if dist > 1.0:
                        to_chunk_norm = to_chunk / dist
                        dot = np.dot(to_chunk_norm, cam_fwd_3d)
                        dot = max(-1.0, min(1.0, dot))
                        angle = np.arccos(dot)
                        angular_radius = np.arcsin(min(1.0, chunk_radius / dist))
                        
                        if angle - angular_radius < fov_half_rad:
                            needed_chunks.add((x, y))
                            keep_loaded_chunks.add((x, y))
                            continue

        # Unload
        chunks_to_unload = []
        for coords, entities in self.loaded_chunks.items():
            if coords not in keep_loaded_chunks:
                chunks_to_unload.append(coords)
                continue
            
            should_be_visible = coords in needed_chunks
            for entity in entities:
                mesh_renderer = entity.get_component(MeshRenderer)
                if mesh_renderer and mesh_renderer.visible != should_be_visible:
                    mesh_renderer.visible = should_be_visible
                    if hasattr(mesh_renderer, '_node_path') and mesh_renderer._node_path:
                        if should_be_visible:
                            mesh_renderer._node_path.show()
                        else:
                            mesh_renderer._node_path.hide()

        for coords in chunks_to_unload:
            self._unload_chunk(coords)

        # Load
        max_concurrent_loads = 4
        current_loads = len(self.pending_data) + len(self.pending_meshes)
        
        sorted_needed = sorted(list(needed_chunks), key=lambda c: (c[0]*self.chunk_size - player_pos[0])**2 + (c[1]*self.chunk_size - player_pos[1])**2)
        
        for coords in sorted_needed:
            if current_loads >= max_concurrent_loads:
                break
            if coords not in self.loaded_chunks and coords not in self.pending_data and coords not in self.pending_meshes:
                self._request_chunk_load(coords)
                current_loads += 1

    def _request_chunk_load(self, coords: Tuple[int, int]):
        future = self.world_generator.generate_region_async(self.current_dimension_id, coords[0], coords[1])
        self.pending_data[coords] = future

    def _process_futures(self):
        # Stage 1: Data Gen
        completed_data = []
        for coords, future in self.pending_data.items():
            if future.done():
                try:
                    region_data = future.result()
                    mesh_future = self.mesh_executor.submit(generate_chunk_meshes, region_data)
                    self.pending_meshes[coords] = mesh_future
                except Exception as e:
                    logger.error(f"Chunk data generation failed for {coords}: {e}")
                completed_data.append(coords)
        for coords in completed_data:
            del self.pending_data[coords]
            
        # Stage 2: Mesh Gen
        completed_meshes = []
        for coords, future in self.pending_meshes.items():
            if future.done():
                try:
                    meshes = future.result()
                    region_data = self.world_generator.generate_region(self.current_dimension_id, coords[0], coords[1])
                    self._instantiate_chunk(region_data, meshes, fade_in=True)
                except Exception as e:
                    logger.error(f"Chunk mesh generation failed for {coords}: {e}")
                completed_meshes.append(coords)
        for coords in completed_meshes:
            del self.pending_meshes[coords]

    def _instantiate_chunk(self, region_data: Dict, meshes: Dict, fade_in: bool = True):
        coords = (region_data['coordinates_x'], region_data['coordinates_y'])
        if coords in self.loaded_chunks:
            return
            
        chunk_entities = []
        
        # Props
        for entity_data, mesh in meshes['props']:
            e = self.world.create_entity()
            t = e.add_component(Transform())
            t.set_world_position(np.array([entity_data['x'], entity_data['y'], entity_data['z']], dtype=np.float32))
            
            if 'model_path' in entity_data:
                e.add_component(MeshRenderer(model_path=entity_data['model_path'], shading_model="world"))
            else:
                e.add_component(MeshRenderer(mesh=mesh, color=(1.0, 1.0, 1.0, 1.0), shading_model="world"))
                
            if fade_in: e.add_component(FadeInEffect(duration=0.5))
            
            if entity_data['type'] == 'prop':
                if entity_data['model'] == 'rock':
                    e.add_component(Collider(MeshCollider(mesh, convex=True)))
                elif entity_data['model'] == 'tree':
                    scale = entity_data.get('scale', 1.0)
                    e.add_component(Collider(BoxCollider(np.array([0.5 * scale, 0.5 * scale, 4.0 * scale], dtype=np.float32))))
            elif entity_data['type'] == 'structure':
                e.add_component(Collider(BoxCollider(np.array([4.0, 3.0, 3.0], dtype=np.float32))))
            
            e.add_component(StaticBody())
            chunk_entities.append(e)
                
        # Terrain
        if meshes['terrain']:
            ground = self.world.create_entity()
            gt = ground.add_component(Transform())
            rx = region_data['coordinates_x'] * 100.0
            ry = region_data['coordinates_y'] * 100.0
            gt.set_world_position(np.array([rx, ry, 0], dtype=np.float32))
            
            # Create material for terrain
            from aurora_engine.rendering.material import Material as EngineMaterial
            # We can't easily use EngineMaterial here because it's not fully integrated with Panda's Material
            # But we can rely on Renderer to apply a default Panda Material if none exists.
            # However, to be safe, let's just rely on the Renderer's default material logic which we updated.
            
            ground.add_component(MeshRenderer(mesh=meshes['terrain'], color=(1.0, 1.0, 1.0, 1.0), shading_model="world"))
            if fade_in: ground.add_component(FadeInEffect(duration=0.5))
            
            ground.add_component(StaticBody())
            ground.add_component(Collider(MeshCollider(meshes['terrain'], convex=False)))
            chunk_entities.append(ground)
            
        # Water
        water = self.world.create_entity()
        wt = water.add_component(Transform())
        rx = region_data['coordinates_x'] * 100.0
        ry = region_data['coordinates_y'] * 100.0
        wt.set_world_position(np.array([rx + 50.0, ry + 50.0, -2.0], dtype=np.float32))
        wt.local_scale = np.array([100.0, 100.0, 1.0], dtype=np.float32)
        
        water_mesh = create_plane_mesh(1.0, 1.0)
        water.add_component(MeshRenderer(mesh=water_mesh, color=(0.2, 0.4, 0.8, 0.8), shading_model="world"))
        water.add_component(Collider(BoxCollider(np.array([100.0, 100.0, 1.0], dtype=np.float32))))
        if fade_in: water.add_component(FadeInEffect(duration=0.5))
        water.add_component(StaticBody())
        chunk_entities.append(water)
        
        self.loaded_chunks[coords] = chunk_entities

    def _unload_chunk(self, coords: Tuple[int, int]):
        if coords in self.loaded_chunks:
            entities = self.loaded_chunks[coords]
            for entity in entities:
                self.world.destroy_entity(entity)
            del self.loaded_chunks[coords]
