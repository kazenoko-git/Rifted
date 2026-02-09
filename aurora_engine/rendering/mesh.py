# aurora_engine/rendering/mesh.py

import numpy as np
from typing import List, Optional, Tuple
from aurora_engine.ecs.component import Component
from aurora_engine.rendering.material import Material
from aurora_engine.core.logging import get_logger
from aurora_engine.utils.profiler import profile_section

logger = get_logger()

class Mesh:
    """
    Mesh data container.
    Stores vertices, normals, UVs, indices.
    """

    def __init__(self, name: str = "Mesh"):
        self.name = name

        # Vertex data
        self.vertices: np.ndarray = np.array([], dtype=np.float32)  # Nx3
        self.normals: np.ndarray = np.array([], dtype=np.float32)  # Nx3
        self.uvs: np.ndarray = np.array([], dtype=np.float32)  # Nx2
        self.colors: Optional[np.ndarray] = None  # Nx4
        self.tangents: np.ndarray = np.array([], dtype=np.float32) # Nx3
        self.binormals: np.ndarray = np.array([], dtype=np.float32) # Nx3

        # Index buffer (for indexed rendering)
        self.indices: Optional[np.ndarray] = None

        # Bounds (for culling)
        self.bounds_min: np.ndarray = np.zeros(3, dtype=np.float32)
        self.bounds_max: np.ndarray = np.zeros(3, dtype=np.float32)

        # Backend handle (Panda3D geometry)
        self._backend_handle = None
        
        # logger.debug(f"Mesh '{name}' created")

    def calculate_bounds(self):
        """Calculate bounding box from vertices."""
        if len(self.vertices) > 0:
            self.bounds_min = np.min(self.vertices, axis=0)
            self.bounds_max = np.max(self.vertices, axis=0)

    def calculate_normals(self):
        """Calculate smooth normals from geometry."""
        with profile_section("CalcNormals"):
            if self.indices is None or len(self.indices) == 0:
                return

            num_verts = len(self.vertices)
            normals = np.zeros((num_verts, 3), dtype=np.float32)

            # Accumulate face normals
            for i in range(0, len(self.indices), 3):
                i0, i1, i2 = self.indices[i], self.indices[i + 1], self.indices[i + 2]

                v0 = self.vertices[i0]
                v1 = self.vertices[i1]
                v2 = self.vertices[i2]

                # Calculate face normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_normal = np.cross(edge1, edge2)

                # Accumulate to vertices
                normals[i0] += face_normal
                normals[i1] += face_normal
                normals[i2] += face_normal

            # Normalize
            # Vectorized normalization for speed
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            self.normals = normals / norms

    def calculate_tangents(self):
        """Calculate tangents and binormals."""
        with profile_section("CalcTangents"):
            if self.indices is None or len(self.indices) == 0 or len(self.uvs) == 0:
                return

            num_verts = len(self.vertices)
            tangents = np.zeros((num_verts, 3), dtype=np.float32)
            binormals = np.zeros((num_verts, 3), dtype=np.float32)

            for i in range(0, len(self.indices), 3):
                i0, i1, i2 = self.indices[i], self.indices[i + 1], self.indices[i + 2]

                v0 = self.vertices[i0]
                v1 = self.vertices[i1]
                v2 = self.vertices[i2]

                uv0 = self.uvs[i0]
                uv1 = self.uvs[i1]
                uv2 = self.uvs[i2]

                delta_pos1 = v1 - v0
                delta_pos2 = v2 - v0

                delta_uv1 = uv1 - uv0
                delta_uv2 = uv2 - uv0

                r = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0] + 1e-6)
                
                tangent = (delta_pos1 * delta_uv2[1] - delta_pos2 * delta_uv1[1]) * r
                binormal = (delta_pos2 * delta_uv1[0] - delta_pos1 * delta_uv2[0]) * r

                tangents[i0] += tangent
                tangents[i1] += tangent
                tangents[i2] += tangent

                binormals[i0] += binormal
                binormals[i1] += binormal
                binormals[i2] += binormal

            # Normalize
            t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            t_norms[t_norms == 0] = 1.0
            self.tangents = tangents / t_norms
            
            b_norms = np.linalg.norm(binormals, axis=1, keepdims=True)
            b_norms[b_norms == 0] = 1.0
            self.binormals = binormals / b_norms


class MeshRenderer(Component):
    """
    Mesh renderer component.
    Renders a mesh with a material.
    """

    def __init__(
        self,
        mesh: Optional[Mesh] = None,
        material: Optional[Material] = None,
        color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        model_path: Optional[str] = None,
        texture_path: Optional[str] = None,
        shading_model: str = "world",  # "world" (smooth) or "character" (toon)
    ):
        super().__init__()

        self.mesh = mesh
        self.material = material
        self.color = color # Simple color override if no material
        self.model_path = model_path
        self.texture_path = texture_path # Path to texture image
        self.shading_model = shading_model
        self.alpha = 1.0 # For fade-in effects
        self._node_path = None # Handle to Panda3D node

        # Cache for transform optimization
        self._last_pos = None
        self._last_rot = None
        self._last_scale = None
        self._applied_shader = None

        # Rendering settings
        self.cast_shadows = True
        self.receive_shadows = True
        self.visible = True
        self.billboard = False # If true, always face camera

        # Toon-specific defaults (used only when shading_model == "character")
        self.toon_bands = 3.0
        self.shadow_color = (0.1, 0.1, 0.3, 1.0)
        
        # logger.debug("MeshRenderer component created")

    def on_destroy(self):
        """Clean up Panda3D node when component is destroyed."""
        if self._node_path:
            # Check if it's an Actor and call cleanup
            if hasattr(self._node_path, 'cleanup'):
                self._node_path.cleanup()
            self._node_path.removeNode()
            self._node_path = None
            # logger.debug("MeshRenderer node removed")


# Primitive mesh generators
def create_cube_mesh(size: float = 1.0) -> Mesh:
    """Create a cube mesh."""
    mesh = Mesh("Cube")

    s = size / 2.0
    
    # Explicitly define 24 vertices (4 per face) to ensure sharp edges (normals)
    # Winding order: Counter-Clockwise (CCW) for Front Facing
    
    vertices = [
        # Front Face (Y+)
        [-s, s, -s], [ s, s, -s], [ s, s,  s], [-s, s,  s], # 0, 1, 2, 3
        # Back Face (Y-)
        [ s, -s, -s], [-s, -s, -s], [-s, -s,  s], [ s, -s,  s], # 4, 5, 6, 7
        # Left Face (X-)
        [-s, -s, -s], [-s,  s, -s], [-s,  s,  s], [-s, -s,  s], # 8, 9, 10, 11
        # Right Face (X+)
        [ s,  s, -s], [ s, -s, -s], [ s, -s,  s], [ s,  s,  s], # 12, 13, 14, 15
        # Top Face (Z+)
        [-s,  s,  s], [ s,  s,  s], [ s, -s,  s], [-s, -s,  s], # 16, 17, 18, 19
        # Bottom Face (Z-)
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s], # 20, 21, 22, 23
    ]

    # Normals
    normals = [
        [ 0,  1,  0], [ 0,  1,  0], [ 0,  1,  0], [ 0,  1,  0], # Front
        [ 0, -1,  0], [ 0, -1,  0], [ 0, -1,  0], [ 0, -1,  0], # Back
        [-1,  0,  0], [-1,  0,  0], [-1,  0,  0], [-1,  0,  0], # Left
        [ 1,  0,  0], [ 1,  0,  0], [ 1,  0,  0], [ 1,  0,  0], # Right
        [ 0,  0,  1], [ 0,  0,  1], [ 0,  0,  1], [ 0,  0,  1], # Top
        [ 0,  0, -1], [ 0,  0, -1], [ 0,  0, -1], [ 0,  0, -1], # Bottom
    ]

    # UVs
    uvs = [[0, 0], [1, 0], [1, 1], [0, 1]] * 6

    # Indices (CCW)
    # 0, 1, 2, 0, 2, 3 pattern for each face
    indices = [
        # Front
        0, 1, 2, 0, 2, 3,
        # Back
        4, 7, 6, 4, 6, 5,
        # Left
        8, 11, 10, 8, 10, 9,
        # Right
        12, 15, 14, 12, 14, 13,
        # Top
        16, 19, 18, 16, 18, 17,
        # Bottom
        20, 23, 22, 20, 22, 21
    ]

    mesh.vertices = np.array(vertices, dtype=np.float32)
    mesh.normals = np.array(normals, dtype=np.float32)
    mesh.uvs = np.array(uvs, dtype=np.float32)
    mesh.indices = np.array(indices, dtype=np.uint32)

    mesh.calculate_bounds()
    mesh.calculate_tangents() # Calculate tangents for primitives

    return mesh


def create_sphere_mesh(radius: float = 1.0, segments: int = 16, rings: int = 8) -> Mesh:
    """Create a UV sphere mesh."""
    mesh = Mesh("Sphere")

    vertices = []
    normals = []
    uvs = []
    indices = []

    # Generate vertices (Z-up)
    for ring in range(rings + 1):
        phi = ring * np.pi / rings # 0 to pi (top to bottom)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        for seg in range(segments + 1):
            theta = seg * 2.0 * np.pi / segments # 0 to 2pi (around Z)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            x = sin_phi * cos_theta
            y = sin_phi * sin_theta
            z = cos_phi

            vertices.append([x * radius, y * radius, z * radius])
            normals.append([x, y, z])
            uvs.append([seg / segments, ring / rings])

    # Generate indices
    for ring in range(rings):
        for seg in range(segments):
            current = ring * (segments + 1) + seg
            next_seg = current + 1
            next_ring = current + segments + 1
            next_both = next_ring + 1

            # CCW Winding:
            # current (TL) -> next_ring (BL) -> next_seg (TR)
            # next_seg (TR) -> next_ring (BL) -> next_both (BR)
            indices.extend([current, next_ring, next_seg])
            indices.extend([next_seg, next_ring, next_both])

    mesh.vertices = np.array(vertices, dtype=np.float32)
    mesh.normals = np.array(normals, dtype=np.float32)
    mesh.uvs = np.array(uvs, dtype=np.float32)
    mesh.indices = np.array(indices, dtype=np.uint32)

    mesh.calculate_bounds()
    mesh.calculate_tangents()

    return mesh


def create_plane_mesh(width: float = 1.0, height: float = 1.0) -> Mesh:
    """Create a plane mesh (XY plane, Z-up)."""
    mesh = Mesh("Plane")

    w = width / 2.0
    h = height / 2.0

    # XY Plane (Z=0)
    vertices = [
        [-w, h, 0],  # 0: Top-Left
        [w, h, 0],   # 1: Top-Right
        [w, -h, 0],  # 2: Bottom-Right
        [-w, -h, 0], # 3: Bottom-Left
    ]

    normals = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]

    uvs = [
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
    ]

    # CCW Winding for +Z normal
    # 0(TL) -> 2(BR) -> 1(TR)
    # 0(TL) -> 3(BL) -> 2(BR)
    indices = [0, 2, 1, 0, 3, 2]

    mesh.vertices = np.array(vertices, dtype=np.float32)
    mesh.normals = np.array(normals, dtype=np.float32)
    mesh.uvs = np.array(uvs, dtype=np.float32)
    mesh.indices = np.array(indices, dtype=np.uint32)

    mesh.calculate_bounds()
    mesh.calculate_tangents()

    return mesh

def create_capsule_mesh(radius: float = 0.5, height: float = 1.0, segments: int = 12, rings: int = 6) -> Mesh:
    """Create a capsule mesh (cylinder with hemispherical caps)."""
    mesh = Mesh("Capsule")
    
    vertices = []
    normals = []
    uvs = []
    indices = []
    
    # Cylinder height is the straight part. Total height = height + 2*radius.
    # Let's assume 'height' is the total height.
    cyl_height = max(0, height - 2 * radius)
    half_cyl = cyl_height / 2.0
    
    # Top Hemisphere
    for ring in range(rings + 1):
        phi = ring * (np.pi / 2) / rings # 0 to pi/2
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        for seg in range(segments + 1):
            theta = seg * 2.0 * np.pi / segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            x = sin_phi * cos_theta * radius
            y = sin_phi * sin_theta * radius
            z = cos_phi * radius + half_cyl
            
            vertices.append([x, y, z])
            normals.append([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi])
            uvs.append([seg / segments, ring / (rings * 2 + 1)]) # Approx UV
            
    # Bottom Hemisphere
    offset_idx = len(vertices)
    for ring in range(rings + 1):
        phi = (np.pi / 2) + ring * (np.pi / 2) / rings # pi/2 to pi
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        for seg in range(segments + 1):
            theta = seg * 2.0 * np.pi / segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            x = sin_phi * cos_theta * radius
            y = sin_phi * sin_theta * radius
            z = cos_phi * radius - half_cyl
            
            vertices.append([x, y, z])
            normals.append([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi])
            uvs.append([seg / segments, 0.5 + ring / (rings * 2 + 1)])
            
    # Indices (similar to sphere but split)
    # Top Cap
    for ring in range(rings):
        for seg in range(segments):
            curr = ring * (segments + 1) + seg
            next_seg = curr + 1
            next_ring = curr + (segments + 1)
            next_both = next_ring + 1
            
            indices.extend([curr, next_ring, next_seg])
            indices.extend([next_seg, next_ring, next_both])
            
    # Bottom Cap
    for ring in range(rings):
        for seg in range(segments):
            curr = offset_idx + ring * (segments + 1) + seg
            next_seg = curr + 1
            next_ring = curr + (segments + 1)
            next_both = next_ring + 1
            
            indices.extend([curr, next_ring, next_seg])
            indices.extend([next_seg, next_ring, next_both])
            
    # Cylinder Body (Connect bottom of top cap to top of bottom cap)
    # Top cap bottom ring is at index: rings * (segments + 1)
    # Bottom cap top ring is at index: offset_idx
    top_ring_start = rings * (segments + 1)
    bottom_ring_start = offset_idx
    
    for seg in range(segments):
        t1 = top_ring_start + seg
        t2 = top_ring_start + seg + 1
        b1 = bottom_ring_start + seg
        b2 = bottom_ring_start + seg + 1
        
        # Quad: t1, b1, b2, t2
        # CCW: t1 -> b1 -> t2, t2 -> b1 -> b2
        indices.extend([t1, b1, t2])
        indices.extend([t2, b1, b2])
        
    mesh.vertices = np.array(vertices, dtype=np.float32)
    mesh.normals = np.array(normals, dtype=np.float32)
    mesh.uvs = np.array(uvs, dtype=np.float32)
    mesh.indices = np.array(indices, dtype=np.uint32)
    
    mesh.calculate_bounds()
    mesh.calculate_tangents()
    return mesh
