# aurora_engine/rendering/panda_backend.py

import numpy as np
from panda3d.core import *
from aurora_engine.rendering.mesh import Mesh
import weakref
from aurora_engine.core.logging import get_logger
from aurora_engine.utils.profiler import profile_section
import os
import sys

logger = get_logger()

class PandaBackend:
    """
    Panda3D rendering backend adapter.
    Isolates Panda3D-specific code from engine.
    """

    def __init__(self, config: dict):
        self.config = config
        self.window = None
        self.scene_graph = None
        self.base = None
        
        # Use a standard dictionary for explicit control
        self._mesh_cache = {}
        # logger.debug("PandaBackend initialized")

    def initialize(self):
        """Initialize Panda3D."""
        # Apply custom patch for bufferView KeyError EARLY
        # This must happen before any GLTF loading or simplepbr init
        self._patch_gltf_loader()

        # Load config
        # AGGRESSIVE MEMORY OPTIMIZATION
        # Added gl-version 3 2 to force Core Profile on macOS for GLSL 1.50+ support
        load_prc_file_data("", f"""
            win-size {self.config.get('width', 1920)} {self.config.get('height', 1080)}
            window-title {self.config.get('title', 'Aurora Engine')}
            framebuffer-multisample 1
            multisamples 2
            framebuffer-stencil 1
            framebuffer-srgb 1
            gl-coordinate-system default
            gl-version 3 2
            
            # --- Memory Optimization ---
            # Cache models to disk to avoid reprocessing
            # model-cache-dir {os.path.abspath('.panda3d_cache')}
            # model-cache-textures 1
            
            # Compress textures in RAM (Huge savings)
            compressed-textures 1
            driver-generate-mipmaps 1
            
            # Limit texture size (Downscale 4k/8k textures)
            # Increased to 4096 to support high-res shadow maps
            max-texture-dimension 4096
            
            # Don't keep a RAM copy of textures if they are on GPU
            # (Might cause hiccups if VRAM fills up, but saves system RAM)
            preload-textures 1
            
            # Aggressive Garbage Collection
            garbage-collect-states 1
            
            # Reduce Geom cache
            geom-cache-size 5000
            
            # Transform cache
            transform-cache-size 5000
        """)

        # Create window
        from direct.showbase.ShowBase import ShowBase
        # Check if ShowBase is already initialized
        import builtins
        if hasattr(builtins, 'base'):
            self.base = builtins.base
        else:
            self.base = ShowBase()
            
        self.window = self.base.win

        # Setup scene graph
        self.scene_graph = self.base.render
        
        # Disable default mouse control
        self.base.disableMouse()
        
        # Set Clear Color to Sky Blue
        self.base.setBackgroundColor(0.53, 0.8, 0.92, 1)
        
        # Ensure current directory is in model path
        getModelPath().appendDirectory(os.getcwd())
        
        # Register gltf loader
        # Try importing panda3d-complexpbr (module name is complexpbr)
        has_complexpbr = False
        try:
            import complexpbr
            # complexpbr.apply_shader() applies the shader to a node
            # We will apply it to the scene graph later or per-object
            # But first, let's see if we can initialize it globally or if it needs per-node setup.
            # complexpbr usually works by calling complexpbr.apply_shader(node)
            # It also handles shadows if configured.
            
            # We will use complexpbr instead of simplepbr if available
            has_complexpbr = True
            logger.info("Found panda3d-complexpbr")
        except ImportError:
            logger.warning("panda3d-complexpbr not found.")
            has_complexpbr = False

        # Fallback to simplepbr if complexpbr is not found (or if we want to support both)
        # The user requested complexpbr specifically for world shaders.
        
        # If complexpbr is present, we might still need simplepbr for other things or just use complexpbr.
        # Let's try to use complexpbr logic in Renderer.
        
        # For now, let's just ensure GLTF loading works.
        try:
            import gltf
            if hasattr(gltf, 'patch_loader'):
                gltf.patch_loader(self.base.loader)
                logger.info("Patched loader with panda3d-gltf")
        except ImportError:
            logger.warning("panda3d-gltf not found. GLB models will not load.")

        # We rely on game systems (DayNightCycle) to provide lighting.
        # self._setup_default_lighting()

        logger.info("Panda3D initialized")

    def _setup_default_lighting(self):
        """Setup basic lighting to ensure visibility."""
        # Ambient Light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.scene_graph.attachNewNode(alight)
        self.scene_graph.setLight(alnp)

        # Directional Light (Sun)
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        # Shadows are handled by DayNightCycle system, but we can enable them here for default scene
        # dlight.setShadowCaster(True, 2048, 2048)
        
        dlnp = self.scene_graph.attachNewNode(dlight)
        dlnp.setHpr(45, -60, 0)
        self.scene_graph.setLight(dlnp)
        
        logger.info("Default lighting initialized")

    def _patch_gltf_loader(self):
        """Patch panda3d-gltf to handle missing bufferView in accessors."""
        logger.info("Attempting to patch gltf loader...")
        try:
            # Force import of the exact modules used in traceback
            import gltf._loader
            import gltf._converter
            
            # --- Define Patched Functions ---
            
            def patched_load_model(*args, **kwargs):
                # logger.info(f"patched_load_model called with {len(args)} args")
                # Scan args for gltf_data (it's a dict with 'accessors')
                found = False
                for arg in args:
                    if isinstance(arg, dict) and 'accessors' in arg:
                        found = True
                        count = 0
                        for acc in arg['accessors']:
                            if 'bufferView' not in acc:
                                acc['bufferView'] = 0
                                count += 1
                        if count > 0:
                            logger.info(f"Sanitized {count} accessors in gltf_data (args)")
                
                if not found and 'gltf_data' in kwargs:
                     arg = kwargs['gltf_data']
                     if isinstance(arg, dict) and 'accessors' in arg:
                         count = 0
                         for acc in arg['accessors']:
                            if 'bufferView' not in acc:
                                acc['bufferView'] = 0
                                count += 1
                         if count > 0:
                            logger.info(f"Sanitized {count} accessors in gltf_data (kwargs)")

                return gltf._loader._original_load_model(*args, **kwargs)

            def patched_update(self, gltf_data, *args, **kwargs):
                # Sanitize accessors again just to be sure
                if isinstance(gltf_data, dict) and 'accessors' in gltf_data:
                    count = 0
                    for acc in gltf_data['accessors']:
                        if 'bufferView' not in acc:
                            acc['bufferView'] = 0
                            count += 1
                    if count > 0:
                        logger.info(f"Sanitized {count} accessors in update")
                            
                return gltf._converter.GltfConverter._original_update(self, gltf_data, *args, **kwargs)

            def patched_load_primitive(self, node, gltf_primitive, gltf_mesh, gltf_data, *args, **kwargs):
                try:
                    return gltf._converter.GltfConverter._original_load_primitive(self, node, gltf_primitive, gltf_mesh, gltf_data, *args, **kwargs)
                except KeyError as e:
                    if 'bufferView' in str(e):
                        logger.error("Caught bufferView KeyError in load_primitive. Attempting recovery.")
                        # Last ditch effort: fix ALL accessors and retry
                        if 'accessors' in gltf_data:
                            for acc in gltf_data['accessors']:
                                if 'bufferView' not in acc:
                                    acc['bufferView'] = 0
                        return gltf._converter.GltfConverter._original_load_primitive(self, node, gltf_primitive, gltf_mesh, gltf_data, *args, **kwargs)
                    raise e
            
            # --- PATCH SORTING KEYERROR ---
            # The traceback shows the error happens in a lambda inside load_primitive (or called by it)
            # specifically: accessors = sorted(accessors, key=lambda x: x['bufferView'])
            # We need to patch GltfConverter.load_primitive to catch this specific sort error if possible,
            # OR patch the data before it gets there.
            # Since we are already patching update/load_model to inject bufferView=0, it SHOULD work.
            # But maybe the data structure is nested or copied?
            
            # Let's try to patch the lambda? No, that's hard.
            # Let's ensure 'bufferView' is present in ALL accessors recursively.

            # --- Apply Patches ---

            # Patch 1: gltf._loader.load_model
            if not getattr(gltf._loader, '_is_patched_by_aurora_load', False):
                gltf._loader._original_load_model = gltf._loader.load_model
                gltf._loader.load_model = patched_load_model
                gltf._loader._is_patched_by_aurora_load = True
                logger.info("Successfully patched gltf._loader.load_model")
            
            # Patch 2: gltf._converter.GltfConverter methods
            TargetClass = gltf._converter.GltfConverter
            
            if not getattr(TargetClass, '_is_patched_by_aurora_update', False):
                TargetClass._original_update = TargetClass.update
                TargetClass.update = patched_update
                TargetClass._is_patched_by_aurora_update = True
                logger.info("Successfully patched gltf._converter.GltfConverter.update")
            
            if not getattr(TargetClass, '_is_patched_by_aurora_prim', False):
                TargetClass._original_load_primitive = TargetClass.load_primitive
                TargetClass.load_primitive = patched_load_primitive
                TargetClass._is_patched_by_aurora_prim = True
                logger.info("Successfully patched gltf._converter.GltfConverter.load_primitive")

            # Patch 3: Force update GltfConverter in gltf._loader namespace
            if hasattr(gltf._loader, 'GltfConverter'):
                # If it's a different object, we need to patch it too
                LoaderConverter = gltf._loader.GltfConverter
                if LoaderConverter is not TargetClass:
                    logger.info("gltf._loader.GltfConverter is a different class object. Patching it.")
                    LoaderConverter._original_update = LoaderConverter.update
                    LoaderConverter.update = patched_update
                    LoaderConverter._original_load_primitive = LoaderConverter.load_primitive
                    LoaderConverter.load_primitive = patched_load_primitive
                else:
                    logger.info("gltf._loader.GltfConverter is the same class object.")

            # Patch 4: Iterate through sys.modules to find ANY loaded GltfConverter
            # This is the nuclear option for when imports are messy
            for module_name, module in list(sys.modules.items()):
                if hasattr(module, 'GltfConverter'):
                    cls = getattr(module, 'GltfConverter')
                    if cls is not TargetClass and cls is not getattr(gltf._loader, 'GltfConverter', None):
                        logger.info(f"Found another GltfConverter in {module_name}. Patching it.")
                        if not getattr(cls, '_is_patched_by_aurora_update', False):
                            cls._original_update = cls.update
                            cls.update = patched_update
                            cls._is_patched_by_aurora_update = True
                        
                        if not getattr(cls, '_is_patched_by_aurora_prim', False):
                            cls._original_load_primitive = cls.load_primitive
                            cls.load_primitive = patched_load_primitive
                            cls._is_patched_by_aurora_prim = True

        except ImportError as e:
            logger.warning(f"Could not import gltf module for patching: {e}")
        except AttributeError as e:
            logger.warning(f"Failed to patch gltf loader (AttributeError): {e}")
        except Exception as e:
            logger.warning(f"Failed to patch gltf loader: {e}")

    def clear_buffers(self):
        """Clear color and depth buffers."""
        # Panda3D handles this automatically
        # But we need to call taskMgr.step() somewhere if we are running our own loop.
        if self.base:
            with profile_section("PandaTaskStep"):
                self.base.taskMgr.step()

    def update_camera_transform(self, pos: np.ndarray, rot: np.ndarray):
        """Update Panda3D camera node transform."""
        if self.base and self.base.camera:
            self.base.camera.setPos(pos[0], pos[1], pos[2])
            # Panda Quat is (w, x, y, z)
            self.base.camera.setQuat(Quat(rot[3], rot[0], rot[1], rot[2]))

    def set_view_projection(self, view: np.ndarray, projection: np.ndarray):
        """Set camera matrices."""
        # Panda3D manages view matrix via camera node transform (set in update_camera_transform)
        # So we don't need to manually set view matrix here unless we are overriding it.
        # However, we might want to set projection matrix (Lens)
        
        # TODO: Update Lens properties if projection changes (FOV, etc.)
        pass

    def draw_mesh(self, mesh: Mesh, world_matrix: np.ndarray):
        """Draw a mesh with given transform."""
        if not mesh:
            return

        # Check cache
        if mesh not in self._mesh_cache:
            self._upload_mesh(mesh)
            
        # This method is for immediate mode which we are not fully using.
        # See update_mesh_node for retained mode updates.
        pass

    def update_mesh_node(self, node_path: NodePath, world_matrix: np.ndarray):
        """Update transform of a node path using matrix."""
        # Slow path
        mat = LMatrix4f()
        for i in range(4):
            for j in range(4):
                # Transpose for Panda3D (Row-Major)
                mat.setCell(j, i, world_matrix[i, j])
        node_path.setMat(mat)

    def update_mesh_transform(self, node_path: NodePath, pos: np.ndarray, rot: np.ndarray, scale: np.ndarray):
        """Update transform of a node path using decomposed values (Faster)."""
        # pos: [x, y, z]
        # rot: [x, y, z, w] (Quaternion)
        # scale: [sx, sy, sz]
        
        node_path.setPos(pos[0], pos[1], pos[2])
        # Panda Quat is (w, x, y, z)
        node_path.setQuat(Quat(rot[3], rot[0], rot[1], rot[2]))
        node_path.setScale(scale[0], scale[1], scale[2])

    def create_mesh_node(self, mesh: Mesh) -> NodePath:
        """Create a NodePath for a mesh."""
        if mesh not in self._mesh_cache:
            self._upload_mesh(mesh)
            
        geom_node = self._mesh_cache[mesh]
        return NodePath(geom_node)
        
    def unload_mesh(self, mesh: Mesh):
        """Explicitly remove a mesh from the cache."""
        if mesh in self._mesh_cache:
            del self._mesh_cache[mesh]
            # logger.debug(f"Unloaded mesh '{mesh.name}' from backend")

    def _upload_mesh(self, mesh: Mesh):
        """Convert Mesh to Panda3D GeomNode."""
        with profile_section("UploadMesh"):
            # Use custom format with Tangent and Binormal for PBR
            array_format = GeomVertexArrayFormat()
            array_format.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
            array_format.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CVector)
            array_format.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
            array_format.addColumn(InternalName.getTexcoord(), 2, Geom.NTFloat32, Geom.CTexcoord)
            array_format.addColumn(InternalName.getTangent(), 3, Geom.NTFloat32, Geom.CVector)
            array_format.addColumn(InternalName.getBinormal(), 3, Geom.NTFloat32, Geom.CVector)
            
            format = GeomVertexFormat()
            format.addArray(array_format)
            format = GeomVertexFormat.registerFormat(format)
            
            vdata = GeomVertexData(mesh.name, format, Geom.UHStatic)
            
            # Vertices
            vertex = GeomVertexWriter(vdata, 'vertex')
            normal = GeomVertexWriter(vdata, 'normal')
            color = GeomVertexWriter(vdata, 'color')
            texcoord = GeomVertexWriter(vdata, 'texcoord')
            tangent = GeomVertexWriter(vdata, 'tangent')
            binormal = GeomVertexWriter(vdata, 'binormal')
            
            # Ensure tangents are calculated
            if len(mesh.tangents) == 0 and len(mesh.uvs) > 0:
                mesh.calculate_tangents()
            
            for i in range(len(mesh.vertices)):
                v = mesh.vertices[i]
                vertex.addData3(v[0], v[1], v[2])
                
                if len(mesh.normals) > i:
                    n = mesh.normals[i]
                    normal.addData3(n[0], n[1], n[2])
                    
                if mesh.colors is not None and len(mesh.colors) > i:
                    c = mesh.colors[i]
                    color.addData4(c[0], c[1], c[2], c[3])
                else:
                    # Default to White so node color works
                    color.addData4(1, 1, 1, 1)
                    
                if len(mesh.uvs) > i:
                    uv = mesh.uvs[i]
                    texcoord.addData2(uv[0], uv[1])
                    
                if len(mesh.tangents) > i:
                    t = mesh.tangents[i]
                    tangent.addData3(t[0], t[1], t[2])
                else:
                    tangent.addData3(1, 0, 0)
                    
                if len(mesh.binormals) > i:
                    b = mesh.binormals[i]
                    binormal.addData3(b[0], b[1], b[2])
                else:
                    binormal.addData3(0, 1, 0)
                    
            # Primitives
            geom = Geom(vdata)
            tris = GeomTriangles(Geom.UHStatic)
            
            if mesh.indices is not None:
                for i in range(0, len(mesh.indices), 3):
                    tris.addVertices(int(mesh.indices[i]), int(mesh.indices[i+1]), int(mesh.indices[i+2]))
            else:
                # Non-indexed
                for i in range(0, len(mesh.vertices), 3):
                    tris.addVertices(i, i+1, i+2)
                    
            geom.addPrimitive(tris)
            
            node = GeomNode(mesh.name)
            node.addGeom(geom)
            
            self._mesh_cache[mesh] = node
            # logger.debug(f"Uploaded mesh '{mesh.name}' to backend")

    def present(self):
        """Present rendered frame."""
        # Panda3D handles swap automatically
        pass

    def shutdown(self):
        """Shutdown Panda3D."""
        if self.base:
            # self.base.destroy() # Usually we don't destroy base if it's global
            pass
        logger.info("PandaBackend shutdown")
