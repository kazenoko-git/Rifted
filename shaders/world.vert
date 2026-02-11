#version 150

// Vertex attributes from Panda3D
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Tangent; // xyz = tangent, w = handedness

// Matrices
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 u_light_mvp; // Light view-projection with bias applied by LightSystem

// Varyings
out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;
out vec4 v_shadow_pos;
out vec3 v_tangent;
out vec3 v_bitangent;

void main() {
    vec4 world_pos = p3d_ModelMatrix * p3d_Vertex;
    v_world_pos = world_pos.xyz;

    // Properly handle non-uniform scale for normals/tangents
    mat3 normal_mat = mat3(transpose(inverse(p3d_ModelMatrix)));
    v_world_normal = normalize(normal_mat * p3d_Normal);

    vec3 t_obj = p3d_Tangent.xyz;
    float handedness = (p3d_Tangent.w >= 0.0) ? 1.0 : -1.0;
    if (length(t_obj) < 0.001) {
        // Fallback tangent for meshes without tangents
        vec3 up = (abs(p3d_Normal.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
        t_obj = normalize(cross(up, p3d_Normal));
        handedness = 1.0;
    }

    v_tangent = normalize(normal_mat * t_obj);
    v_bitangent = normalize(cross(v_world_normal, v_tangent) * handedness);

    v_uv = p3d_MultiTexCoord0;

    // Light clip coords (already biased to 0..1 in LightSystem)
    v_shadow_pos = u_light_mvp * world_pos;

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
