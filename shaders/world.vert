#version 150

// Vertex attributes from Panda3D
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

// Matrices
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 u_light_mvp; // Light view-projection with bias applied by LightSystem

// Varyings
out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;
out vec4 v_shadow_pos;

void main() {
    vec4 world_pos = p3d_ModelMatrix * p3d_Vertex;
    v_world_pos = world_pos.xyz;

    // Properly handle non-uniform scale for normals/tangents
    mat3 normal_mat = mat3(transpose(inverse(p3d_ModelMatrix)));
    v_world_normal = normalize(normal_mat * p3d_Normal);

    v_uv = p3d_MultiTexCoord0;

    // Light clip coords (already biased to 0..1 in LightSystem)
    v_shadow_pos = u_light_mvp * world_pos;

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
