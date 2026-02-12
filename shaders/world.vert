#version 150

// Vertex attributes from Panda3D
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

// Matrices
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 u_light_mvp; // World -> Light clip (bias included), provided by LightSystem
struct p3d_LightSourceParameters {
    vec4 color;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec4 position;
    vec3 spotDirection;
    float spotExponent;
    float spotCutoff;
    float spotCosCutoff;
    vec3 attenuation;
    sampler2DShadow shadowMap;
    mat4 shadowMatrix;
};
uniform p3d_LightSourceParameters p3d_LightSource[1];

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

    // Prefer engine-provided light MVP (bias included) to avoid driver differences.
    // Fallback to Panda's built-in shadowMatrix if needed.
    vec4 sc_a = u_light_mvp * world_pos;
    vec4 sc_b = p3d_LightSource[0].shadowMatrix * world_pos;
    v_shadow_pos = (abs(sc_a.w) > 0.00001) ? sc_a : sc_b;

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
