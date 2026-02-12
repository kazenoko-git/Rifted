#version 330

// Toon + shadow vertex shader for Panda3D
// - Emits homogeneous shadow coords for sampler2DShadow lookup
// - Keeps normals in view space to match Panda3D light vectors (p3d_LightSource position)

// Vertex attributes provided by Panda3D
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

// Built-in transforms
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix; // view-space normal transform

// Light info (subset of Panda3D's p3d_LightSourceParameters)
struct p3d_LightSourceParameters {
    vec4 color;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec4 position;          // view-space; w = 0 for directional
    vec3 spotDirection;
    float spotExponent;
    float spotCutoff;
    float spotCosCutoff;
    vec3 attenuation;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;  // world -> light clip (no bias)
};
uniform p3d_LightSourceParameters p3d_LightSource[1];

// Varyings to fragment shader
out vec4 v_shadowCoord;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 worldPos = p3d_ModelMatrix * p3d_Vertex;

    // Transform world position into light clip space (bias happens in fragment when dividing by w)
    v_shadowCoord = p3d_LightSource[0].shadowViewMatrix * worldPos;

    // View-space normal to match light position space
    v_normal = normalize(p3d_NormalMatrix * p3d_Normal);

    v_uv = p3d_MultiTexCoord0;

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
