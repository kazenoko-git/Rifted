#version 150

// Panda3D Inputs
in vec4 p3d_Vertex;
in vec3 p3d_Normal;

// Uniforms
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;

// Shadow related
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
    mat4 shadowViewMatrix;
};
uniform p3d_LightSourceParameters p3d_LightSource[1];

// Outputs
out vec3 fPosition;
out vec3 fNormal;
out vec4 v_shadow_pos;

void main() {
    // World Position
    vec4 worldPos = p3d_ModelMatrix * p3d_Vertex;
    fPosition = worldPos.xyz;

    // World Normal (Correctly handling non-uniform scaling)
    fNormal = mat3(transpose(inverse(p3d_ModelMatrix))) * p3d_Normal;

    // Shadow Coordinates
    // Use Panda's built-in shadow matrix which handles World->LightClip transformation
    v_shadow_pos = p3d_LightSource[0].shadowViewMatrix * worldPos;

    // Clip Space Position
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}