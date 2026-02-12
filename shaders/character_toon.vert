#version 330

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;

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

out vec3 v_worldPos;
out vec3 v_worldNormal;
out vec2 v_uv;
out vec4 v_shadowCoord;

void main() {
    vec4 worldPos = p3d_ModelMatrix * p3d_Vertex;
    v_worldPos = worldPos.xyz;

    mat3 normalMatrix = mat3(transpose(inverse(p3d_ModelMatrix)));
    v_worldNormal = normalize(normalMatrix * p3d_Normal);

    v_uv = p3d_MultiTexCoord0;

    // Required shadow matrix source for consistency with world shader.
    v_shadowCoord = p3d_LightSource[0].shadowMatrix * worldPos;

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
