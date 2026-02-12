#version 330

// Toon + shadow fragment shader for Panda3D
// - Samples Panda3D's shadow map with hardware depth comparison
// - Quantizes diffuse term into 3 bands for a simple toon ramp
// - Uses the light's shadowViewMatrix output from the vertex shader

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
    mat4 shadowViewMatrix;
};
uniform p3d_LightSourceParameters p3d_LightSource[1];
uniform vec4 u_object_color;  // per-object color supplied by engine (fallback if no texture)

in vec4 v_shadowCoord;
in vec3 v_normal;
in vec2 v_uv;

out vec4 p3d_FragColor;

// Dummy use of v_uv to silence driver warning; real albedo comes from u_object_color for now.
// You can swap in a texture sample here later.
vec3 sample_albedo(vec2 uv) {
    return u_object_color.rgb;
}

// Simple 3x3 PCF around the current pixel for softer edges
float sample_shadow(vec4 shadowCoord, float bias) {
    // Perspective divide and bias to [0,1]
    vec3 proj = shadowCoord.xyz / shadowCoord.w;

    // Outside the light frustum -> lit
    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
        return 1.0;
    }

    float shadow = 0.0;
    vec2 texel = 1.0 / textureSize(p3d_LightSource[0].shadowMap, 0);

    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec3 offset = vec3(proj.xy + vec2(x, y) * texel, proj.z - bias);
            shadow += texture(p3d_LightSource[0].shadowMap, offset);
        }
    }
    return shadow / 9.0;
}

void main() {
    // View-space normal to match Panda3D light direction
    vec3 N = normalize(v_normal);
    // Panda3D passes directional light direction in position.xyz (view space), w = 0
    vec3 L = normalize(-p3d_LightSource[0].position.xyz);

    float NdotL = max(dot(N, L), 0.0);

    // --- Toon ramp: 3 discrete bands ---
    float toon;
    if (NdotL > 0.66) toon = 1.0;          // fully lit
    else if (NdotL > 0.33) toon = 0.6;     // mid tone
    else toon = 0.3;                       // dark tone

    // --- Shadow sampling ---
    float bias = max(0.0005, 0.002 * (1.0 - NdotL)); // reduce acne on grazed angles
    float shadow = sample_shadow(v_shadowCoord, bias);

    vec3 baseColor = sample_albedo(v_uv);       // placeholder albedo (texture hook)
    vec3 lightColor = p3d_LightSource[0].color.rgb;

    // Multiply toon by shadow factor (1=lit, 0=occluded)
    vec3 finalColor = baseColor * lightColor * toon * shadow;

    p3d_FragColor = vec4(finalColor, 1.0);
}
