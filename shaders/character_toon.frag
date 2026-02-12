#version 330

// Shared directional light + shadow logic (same file as world shader).
#pragma include "lighting.glsl"

in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec2 v_uv;
in vec4 v_shadowCoord;

uniform sampler2D u_albedoMap;
uniform vec4 u_baseColor;
uniform vec3 u_ambientColor;
uniform vec3 u_shadowTint;
uniform vec3 u_cameraWorldPos;

// 3-band toon thresholds:
// lit < t1 -> dark band, t1..t2 -> mid band, lit >= t2 -> bright band.
uniform float u_toonThreshold1;
uniform float u_toonThreshold2;

out vec4 p3d_FragColor;

float computeToonBand(float value, float t1, float t2) {
    if (value < t1) {
        return 0.20;
    }
    if (value < t2) {
        return 0.60;
    }
    return 1.00;
}

void main() {
    vec3 albedo = texture(u_albedoMap, v_uv).rgb * u_baseColor.rgb;
    vec3 normalWS = normalize(v_worldNormal);
    vec3 viewDirWS = normalize(u_cameraWorldPos - v_worldPos);
    vec3 lightDirWS = normalize(u_lightDirection);

    float nDotL = max(dot(normalWS, lightDirWS), 0.0);

    // Shared shadow logic from lighting.glsl (same as world shader).
    float shadowBias = computeShadowBias(normalWS);
    float shadowFactor = getShadowFactor(v_shadowCoord, shadowBias);

    // Requirement: apply shadow BEFORE ramp thresholding.
    float rampInput = nDotL * shadowFactor;
    float band = computeToonBand(rampInput, u_toonThreshold1, u_toonThreshold2);

    vec3 ambientLighting = albedo * u_ambientColor;
    vec3 litBandColor = albedo * p3d_LightSource[0].color.rgb * band;

    // Reuse shared light function for a small stylized specular accent.
    vec3 specAccent = computeDirectionalLight(normalWS, viewDirWS, vec3(0.0), 0.20) * shadowFactor;

    // Final lighting is already shadowed through rampInput/specAccent.
    vec3 finalColor = mix(albedo * u_shadowTint, ambientLighting + litBandColor + specAccent * 0.20, band);
    p3d_FragColor = vec4(finalColor, u_baseColor.a);
}
