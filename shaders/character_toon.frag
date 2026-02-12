#version 330

// Shared directional light + shadow logic (same file as world shader).
#pragma include "lighting.glsl"

in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec2 v_uv;
in vec4 v_color;
in vec4 v_shadowCoord;

uniform sampler2D p3d_Texture0;
uniform sampler2D u_albedoMap;
uniform vec4 u_baseColor;
uniform vec3 u_ambientColor;
uniform vec3 u_shadowTint;
uniform vec3 u_cameraWorldPos;
uniform float u_useVertexColor;

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
    vec3 albedoTex = texture(p3d_Texture0, v_uv).rgb;
    if (length(albedoTex) < 0.001) {
        albedoTex = texture(u_albedoMap, v_uv).rgb;
    }
    vec3 vertexColor = mix(vec3(1.0), v_color.rgb, clamp(u_useVertexColor, 0.0, 1.0));
    vec3 albedo = albedoTex * u_baseColor.rgb * vertexColor;
    if (length(albedo) < 0.01) {
        albedo = max(u_baseColor.rgb, vec3(0.20));
    }
    vec3 viewDirWS = normalize(u_cameraWorldPos - v_worldPos);
    vec3 normalWS = normalize(v_worldNormal);
    // Two-sided toon shading for thin cloth/underarm geometry.
    if (!gl_FrontFacing) {
        normalWS = -normalWS;
    }
    normalWS = faceforward(normalWS, -viewDirWS, normalWS);
    vec3 lightDirWS = normalize(u_lightDirection);

    // Wrap diffuse to avoid crushing underclothes to pure black.
    float nDotL = clamp((dot(normalWS, lightDirWS) + 0.25) / 1.25, 0.0, 1.0);

    // Shared shadow logic from lighting.glsl (same as world shader).
    float shadowBias = computeShadowBias(normalWS);
    float shadowFactor = getShadowFactor(v_shadowCoord, shadowBias);

    // Requirement: apply shadow BEFORE ramp thresholding.
    float rampInput = nDotL * max(shadowFactor, 0.35);
    float band = computeToonBand(rampInput, u_toonThreshold1, u_toonThreshold2);

    vec3 ambientLighting = albedo * max(u_ambientColor, vec3(0.08));
    vec3 litBandColor = albedo * u_lightColor * band;

    // Reuse shared light function for a small stylized specular accent.
    vec3 specAccent = computeDirectionalLight(normalWS, viewDirWS, vec3(0.0), 0.20) * shadowFactor;

    // Final lighting is already shadowed through rampInput/specAccent.
    vec3 finalColor = mix(albedo * u_shadowTint, ambientLighting + litBandColor + specAccent * 0.20, band);
    p3d_FragColor = vec4(finalColor, u_baseColor.a);
}
