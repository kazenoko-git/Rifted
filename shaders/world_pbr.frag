#version 330

// Shared directional light + shadow logic.
#pragma include "lighting.glsl"

in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec2 v_uv;
in vec4 v_color;
in vec4 v_shadowCoord;

uniform sampler2D p3d_Texture0;
uniform sampler2D u_albedoMap;
uniform sampler2D u_normalMap;
uniform sampler2D u_roughnessMap;

uniform vec4 u_baseColor;
uniform vec3 u_ambientColor;
uniform vec3 u_cameraWorldPos;
uniform float u_roughnessScale;
uniform float u_useVertexColor;

out vec4 p3d_FragColor;

// Derivative-based tangent frame. This avoids requiring mesh tangents and keeps
// the shader minimal while still supporting normal maps.
mat3 computeCotangentFrame(vec3 normalWS, vec3 worldPos, vec2 uv) {
    vec3 dp1 = dFdx(worldPos);
    vec3 dp2 = dFdy(worldPos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    vec3 dp2perp = cross(dp2, normalWS);
    vec3 dp1perp = cross(normalWS, dp1);
    vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;

    float denom = max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-8);
    float scale = inversesqrt(denom);
    return mat3(tangent * scale, bitangent * scale, normalWS);
}

vec3 sampleWorldNormal() {
    vec3 normalTS = texture(u_normalMap, v_uv).xyz * 2.0 - 1.0;
    mat3 tbn = computeCotangentFrame(normalize(v_worldNormal), v_worldPos, v_uv);
    return normalize(tbn * normalTS);
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
    float roughness = clamp(texture(u_roughnessMap, v_uv).r * u_roughnessScale, 0.04, 1.0);

    vec3 normalWS = sampleWorldNormal();
    vec3 viewDirWS = normalize(u_cameraWorldPos - v_worldPos);

    // Shared shadow logic from lighting.glsl.
    float shadowBias = computeShadowBias(normalWS);
    float shadowFactor = getShadowFactor(v_shadowCoord, shadowBias);

    vec3 directLighting = computeDirectionalLight(normalWS, viewDirWS, albedo, roughness);

    // Requirement: apply shadow factor before final color output.
    directLighting *= shadowFactor;

    vec3 ambientLighting = albedo * max(u_ambientColor, vec3(0.08));
    vec3 finalColor = ambientLighting + directLighting;
    p3d_FragColor = vec4(finalColor, u_baseColor.a);
}
