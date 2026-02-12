#ifndef ETERNAE_LIGHTING_GLSL
#define ETERNAE_LIGHTING_GLSL

// Shared Panda3D light struct.
// This file is included by both world and character fragment shaders so they
// use identical directional-light and shadow sampling logic.
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

// World-space vector that points FROM the shaded point TO the directional light.
uniform vec3 u_lightDirection;

// Base bias for shadow depth comparison. Typical values: 0.0008 - 0.003.
uniform float u_shadowBias;

// Sample spread in texels for 3x3 PCF. 1.0 = immediate neighbors.
uniform float u_shadowPcfRadius;

// Helper used by world and character shaders to get identical slope-scaled bias.
float computeShadowBias(vec3 normalWS) {
    vec3 n = normalize(normalWS);
    vec3 l = normalize(u_lightDirection);
    float nDotL = max(dot(n, l), 0.0);
    float slopeBias = u_shadowBias * (1.0 - nDotL);
    return max(slopeBias, u_shadowBias * 0.25);
}

// Core shadow lookup with explicit bias. Both shaders call this so the sampling
// path is guaranteed to remain identical.
float getShadowFactor(vec4 shadowCoord, float shadowBias) {
    if (shadowCoord.w <= 0.0) {
        return 1.0;
    }

    // Perspective divide from homogeneous light clip space to [0, 1] UVZ.
    vec3 proj = shadowCoord.xyz / shadowCoord.w;

    // Outside the shadow map frustum means fully lit.
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z <= 0.0 || proj.z > 1.0) {
        return 1.0;
    }

    ivec2 mapSize = textureSize(p3d_LightSource[0].shadowMap, 0);
    if (mapSize.x <= 0 || mapSize.y <= 0) {
        return 1.0;
    }

    vec2 texelSize = 1.0 / vec2(mapSize);
    float radius = max(u_shadowPcfRadius, 1.0);
    float compareDepth = proj.z - shadowBias;

    // 3x3 PCF minimum as requested.
    float litSamples = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(float(x), float(y)) * texelSize * radius;
            litSamples += texture(
                p3d_LightSource[0].shadowMap,
                vec3(proj.xy + offset, compareDepth)
            );
        }
    }

    return litSamples / 9.0;
}

// Required public helper: uses shared default bias.
float getShadowFactor(vec4 shadowCoord) {
    float safeBias = max(u_shadowBias, 0.00001);
    return getShadowFactor(shadowCoord, safeBias);
}

// Shared stylized directional-light function.
// Returns direct-light contribution before shadows so each shader can decide
// how and where to apply its shadow factor.
vec3 computeDirectionalLight(
    vec3 normalWS,
    vec3 viewDirWS,
    vec3 albedo,
    float roughness
) {
    vec3 n = normalize(normalWS);
    vec3 v = normalize(viewDirWS);
    vec3 l = normalize(u_lightDirection);
    vec3 h = normalize(v + l);

    float nDotL = max(dot(n, l), 0.0);
    float nDotH = max(dot(n, h), 0.0);

    float r = clamp(roughness, 0.04, 1.0);
    float gloss = 1.0 - r;
    float specPower = mix(8.0, 96.0, gloss * gloss);
    float specStrength = mix(0.04, 0.20, gloss);
    float specular = pow(nDotH, specPower) * nDotL * specStrength;

    vec3 diffuse = albedo * nDotL;
    vec3 lightColor = p3d_LightSource[0].color.rgb;
    return (diffuse + vec3(specular)) * lightColor;
}

#endif
