#version 150
/*
 * WORLD SHADER (Cook-Torrance PBR)
 * - Uses glTF-style inputs when available (albedo in p3d_Texture0, metallic/roughness in p3d_Texture1 G/B, normal map in p3d_Texture2, AO in p3d_Texture3).
 * - Falls back gracefully to flat color uniforms.
 * - Shadowing driven by LightSystem (u_light_mvp + p3d_LightShadowMap0).
 */

// Inputs from vertex shader
in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in vec4 v_shadow_pos;

// Camera
uniform vec3 p3d_CameraPosition;

// Textures (optional, safe to leave unbound)
uniform sampler2D p3d_Texture0;        // Albedo
uniform sampler2D p3d_Texture1;        // Metallic (B) / Roughness (G)
uniform sampler2D p3d_Texture2;        // Reserved for future normal-map support
uniform sampler2D p3d_Texture3;        // AO
uniform sampler2D p3d_Texture4;        // Emissive
uniform sampler2DShadow p3d_LightShadowMap0;

// Scene inputs
uniform vec4 u_object_color;    // Base color multiplier
uniform vec4 u_ambient_color;   // From LightSystem
uniform vec4 u_sun_color;       // From LightSystem
uniform vec3 u_sun_direction;   // World-space direction TO light
uniform int  u_use_shadows;     // 1 when shadow map valid

// Material tweak uniforms
uniform float u_metallic      = 0.0;
uniform float u_roughness     = 0.8;
uniform float u_ao            = 1.0;
uniform float u_normal_scale  = 1.0;
uniform vec3  u_emissive_color = vec3(0.0);
uniform float u_emissive_strength = 0.0;

out vec4 fragColor;

const float PI = 3.14159265359;

// --- PBR helpers ---
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-5);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 1e-5);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// --- Shadow sampling (3x3 PCF) ---
float shadow_factor(vec3 N, vec3 L) {
    if (u_use_shadows == 0 || v_shadow_pos.w <= 0.0) return 1.0;

    vec3 proj = v_shadow_pos.xyz / v_shadow_pos.w;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z < 0.0 || proj.z > 1.0) {
        return 1.0;
    }

    float bias = max(0.0005, 0.002 * (1.0 - dot(N, L)));
    float shadow = 0.0;
    float texel_offset = 1.0 / 2048.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec4 sample_coord = v_shadow_pos;
            sample_coord.xy += vec2(x, y) * texel_offset * v_shadow_pos.w;
            sample_coord.z -= bias * v_shadow_pos.w;
            shadow += textureProj(p3d_LightShadowMap0, sample_coord);
        }
    }
    return shadow / 9.0;
}

void main() {
    // --- Material inputs ---
    vec3 albedo_tex = texture(p3d_Texture0, v_uv).rgb;
    if (length(albedo_tex) < 0.001) albedo_tex = vec3(1.0);
    vec3 albedo = albedo_tex * u_object_color.rgb;

    vec3 mr_sample = texture(p3d_Texture1, v_uv).rgb;
    float metallic = clamp(mr_sample.b + u_metallic, 0.0, 1.0);
    float roughness = clamp(mr_sample.g + u_roughness, 0.05, 1.0);
    float ao = clamp(texture(p3d_Texture3, v_uv).r * u_ao, 0.0, 1.0);

    vec3 emissive = texture(p3d_Texture4, v_uv).rgb * u_emissive_color * u_emissive_strength;

    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(p3d_CameraPosition - v_world_pos);
    vec3 L = (length(u_sun_direction) > 0.0001) ? normalize(u_sun_direction) : vec3(0.0, 1.0, 0.0);
    vec3 H = normalize(V + L);

    // Fresnel base reflectance
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // BRDF terms
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denom = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 1e-5;
    vec3 specular = numerator / denom;

    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    float NdotL = max(dot(N, L), 0.0);

    float shadow = shadow_factor(N, L);
    vec3 radiance = u_sun_color.rgb;

    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow;

    vec3 ambient = u_ambient_color.rgb * albedo * ao;

    vec3 color = ambient + Lo + emissive;

    // ACES-ish tone map (Narkowicz)
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    color = clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0, 1.0);

    // Gamma
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, u_object_color.a);
}
