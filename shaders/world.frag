#version 150
/*
 * WORLD SHADER (PBR)
 *
 * - Implements a simplified Cook-Torrance PBR lighting model.
 * - Supports Shadows with PCF.
 */

// Inputs
in vec3 fNormal;
in vec3 fPosition;
in vec4 v_shadow_pos;

// Uniforms
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

uniform vec3 p3d_CameraPosition;

uniform vec3 u_sun_direction;
uniform vec4 u_sun_color;
uniform vec4 u_ambient_color;
uniform vec4 u_object_color;

out vec4 fragColor;

const float PI = 3.14159265359;

// --- PBR Functions ---

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.0000001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / max(denom, 0.0000001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// --- Shadow Calculation ---
float calculate_shadow_factor(vec3 N, vec3 L) {
    if (v_shadow_pos.w <= 0.0) return 1.0;

    vec3 proj_coords = v_shadow_pos.xyz / v_shadow_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;

    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        proj_coords.z > 1.0) {
        return 1.0;
    }
    
    // Bias to prevent acne
    float bias = max(0.005 * (1.0 - dot(N, L)), 0.001);
    
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(p3d_LightSource[0].shadowMap, 0);

    // 2x2 PCF
    for(int x = -1; x <= 1; x += 2) {
        for(int y = -1; y <= 1; y += 2) {
            shadow += texture(p3d_LightSource[0].shadowMap, vec3(proj_coords.xy + vec2(x, y) * texel_size, proj_coords.z - bias));
        }
    }
    return shadow / 4.0;
}

void main() {
    // Material Properties (Hardcoded for now, could be uniforms)
    vec3 albedo = u_object_color.rgb;
    // Fallback if albedo is black
    if (length(albedo) < 0.01) albedo = vec3(0.8);

    float roughness = 0.8; // Rough ground
    float metallic = 0.0;  // Non-metallic
    float ao = 1.0;

    vec3 N = normalize(fNormal);
    vec3 V = normalize(p3d_CameraPosition - fPosition);
    vec3 L = (length(u_sun_direction) > 0.0001) ? normalize(u_sun_direction) : vec3(0,0,1);
    vec3 H = normalize(V + L);

    // F0 for dielectrics is 0.04
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // --- Direct Lighting (Sun) ---
    vec3 radiance = u_sun_color.rgb * 3.0; // Boost sun intensity

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3 F    = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;

    // kS is equal to Fresnel
    vec3 kS = F;
    // kD is 1.0 - kS
    vec3 kD = vec3(1.0) - kS;
    // Multiply kD by (1.0 - metallic) because metals have no diffuse
    kD *= 1.0 - metallic;

    float NdotL = max(dot(N, L), 0.0);

    // Shadow
    float shadow = calculate_shadow_factor(N, L);
    // Darker shadows (0.1 multiplier)
    float shadow_mult = mix(0.1, 1.0, shadow);

    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow_mult;
    
    // --- Ambient Lighting ---
    // Increased ambient floor significantly to prevent black ground
    vec3 ambient = max(u_ambient_color.rgb, vec3(0.2)) * albedo * ao;

    vec3 color = ambient + Lo;

    // HDR Tonemapping (Reinhard)
    color = color / (color + vec3(1.0));
    
    // --- Contrast Boost ---
    // Simple contrast adjustment: push values away from middle gray (0.5)
    float contrast = 1.2;
    color = 0.5 + contrast * (color - 0.5);
    color = clamp(color, 0.0, 1.0);

    // Gamma Correction
    color = pow(color, vec3(1.0/2.2));

    fragColor = vec4(color, u_object_color.a);
}