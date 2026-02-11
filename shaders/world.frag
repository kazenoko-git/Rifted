#version 150
/*
 * WORLD SHADER (Stylized, Non-Toon)
 *
 * - Uses a Half-Lambert lighting model to prevent completely black shadows.
 * - Integrates with the global shadow map.
 * - Implements 2x2 PCF for soft shadow edges.
 */

// Inputs from Vertex Shader
in vec3 v_world_normal;
in vec4 v_world_pos;
in vec4 v_shadow_pos; // Position of the fragment in light space

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

uniform vec3 u_sun_direction;
uniform vec4 u_sun_color;
uniform vec4 u_ambient_color;

// Per-Mesh Uniforms
uniform vec4 u_object_color;

out vec4 fragColor;

// --- Shadow Calculation (PCF) ---
float calculate_shadow_factor() {
    if (v_shadow_pos.w <= 0.0) {
        return 1.0;
    }

    vec3 shadow_coord = v_shadow_pos.xyz / v_shadow_pos.w;
    shadow_coord = shadow_coord * 0.5 + 0.5;
    
    if (shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z > 1.0) {
        return 1.0;
    }
    
    float shadow = 0.0;
    float bias = 0.002;
    vec2 texel_size = 1.0 / textureSize(p3d_LightSource[0].shadowMap, 0);

    for (int x = -1; x <= 1; x += 2) {
        for (int y = -1; y <= 1; y += 2) {
            shadow += texture(
                p3d_LightSource[0].shadowMap,
                vec3(shadow_coord.xy + vec2(x, y) * texel_size, shadow_coord.z - bias)
            );
        }
    }
    
    return shadow / 4.0;
}

void main() {
    // Prevent optimization of v_world_pos
    if (v_world_pos.w < 0.0) discard;

    // --- 1. Get Material & Lighting Properties ---
    vec3 N = normalize(v_world_normal);
    vec3 L = (length(u_sun_direction) > 0.0001) ? normalize(u_sun_direction) : normalize(vec3(0.3, 0.4, 0.85));
    
    vec3 base_color = u_object_color.rgb;
    // For an ambient light, its contribution is stored in its 'color' property
    vec3 ambient_light = max(u_ambient_color.rgb, vec3(0.16));
    vec3 directional_light_color = u_sun_color.rgb;

    // --- 2. Calculate Lighting (Half-Lambert) ---
    float NdotL = dot(N, L);
    float half_lambert = (NdotL * 0.5) + 0.5;
    float diffuse_intensity = half_lambert * half_lambert;

    // --- 3. Calculate Shadows ---
    float shadow_factor = calculate_shadow_factor();
    // Keep stylized terrain readable even in deep shadow.
    float softened_shadow = mix(0.65, 1.0, shadow_factor);

    // --- 4. Combine Lighting and Shadows ---
    vec3 final_lighting = ambient_light + (diffuse_intensity * directional_light_color * softened_shadow);
    final_lighting = max(final_lighting, vec3(0.12));
    
    vec3 final_color = base_color * final_lighting;

    // FORCE USAGE: Add a tiny fraction of shadow pos to color to prevent optimization
    final_color += v_shadow_pos.rgb * 0.000001;
    
    fragColor = vec4(final_color, u_object_color.a);
}
