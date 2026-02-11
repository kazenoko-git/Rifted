#version 150

uniform sampler2D p3d_Texture0;
uniform sampler2DShadow p3d_LightShadowMap0;

uniform vec3 u_sun_direction;
uniform vec4 u_sun_color;
uniform vec4 u_ambient_color;
uniform vec4 u_object_color; // Flat color if no texture
uniform int u_use_shadows;

// Toon Settings
const float TOON_THRESHOLD = 0.5;
const float TOON_FEATHER = 0.05;

in vec3 v_normal;
in vec2 v_uv;
in vec4 v_shadow_pos;

out vec4 fragColor;

float calculate_shadow(vec4 shadow_pos) {
    if (u_use_shadows == 0) return 1.0;

    // Same PCF logic as world shader
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(p3d_LightShadowMap0, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            shadow += textureProj(p3d_LightShadowMap0, shadow_pos + vec4(vec2(x, y) * texel_size * shadow_pos.w, 0.0, 0.0));
        }
    }
    return shadow / 9.0;
}

void main() {
    vec4 tex_color = texture(p3d_Texture0, v_uv);
    vec4 base_color = tex_color * u_object_color;
    
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_sun_direction);
    
    // --- CHARACTER TOON SHADING ---
    
    // 1. Raw Lambert
    float NdotL = max(0.0, dot(N, L));
    
    // 2. Shadow Sample
    float shadow = calculate_shadow(v_shadow_pos);
    
    // 3. Combine Light & Shadow BEFORE Quantization
    // This is the key to anime shadows. The shadow map forces the value down,
    // causing the toon ramp to pick the dark color.
    float effective_light = NdotL * shadow;
    
    // 4. Quantize (Toon Ramp)
    // Smoothstep creates a small anti-aliased edge between light and dark
    float light_intensity = smoothstep(TOON_THRESHOLD - TOON_FEATHER, TOON_THRESHOLD + TOON_FEATHER, effective_light);
    
    // 5. Rim Light (Optional, adds depth)
    // float fresnel = 1.0 - max(0.0, dot(N, V)); ... (omitted for brevity/perf)
    
    // 6. Final Composition
    // Light area gets Sun Color, Dark area gets nothing (just Ambient)
    vec3 direct_light = u_sun_color.rgb * light_intensity;
    vec3 lighting = u_ambient_color.rgb + direct_light;
    
    fragColor = vec4(base_color.rgb * lighting, base_color.a);
}