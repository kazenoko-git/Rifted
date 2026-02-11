#version 150

uniform sampler2D p3d_Texture0;
uniform sampler2DShadow p3d_LightShadowMap0; // Panda auto-binds this if configured

uniform vec3 u_sun_direction;
uniform vec4 u_sun_color;
uniform vec4 u_ambient_color;
uniform int u_use_shadows;

in vec3 v_normal;
in vec2 v_uv;
in vec4 v_shadow_pos;

out vec4 fragColor;

float calculate_shadow(vec4 shadow_pos) {
    if (u_use_shadows == 0) return 1.0;

    vec3 proj_coords = shadow_pos.xyz / shadow_pos.w;
    // Panda3D shadow maps are 0-1, but we need to handle the bias
    // The bias matrix is usually applied in vertex shader, assuming u_light_mvp handles it.
    
    // Simple PCF (3x3)
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(p3d_LightShadowMap0, 0);
    
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            // textureProj performs the depth comparison automatically for sampler2DShadow
            shadow += textureProj(p3d_LightShadowMap0, shadow_pos + vec4(vec2(x, y) * texel_size * shadow_pos.w, 0.0, 0.0));
        }
    }
    return shadow / 9.0;
}

void main() {
    vec4 albedo = texture(p3d_Texture0, v_uv);
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_sun_direction); // Direction TO light
    
    // --- STYLIZED WORLD SHADING (Half-Lambert) ---
    // Standard Lambert is max(dot(N, L), 0.0)
    // Half-Lambert wraps the light to prevent harsh black shadows on terrain
    float NdotL = dot(N, L);
    float half_lambert = NdotL * 0.5 + 0.5;
    
    // Shadow Calculation
    float shadow = calculate_shadow(v_shadow_pos);
    
    // Combine
    // Ambient is constant.
    // Direct light is modulated by Half-Lambert AND Shadow.
    vec3 direct_light = u_sun_color.rgb * half_lambert * shadow;
    vec3 lighting = u_ambient_color.rgb + direct_light;
    
    fragColor = vec4(albedo.rgb * lighting, albedo.a);
}