#version 150

uniform vec4 u_outline_color;

// Inputs from Vertex Shader
in vec3 v_view_normal;
in vec3 v_view_pos;

out vec4 fragColor;

void main() {
    fragColor = u_outline_color;
}