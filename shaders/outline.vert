#version 150

// Uniforms
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
uniform float u_outline_width;

// Inputs
in vec4 p3d_Vertex;
in vec3 p3d_Normal;

void main() {
    // Extrude in view space so outline thickness remains stable across camera distance.
    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    vec3 view_n = normalize(p3d_NormalMatrix * p3d_Normal);

    // Push vertex along normal
    // We use a constant width in view space, but we can scale it by depth if we want constant screen size
    // For "constant world size" outline, just add to view_pos.
    // For "constant screen size", we multiply by -view_pos.z

    // Let's stick to constant world size for now as it's more predictable for 3D objects
    view_pos.xyz += view_n * u_outline_width;

    gl_Position = p3d_ProjectionMatrix * view_pos;
}