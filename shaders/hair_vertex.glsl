#version 430 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aStrandId;
layout(location = 3) in float aSegmentId;

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform vec3 uLightDir;
uniform float uHairThickness;

out vec3 vColor;
out vec3 vNormal;
out vec3 vViewDir;
out float vSegmentId;
out vec3 vWorldPos;

void main() {
    vec4 viewPos = uViewMatrix * vec4(aPosition, 1.0);
    gl_Position = uProjectionMatrix * viewPos;
    
    vColor = aColor;
    vSegmentId = aSegmentId;
    vWorldPos = aPosition;
    
    vec3 normal = normalize(vec3(0.0, 1.0, 0.0));
    vNormal = normal;
    vViewDir = normalize(-viewPos.xyz);
}
