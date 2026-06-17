#version 430 core

in vec3 vColor;
in vec3 vNormal;
in vec3 vViewDir;
in float vSegmentId;
in vec3 vWorldPos;

uniform vec3 uLightDir;
uniform vec3 uHairColor;
uniform float uTime;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);
    vec3 viewDir = normalize(vViewDir);
    
    float ndotl = max(dot(normal, lightDir), 0.0);
    
    vec3 halfDir = normalize(lightDir + viewDir);
    float specular = pow(max(dot(normal, halfDir), 0.0), 32.0);
    
    float tipFactor = vSegmentId / 20.0;
    vec3 baseColor = mix(vColor * 0.8, vColor, tipFactor);
    
    float noise = fract(sin(dot(vWorldPos.xz, vec2(12.9898, 78.233)) + uTime) * 43758.5453);
    baseColor *= 0.9 + noise * 0.2;
    
    vec3 ambient = baseColor * 0.3;
    vec3 diffuse = baseColor * ndotl * 0.7;
    vec3 spec = vec3(0.3) * specular;
    
    vec3 finalColor = ambient + diffuse + spec;
    
    finalColor = pow(finalColor, vec3(1.0 / 2.2));
    
    fragColor = vec4(finalColor, 1.0);
}
