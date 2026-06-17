#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#define WIND_THREADS_PER_BLOCK 256

__device__ __forceinline__ glm::vec3 wind_vec3_add(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ glm::vec3 wind_vec3_sub(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ glm::vec3 wind_vec3_mul(const glm::vec3& v, float s) {
    return glm::vec3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ float wind_vec3_dot(const glm::vec3& a, const glm::vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ glm::vec3 wind_vec3_cross(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float wind_vec3_length(const glm::vec3& v) {
    return sqrtf(wind_vec3_dot(v, v));
}

__device__ __forceinline__ glm::vec3 wind_vec3_normalize(const glm::vec3& v) {
    float len = wind_vec3_length(v);
    if (len < 1e-6f) return glm::vec3(0.0f);
    return wind_vec3_mul(v, 1.0f / len);
}

__device__ __forceinline__ float wind_noise(float x, float y, float z, float t) {
    float n = sinf(x * 12.9898f + y * 78.233f + z * 45.164f + t * 37.563f) * 43758.5453f;
    return n - floorf(n);
}

__global__ void globalWindKernel(
    glm::vec3* positions,
    glm::vec3* forces,
    glm::vec3 windDirection,
    float windStrength,
    float turbulence,
    float time,
    int numPoints,
    int pointsPerStrand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    int localIdx = idx % pointsPerStrand;
    float tipFactor = float(localIdx) / float(pointsPerStrand - 1);
    tipFactor = tipFactor * tipFactor;
    
    glm::vec3 pos = positions[idx];
    
    float noiseScale = 0.5f;
    float n1 = wind_noise(pos.x * noiseScale, pos.y * noiseScale, pos.z * noiseScale, time);
    float n2 = wind_noise(pos.y * noiseScale + 100.0f, pos.z * noiseScale + 200.0f, pos.x * noiseScale + 300.0f, time + 5.0f);
    float n3 = wind_noise(pos.z * noiseScale + 50.0f, pos.x * noiseScale + 150.0f, pos.y * noiseScale + 250.0f, time + 10.0f);
    
    glm::vec3 turbulenceVec = glm::vec3(
        (n1 - 0.5f) * 2.0f,
        (n2 - 0.5f) * 2.0f,
        (n3 - 0.5f) * 2.0f
    );
    turbulenceVec = wind_vec3_mul(turbulenceVec, turbulence * windStrength);
    
    glm::vec3 baseWind = wind_vec3_mul(windDirection, windStrength);
    
    float gustFactor = 0.5f + 0.5f * sinf(time * 2.0f + pos.x * 0.3f);
    baseWind = wind_vec3_mul(baseWind, 0.8f + 0.4f * gustFactor);
    
    glm::vec3 totalWind = wind_vec3_add(baseWind, turbulenceVec);
    totalWind = wind_vec3_mul(totalWind, tipFactor);
    
    forces[idx] = wind_vec3_add(forces[idx], wind_vec3_mul(totalWind, 0.5f));
}

__global__ void vortexWindKernel(
    glm::vec3* positions,
    glm::vec3* forces,
    glm::vec3* vortexPositions,
    float* vortexStrengths,
    float* vortexRadii,
    int numVortex,
    float time,
    int numPoints,
    int pointsPerStrand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    int localIdx = idx % pointsPerStrand;
    float tipFactor = float(localIdx) / float(pointsPerStrand - 1);
    
    glm::vec3 pos = positions[idx];
    glm::vec3 totalForce(0.0f);
    
    for (int v = 0; v < numVortex; v++) {
        glm::vec3 vortexPos = vortexPositions[v];
        float strength = vortexStrengths[v];
        float radius = vortexRadii[v];
        
        float phaseOffset = float(v) * 1.3f;
        glm::vec3 animatedPos = vortexPos;
        animatedPos.x += sinf(time * 0.5f + phaseOffset) * radius * 0.3f;
        animatedPos.z += cosf(time * 0.7f + phaseOffset * 1.5f) * radius * 0.3f;
        
        glm::vec3 toPos = wind_vec3_sub(pos, animatedPos);
        float dist = wind_vec3_length(toPos);
        
        float influenceRadius = radius * 3.0f;
        
        if (dist < influenceRadius && dist > radius * 0.1f) {
            float normalizedDist = dist / influenceRadius;
            float falloff = 1.0f - normalizedDist * normalizedDist;
            falloff = fmaxf(0.0f, falloff);
            
            glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
            glm::vec3 radial = wind_vec3_normalize(toPos);
            glm::vec3 tangent = wind_vec3_cross(up, radial);
            
            float angularSpeed = strength * falloff / fmaxf(dist, radius * 0.5f);
            glm::vec3 rotationalForce = wind_vec3_mul(tangent, angularSpeed);
            
            float liftStrength = strength * falloff * 0.4f;
            glm::vec3 liftForce = wind_vec3_mul(up, liftStrength);
            
            float inwardPull = strength * falloff * 0.2f * (1.0f - normalizedDist);
            glm::vec3 inwardForce = wind_vec3_mul(radial, -inwardPull);
            
            totalForce = wind_vec3_add(totalForce, rotationalForce);
            totalForce = wind_vec3_add(totalForce, liftForce);
            totalForce = wind_vec3_add(totalForce, inwardForce);
        }
    }
    
    totalForce = wind_vec3_mul(totalForce, tipFactor * 0.3f);
    forces[idx] = wind_vec3_add(forces[idx], totalForce);
}

class WindFieldGenerator {
public:
    static void applyGlobalWind(
        glm::vec3* d_positions,
        glm::vec3* d_forces,
        const glm::vec3& windDirection,
        float windStrength,
        float turbulence,
        float time,
        int numPoints,
        int pointsPerStrand
    ) {
        int blocks = (numPoints + WIND_THREADS_PER_BLOCK - 1) / WIND_THREADS_PER_BLOCK;
        
        globalWindKernel<<<blocks, WIND_THREADS_PER_BLOCK>>>(
            d_positions,
            d_forces,
            windDirection,
            windStrength,
            turbulence,
            time,
            numPoints,
            pointsPerStrand
        );
    }
    
    static void applyVortices(
        glm::vec3* d_positions,
        glm::vec3* d_forces,
        glm::vec3* d_vortexPositions,
        float* d_vortexStrengths,
        float* d_vortexRadii,
        int numVortex,
        float time,
        int numPoints,
        int pointsPerStrand
    ) {
        if (numVortex <= 0) return;
        
        int blocks = (numPoints + WIND_THREADS_PER_BLOCK - 1) / WIND_THREADS_PER_BLOCK;
        
        vortexWindKernel<<<blocks, WIND_THREADS_PER_BLOCK>>>(
            d_positions,
            d_forces,
            d_vortexPositions,
            d_vortexStrengths,
            d_vortexRadii,
            numVortex,
            time,
            numPoints,
            pointsPerStrand
        );
    }
};
