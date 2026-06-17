#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include "backend/cuda_solver.h"

#define THREADS_PER_BLOCK 256

__device__ __forceinline__ glm::vec3 operator+(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ glm::vec3 operator-(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ glm::vec3 operator*(const glm::vec3& v, float s) {
    return glm::vec3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ glm::vec3 operator*(float s, const glm::vec3& v) {
    return v * s;
}

__device__ __forceinline__ glm::vec3 operator/(const glm::vec3& v, float s) {
    return glm::vec3(v.x / s, v.y / s, v.z / s);
}

__device__ __forceinline__ float dot(const glm::vec3& a, const glm::vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ glm::vec3 cross(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float length(const glm::vec3& v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ glm::vec3 normalize(const glm::vec3& v) {
    float len = length(v);
    if (len < 1e-6f) return glm::vec3(0.0f);
    return v / len;
}

__global__ void applyGravityKernel(
    glm::vec3* forces,
    float mass,
    float gravity,
    int numPoints,
    int pointsPerStrand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    int localIdx = idx % pointsPerStrand;
    if (localIdx == 0) {
        forces[idx] = glm::vec3(0.0f);
        return;
    }
    
    forces[idx] += glm::vec3(0.0f, -mass * gravity, 0.0f);
}

__global__ void applyDampingKernel(
    glm::vec3* forces,
    glm::vec3* velocities,
    float damping,
    int numPoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    forces[idx] -= velocities[idx] * damping;
}

__global__ void integrateKernel(
    glm::vec3* positions,
    glm::vec3* prevPositions,
    glm::vec3* velocities,
    glm::vec3* forces,
    float mass,
    float dt,
    int numPoints,
    int pointsPerStrand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    int localIdx = idx % pointsPerStrand;
    if (localIdx == 0) return;
    
    prevPositions[idx] = positions[idx];
    
    glm::vec3 acceleration = forces[idx] / mass;
    velocities[idx] += acceleration * dt;
    positions[idx] += velocities[idx] * dt;
}

__global__ void solveStretchConstraintsKernel(
    glm::vec3* positions,
    float* restLengths,
    float stiffness,
    int numCurves,
    int pointsPerStrand
) {
    int curveIdx = blockIdx.x;
    if (curveIdx >= numCurves) return;
    
    __shared__ glm::vec3 s_positions[64];
    
    int threadIdx = threadIdx.x;
    int strandOffset = curveIdx * pointsPerStrand;
    
    for (int i = threadIdx; i < pointsPerStrand && i < 64; i += blockDim.x) {
        s_positions[i] = positions[strandOffset + i];
    }
    __syncthreads();
    
    for (int iter = 0; iter < 5; iter++) {
        for (int i = threadIdx; i < pointsPerStrand - 1; i += blockDim.x) {
            if (i == 0) continue;
            
            glm::vec3 p0 = s_positions[i];
            glm::vec3 p1 = s_positions[i + 1];
            glm::vec3 diff = p1 - p0;
            float len = length(diff);
            
            if (len < 1e-6f) continue;
            
            float restLen = restLengths[curveIdx * (pointsPerStrand - 1) + i];
            float error = (len - restLen) / len;
            
            glm::vec3 correction = diff * error * 0.5f * stiffness;
            
            s_positions[i] += correction;
            s_positions[i + 1] -= correction;
        }
        __syncthreads();
    }
    
    for (int i = threadIdx; i < pointsPerStrand && i < 64; i += blockDim.x) {
        positions[strandOffset + i] = s_positions[i];
    }
}

__global__ void solveBendConstraintsKernel(
    glm::vec3* positions,
    float bendStiffness,
    int numCurves,
    int pointsPerStrand
) {
    int curveIdx = blockIdx.x;
    if (curveIdx >= numCurves) return;
    
    int strandOffset = curveIdx * pointsPerStrand;
    
    for (int i = threadIdx.x; i < pointsPerStrand - 2; i += blockDim.x) {
        if (i == 0) continue;
        
        glm::vec3 p0 = positions[strandOffset + i];
        glm::vec3 p1 = positions[strandOffset + i + 1];
        glm::vec3 p2 = positions[strandOffset + i + 2];
        
        glm::vec3 e1 = p1 - p0;
        glm::vec3 e2 = p2 - p1;
        
        float len1 = length(e1);
        float len2 = length(e2);
        
        if (len1 < 1e-6f || len2 < 1e-6f) continue;
        
        e1 = normalize(e1);
        e2 = normalize(e2);
        
        float cosTheta = dot(e1, e2);
        cosTheta = fminf(1.0f, fmaxf(-1.0f, cosTheta));
        
        glm::vec3 bendForceDir = cross(e1, e2);
        if (length(bendForceDir) < 1e-6f) continue;
        bendForceDir = normalize(bendForceDir);
        
        float bendAmount = acosf(cosTheta);
        
        glm::vec3 force = bendForceDir * bendStiffness * bendAmount;
        
        positions[strandOffset + i] += force * 0.25f;
        positions[strandOffset + i + 1] -= force * 0.5f;
        positions[strandOffset + i + 2] += force * 0.25f;
    }
}

__global__ void computeVortexWindKernel(
    glm::vec3* positions,
    glm::vec3* forces,
    glm::vec3* vortexPositions,
    float* vortexStrengths,
    float* vortexRadii,
    int numVortex,
    float windStrength,
    float time,
    int numPoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    glm::vec3 pos = positions[idx];
    glm::vec3 totalWind(0.0f);
    
    for (int v = 0; v < numVortex; v++) {
        glm::vec3 vortexPos = vortexPositions[v];
        float strength = vortexStrengths[v];
        float radius = vortexRadii[v];
        
        glm::vec3 toPos = pos - vortexPos;
        float dist = length(toPos);
        
        if (dist < radius * 3.0f && dist > 0.01f) {
            float falloff = 1.0f - fminf(1.0f, dist / (radius * 3.0f));
            
            glm::vec3 tangent = cross(glm::vec3(0.0f, 1.0f, 0.0f), normalize(toPos));
            float angularSpeed = strength * falloff / fmaxf(dist, radius * 0.5f);
            
            totalWind += tangent * angularSpeed * windStrength;
            
            float verticalPull = strength * falloff * 0.3f;
            totalWind += glm::vec3(0.0f, verticalPull, 0.0f);
        }
    }
    
    forces[idx] += totalWind * 0.1f;
}

__global__ void applyGlobalWindKernel(
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
    
    glm::vec3 pos = positions[idx];
    
    float noiseX = sinf(pos.x * 0.5f + time * 2.0f) * cosf(pos.z * 0.3f + time * 1.5f);
    float noiseY = sinf(pos.y * 0.4f + time * 1.8f) * cosf(pos.x * 0.6f + time * 2.2f);
    float noiseZ = cosf(pos.z * 0.5f + time * 1.7f) * sinf(pos.y * 0.3f + time * 2.0f);
    
    glm::vec3 turbulenceVec(noiseX, noiseY, noiseZ);
    turbulenceVec *= turbulence * windStrength;
    
    glm::vec3 windForce = windDirection * windStrength * tipFactor;
    windForce += turbulenceVec * tipFactor;
    
    forces[idx] += windForce * 0.5f;
}

__global__ void capsuleCollisionKernel(
    glm::vec3* positions,
    glm::vec3* velocities,
    CapsuleCollider* colliders,
    int numColliders,
    float friction,
    float collisionRadius,
    int numPoints,
    int pointsPerStrand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    int localIdx = idx % pointsPerStrand;
    if (localIdx == 0) return;
    
    glm::vec3 pos = positions[idx];
    glm::vec3 vel = velocities[idx];
    
    for (int c = 0; c < numColliders; c++) {
        CapsuleCollider col = colliders[c];
        
        glm::vec3 ab = col.end - col.start;
        glm::vec3 ap = pos - col.start;
        
        float t = dot(ap, ab) / dot(ab, ab);
        t = fmaxf(0.0f, fminf(1.0f, t));
        
        glm::vec3 closestPoint = col.start + ab * t;
        glm::vec3 diff = pos - closestPoint;
        float dist = length(diff);
        
        float totalRadius = col.radius + collisionRadius;
        
        if (dist < totalRadius && dist > 1e-6f) {
            glm::vec3 normal = diff / dist;
            float penetration = totalRadius - dist;
            
            pos += normal * penetration * 1.1f;
            
            float velNormal = dot(vel, normal);
            if (velNormal < 0.0f) {
                vel -= normal * velNormal * (1.0f + friction);
            }
        }
    }
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

__global__ void selfCollisionKernel(
    glm::vec3* positions,
    glm::vec3* velocities,
    float collisionRadius,
    int numCurves,
    int pointsPerStrand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCurves) return;
    
    int strandOffset = idx * pointsPerStrand;
    
    for (int p = 1; p < pointsPerStrand; p += 4) {
        int pointIdx = strandOffset + p;
        glm::vec3 pos = positions[pointIdx];
        glm::vec3 vel = velocities[pointIdx];
        
        for (int other = 0; other < numCurves; other++) {
            if (other == idx) continue;
            
            int otherOffset = other * pointsPerStrand;
            
            for (int op = 1; op < pointsPerStrand; op += 3) {
                int otherPointIdx = otherOffset + op;
                glm::vec3 otherPos = positions[otherPointIdx];
                
                glm::vec3 diff = pos - otherPos;
                float dist = length(diff);
                
                float minDist = collisionRadius * 2.0f;
                
                if (dist < minDist && dist > 1e-6f) {
                    glm::vec3 normal = diff / dist;
                    float penetration = minDist - dist;
                    
                    pos += normal * penetration * 0.5f;
                    velocities[otherPointIdx] -= normal * penetration * 0.05f;
                }
            }
        }
        
        positions[pointIdx] = pos;
        velocities[pointIdx] = vel;
    }
}

void CudaHairSolver::applyForces(float dt)
{
    int blocks = (m_totalPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    cudaMemset(d_forces, 0, m_totalPoints * sizeof(glm::vec3));
    
    applyGravityKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_forces,
        m_params.mass,
        m_params.gravity,
        m_totalPoints,
        m_pointsPerStrand
    );
    
    computeWindForces();
    
    applyDampingKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_forces,
        d_velocities,
        m_params.damping,
        m_totalPoints
    );
}

void CudaHairSolver::integrate(float dt)
{
    int blocks = (m_totalPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    integrateKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_positions,
        d_prevPositions,
        d_velocities,
        d_forces,
        m_params.mass,
        dt,
        m_totalPoints,
        m_pointsPerStrand
    );
}

void CudaHairSolver::solveConstraints()
{
    solveStretchConstraintsKernel<<<m_numGuideCurves, 32>>>(
        d_positions,
        d_restLengths,
        m_params.stiffness,
        m_numGuideCurves,
        m_pointsPerStrand
    );
    
    solveBendConstraintsKernel<<<m_numGuideCurves, 32>>>(
        d_positions,
        m_params.bendStiffness,
        m_numGuideCurves,
        m_pointsPerStrand
    );
}

void CudaHairSolver::computeWindForces()
{
    int blocks = (m_totalPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    applyGlobalWindKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_positions,
        d_forces,
        m_params.windDirection,
        m_params.windStrength,
        m_params.turbulence,
        m_elapsedTime,
        m_totalPoints,
        m_pointsPerStrand
    );
    
    if (m_params.numVortexSources > 0) {
        glm::vec3* d_vortexPos;
        float* d_vortexStrengths;
        float* d_vortexRadii;
        
        cudaMalloc(&d_vortexPos, 5 * sizeof(glm::vec3));
        cudaMalloc(&d_vortexStrengths, 5 * sizeof(float));
        cudaMalloc(&d_vortexRadii, 5 * sizeof(float));
        
        cudaMemcpy(d_vortexPos, m_params.vortexPositions, 5 * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vortexStrengths, m_params.vortexStrengths, 5 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vortexRadii, m_params.vortexRadii, 5 * sizeof(float), cudaMemcpyHostToDevice);
        
        computeVortexWindKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_positions,
            d_forces,
            d_vortexPos,
            d_vortexStrengths,
            d_vortexRadii,
            m_params.numVortexSources,
            m_params.windStrength,
            m_elapsedTime,
            m_totalPoints
        );
        
        cudaFree(d_vortexPos);
        cudaFree(d_vortexStrengths);
        cudaFree(d_vortexRadii);
    }
}

void CudaHairSolver::handleCollisions()
{
    if (m_params.numColliders > 0) {
        int blocks = (m_totalPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        capsuleCollisionKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_positions,
            d_velocities,
            d_colliders,
            m_params.numColliders,
            m_params.friction,
            m_params.collisionRadius,
            m_totalPoints,
            m_pointsPerStrand
        );
    }
    
    selfCollisionKernel<<<(m_numGuideCurves + 63) / 64, 64>>>(
        d_positions,
        d_velocities,
        m_params.collisionRadius,
        m_numGuideCurves,
        m_pointsPerStrand
    );
}
