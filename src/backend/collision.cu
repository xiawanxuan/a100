#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include "common/hair_data.h"

#define COLLISION_THREADS_PER_BLOCK 256

__device__ __forceinline__ glm::vec3 col_vec3_add(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ glm::vec3 col_vec3_sub(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ glm::vec3 col_vec3_mul(const glm::vec3& v, float s) {
    return glm::vec3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ float col_vec3_dot(const glm::vec3& a, const glm::vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float col_vec3_length(const glm::vec3& v) {
    return sqrtf(col_vec3_dot(v, v));
}

__device__ __forceinline__ glm::vec3 col_vec3_normalize(const glm::vec3& v) {
    float len = col_vec3_length(v);
    if (len < 1e-6f) return glm::vec3(0.0f);
    return col_vec3_mul(v, 1.0f / len);
}

__device__ __forceinline__ float pointToSegmentDistance(
    const glm::vec3& point,
    const glm::vec3& segStart,
    const glm::vec3& segEnd,
    glm::vec3& closestPoint
) {
    glm::vec3 ab = col_vec3_sub(segEnd, segStart);
    glm::vec3 ap = col_vec3_sub(point, segStart);
    
    float abSq = col_vec3_dot(ab, ab);
    if (abSq < 1e-10f) {
        closestPoint = segStart;
        return col_vec3_length(ap);
    }
    
    float t = col_vec3_dot(ap, ab) / abSq;
    t = fmaxf(0.0f, fminf(1.0f, t));
    
    closestPoint = col_vec3_add(segStart, col_vec3_mul(ab, t));
    return col_vec3_length(col_vec3_sub(point, closestPoint));
}

__global__ void capsuleCollisionKernel(
    glm::vec3* positions,
    glm::vec3* velocities,
    CapsuleCollider* colliders,
    int numColliders,
    float collisionRadius,
    float friction,
    float restitution,
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
        
        glm::vec3 closestPoint;
        float dist = pointToSegmentDistance(pos, col.start, col.end, closestPoint);
        
        float totalRadius = col.radius + collisionRadius;
        
        if (dist < totalRadius && dist > 1e-6f) {
            glm::vec3 normal = col_vec3_normalize(col_vec3_sub(pos, closestPoint));
            
            float penetration = totalRadius - dist;
            pos = col_vec3_add(pos, col_vec3_mul(normal, penetration * 1.05f));
            
            float velNormal = col_vec3_dot(vel, normal);
            if (velNormal < 0.0f) {
                vel = col_vec3_sub(vel, col_vec3_mul(normal, velNormal * (1.0f + restitution)));
                
                glm::vec3 velTangent = col_vec3_sub(vel, col_vec3_mul(normal, col_vec3_dot(vel, normal)));
                float tangentSpeed = col_vec3_length(velTangent);
                if (tangentSpeed > 1e-6f) {
                    float frictionForce = fabsf(velNormal) * friction;
                    frictionForce = fminf(frictionForce, tangentSpeed);
                    vel = col_vec3_sub(vel, col_vec3_mul(col_vec3_normalize(velTangent), frictionForce));
                }
            }
        }
    }
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

__global__ void spatialHashSelfCollisionKernel(
    glm::vec3* positions,
    glm::vec3* velocities,
    float collisionRadius,
    int numCurves,
    int pointsPerStrand,
    float cellSize
) {
    __shared__ int sh_indices[512];
    __shared__ glm::vec3 sh_positions[512];
    
    int threadIdx = threadIdx.x;
    int blockIdx = blockIdx.x;
    
    int strandIdx = blockIdx;
    if (strandIdx >= numCurves) return;
    
    int strandOffset = strandIdx * pointsPerStrand;
    
    for (int pointIdx = 1 + threadIdx; pointIdx < pointsPerStrand; pointIdx += blockDim.x) {
        int globalIdx = strandOffset + pointIdx;
        glm::vec3 pos = positions[globalIdx];
        glm::vec3 vel = velocities[globalIdx];
        
        for (int otherStrand = 0; otherStrand < numCurves; otherStrand++) {
            if (otherStrand == strandIdx) continue;
            
            int otherOffset = otherStrand * pointsPerStrand;
            
            for (int otherPoint = 1; otherPoint < pointsPerStrand; otherPoint += 2) {
                int otherGlobalIdx = otherOffset + otherPoint;
                glm::vec3 otherPos = positions[otherGlobalIdx];
                
                glm::vec3 diff = col_vec3_sub(pos, otherPos);
                float dist = col_vec3_length(diff);
                
                float minDist = collisionRadius * 2.5f;
                
                if (dist < minDist && dist > 1e-6f) {
                    glm::vec3 normal = col_vec3_normalize(diff);
                    float penetration = minDist - dist;
                    
                    pos = col_vec3_add(pos, col_vec3_mul(normal, penetration * 0.5f));
                    
                    float relVel = col_vec3_dot(col_vec3_sub(vel, velocities[otherGlobalIdx]), normal);
                    if (relVel < 0.0f) {
                        vel = col_vec3_sub(vel, col_vec3_mul(normal, relVel * 0.3f));
                    }
                }
            }
        }
        
        positions[globalIdx] = pos;
        velocities[globalIdx] = vel;
    }
}

__global__ void groundCollisionKernel(
    glm::vec3* positions,
    glm::vec3* velocities,
    float groundY,
    float friction,
    int numPoints,
    int pointsPerStrand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    int localIdx = idx % pointsPerStrand;
    if (localIdx == 0) return;
    
    glm::vec3 pos = positions[idx];
    glm::vec3 vel = velocities[idx];
    
    if (pos.y < groundY) {
        pos.y = groundY;
        
        if (vel.y < 0.0f) {
            vel.y = -vel.y * 0.2f;
            
            vel.x *= (1.0f - friction);
            vel.z *= (1.0f - friction);
        }
    }
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

class CollisionDetector {
public:
    static void handleCapsuleCollisions(
        glm::vec3* d_positions,
        glm::vec3* d_velocities,
        CapsuleCollider* d_colliders,
        int numColliders,
        float collisionRadius,
        float friction,
        float restitution,
        int numPoints,
        int pointsPerStrand
    ) {
        if (numColliders <= 0) return;
        
        int blocks = (numPoints + COLLISION_THREADS_PER_BLOCK - 1) / COLLISION_THREADS_PER_BLOCK;
        
        capsuleCollisionKernel<<<blocks, COLLISION_THREADS_PER_BLOCK>>>(
            d_positions,
            d_velocities,
            d_colliders,
            numColliders,
            collisionRadius,
            friction,
            restitution,
            numPoints,
            pointsPerStrand
        );
    }
    
    static void handleSelfCollisions(
        glm::vec3* d_positions,
        glm::vec3* d_velocities,
        float collisionRadius,
        int numCurves,
        int pointsPerStrand
    ) {
        float cellSize = collisionRadius * 4.0f;
        
        spatialHashSelfCollisionKernel<<<numCurves, 32>>>(
            d_positions,
            d_velocities,
            collisionRadius,
            numCurves,
            pointsPerStrand,
            cellSize
        );
    }
    
    static void handleGroundCollision(
        glm::vec3* d_positions,
        glm::vec3* d_velocities,
        float groundY,
        float friction,
        int numPoints,
        int pointsPerStrand
    ) {
        int blocks = (numPoints + COLLISION_THREADS_PER_BLOCK - 1) / COLLISION_THREADS_PER_BLOCK;
        
        groundCollisionKernel<<<blocks, COLLISION_THREADS_PER_BLOCK>>>(
            d_positions,
            d_velocities,
            groundY,
            friction,
            numPoints,
            pointsPerStrand
        );
    }
};
