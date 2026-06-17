#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "common/hair_data.h"

struct CudaHairParams {
    int numGuideCurves;
    int pointsPerStrand;
    float stiffness;
    float bendStiffness;
    float twistStiffness;
    float damping;
    float gravity;
    float mass;
    float timeStep;
    int substeps;
    float windStrength;
    glm::vec3 windDirection;
    float turbulence;
    int numVortexSources;
    glm::vec3 vortexPositions[5];
    float vortexStrengths[5];
    float vortexRadii[5];
    float collisionRadius;
    float friction;
    int numColliders;
};

class CudaHairSolver {
public:
    CudaHairSolver();
    ~CudaHairSolver();
    
    bool initialize(const HairParams& params);
    void shutdown();
    
    void step(float deltaTime);
    void reset();
    
    void uploadHairData(const HairData& hairData);
    void downloadHairData(HairData& hairData);
    
    void updateParams(const HairParams& params);
    
    void setColliders(const std::vector<CapsuleCollider>& colliders);
    
    float getLastFrameTime() const { return m_lastFrameTime; }
    
private:
    void allocateBuffers();
    void freeBuffers();
    void computeVelocities();
    void applyForces(float dt);
    void integrate(float dt);
    void solveConstraints();
    void computeWindForces();
    void handleCollisions();
    
    bool m_initialized;
    
    CudaHairParams m_params;
    
    glm::vec3* d_positions;
    glm::vec3* d_prevPositions;
    glm::vec3* d_velocities;
    glm::vec3* d_forces;
    float* d_restLengths;
    glm::vec3* d_guidePositions;
    
    CapsuleCollider* d_colliders;
    
    int m_totalPoints;
    int m_numGuideCurves;
    int m_pointsPerStrand;
    
    float m_lastFrameTime;
    float m_elapsedTime;
};
