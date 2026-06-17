#include "backend/cuda_solver.h"
#include <iostream>
#include <cuda_runtime.h>

CudaHairSolver::CudaHairSolver()
    : m_initialized(false)
    , d_positions(nullptr)
    , d_prevPositions(nullptr)
    , d_velocities(nullptr)
    , d_forces(nullptr)
    , d_restLengths(nullptr)
    , d_guidePositions(nullptr)
    , d_colliders(nullptr)
    , m_totalPoints(0)
    , m_numGuideCurves(0)
    , m_pointsPerStrand(0)
    , m_lastFrameTime(0.0f)
    , m_elapsedTime(0.0f)
{
}

CudaHairSolver::~CudaHairSolver()
{
    shutdown();
}

bool CudaHairSolver::initialize(const HairParams& params)
{
    if (m_initialized) {
        shutdown();
    }
    
    m_params.numGuideCurves = params.numGuideCurves;
    m_params.pointsPerStrand = params.pointsPerStrand;
    m_params.stiffness = params.stiffness;
    m_params.bendStiffness = params.bendStiffness;
    m_params.twistStiffness = params.twistStiffness;
    m_params.damping = params.damping;
    m_params.gravity = params.gravity;
    m_params.mass = params.mass;
    m_params.timeStep = params.timeStep;
    m_params.substeps = params.substeps;
    m_params.windStrength = params.windStrength;
    m_params.windDirection = params.windDirection;
    m_params.turbulence = params.turbulence;
    m_params.numVortexSources = params.numVortexSources;
    m_params.collisionRadius = params.collisionRadius;
    m_params.friction = params.friction;
    m_params.numColliders = 0;
    
    for (int i = 0; i < 5; i++) {
        m_params.vortexPositions[i] = params.vortexPositions[i];
        m_params.vortexStrengths[i] = params.vortexStrengths[i];
        m_params.vortexRadii[i] = params.vortexRadii[i];
    }
    
    m_numGuideCurves = params.numGuideCurves;
    m_pointsPerStrand = params.pointsPerStrand;
    m_totalPoints = m_numGuideCurves * m_pointsPerStrand;
    
    allocateBuffers();
    
    m_initialized = true;
    m_elapsedTime = 0.0f;
    
    return true;
}

void CudaHairSolver::shutdown()
{
    if (!m_initialized) return;
    
    freeBuffers();
    m_initialized = false;
}

void CudaHairSolver::allocateBuffers()
{
    cudaMalloc(&d_positions, m_totalPoints * sizeof(glm::vec3));
    cudaMalloc(&d_prevPositions, m_totalPoints * sizeof(glm::vec3));
    cudaMalloc(&d_velocities, m_totalPoints * sizeof(glm::vec3));
    cudaMalloc(&d_forces, m_totalPoints * sizeof(glm::vec3));
    
    int restLengthCount = m_numGuideCurves * (m_pointsPerStrand - 1);
    cudaMalloc(&d_restLengths, restLengthCount * sizeof(float));
    
    cudaMalloc(&d_colliders, 20 * sizeof(CapsuleCollider));
    
    cudaMemset(d_velocities, 0, m_totalPoints * sizeof(glm::vec3));
    cudaMemset(d_forces, 0, m_totalPoints * sizeof(glm::vec3));
}

void CudaHairSolver::freeBuffers()
{
    if (d_positions) cudaFree(d_positions);
    if (d_prevPositions) cudaFree(d_prevPositions);
    if (d_velocities) cudaFree(d_velocities);
    if (d_forces) cudaFree(d_forces);
    if (d_restLengths) cudaFree(d_restLengths);
    if (d_guidePositions) cudaFree(d_guidePositions);
    if (d_colliders) cudaFree(d_colliders);
    
    d_positions = nullptr;
    d_prevPositions = nullptr;
    d_velocities = nullptr;
    d_forces = nullptr;
    d_restLengths = nullptr;
    d_guidePositions = nullptr;
    d_colliders = nullptr;
}

void CudaHairSolver::uploadHairData(const HairData& hairData)
{
    const auto& guideCurves = hairData.getGuideCurves();
    
    std::vector<glm::vec3> positions(m_totalPoints);
    std::vector<float> restLengths(m_numGuideCurves * (m_pointsPerStrand - 1));
    
    for (int i = 0; i < m_numGuideCurves; i++) {
        for (int j = 0; j < m_pointsPerStrand; j++) {
            positions[i * m_pointsPerStrand + j] = guideCurves[i].points[j];
        }
        for (int j = 0; j < m_pointsPerStrand - 1; j++) {
            restLengths[i * (m_pointsPerStrand - 1) + j] = guideCurves[i].restLengths[j];
        }
    }
    
    cudaMemcpy(d_positions, positions.data(), m_totalPoints * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevPositions, positions.data(), m_totalPoints * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_restLengths, restLengths.data(), restLengths.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaHairSolver::downloadHairData(HairData& hairData)
{
    std::vector<glm::vec3> positions(m_totalPoints);
    
    cudaMemcpy(positions.data(), d_positions, m_totalPoints * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    
    auto& guideCurves = hairData.getGuideCurves();
    for (int i = 0; i < m_numGuideCurves; i++) {
        for (int j = 0; j < m_pointsPerStrand; j++) {
            guideCurves[i].points[j] = positions[i * m_pointsPerStrand + j];
        }
    }
    
    hairData.updateRenderStrandsFromGuides();
}

void CudaHairSolver::updateParams(const HairParams& params)
{
    m_params.stiffness = params.stiffness;
    m_params.bendStiffness = params.bendStiffness;
    m_params.twistStiffness = params.twistStiffness;
    m_params.damping = params.damping;
    m_params.gravity = params.gravity;
    m_params.mass = params.mass;
    m_params.windStrength = params.windStrength;
    m_params.windDirection = params.windDirection;
    m_params.turbulence = params.turbulence;
    m_params.collisionRadius = params.collisionRadius;
    m_params.friction = params.friction;
    m_params.timeStep = params.timeStep;
    m_params.substeps = params.substeps;
    
    m_params.numVortexSources = params.numVortexSources;
    for (int i = 0; i < 5; i++) {
        m_params.vortexPositions[i] = params.vortexPositions[i];
        m_params.vortexStrengths[i] = params.vortexStrengths[i];
        m_params.vortexRadii[i] = params.vortexRadii[i];
    }
}

void CudaHairSolver::setColliders(const std::vector<CapsuleCollider>& colliders)
{
    m_params.numColliders = std::min((int)colliders.size(), 20);
    
    if (m_params.numColliders > 0) {
        cudaMemcpy(d_colliders, colliders.data(), m_params.numColliders * sizeof(CapsuleCollider), cudaMemcpyHostToDevice);
    }
}

void CudaHairSolver::step(float deltaTime)
{
    if (!m_initialized) return;
    
    float totalTime = deltaTime;
    float subDt = m_params.timeStep;
    int numSubsteps = m_params.substeps;
    
    if (totalTime < subDt) {
        subDt = totalTime;
        numSubsteps = 1;
    }
    
    for (int i = 0; i < numSubsteps; i++) {
        m_elapsedTime += subDt;
        
        applyForces(subDt);
        integrate(subDt);
        solveConstraints();
        handleCollisions();
    }
    
    cudaDeviceSynchronize();
    
    m_lastFrameTime = deltaTime;
}

void CudaHairSolver::reset()
{
    m_elapsedTime = 0.0f;
    cudaMemcpy(d_prevPositions, d_positions, m_totalPoints * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemset(d_velocities, 0, m_totalPoints * sizeof(glm::vec3));
}
