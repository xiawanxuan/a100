#pragma once

#include <vector>
#include <glm/glm.hpp>

struct GuideCurve {
    std::vector<glm::vec3> points;
    std::vector<float> restLengths;
    int strandId;
};

struct RenderStrand {
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> tangents;
    float thickness;
    glm::vec3 color;
};

struct HairParams {
    int numGuideCurves;
    int numRenderStrands;
    int pointsPerStrand;
    
    float stiffness;
    float bendStiffness;
    float twistStiffness;
    float damping;
    float gravity;
    float mass;
    
    float windStrength;
    glm::vec3 windDirection;
    float turbulence;
    
    int numVortexSources;
    glm::vec3 vortexPositions[5];
    float vortexStrengths[5];
    float vortexRadii[5];
    
    float collisionRadius;
    float friction;
    
    float strandLength;
    float hairThickness;
    glm::vec3 hairColor;
    
    float simulationScale;
    float timeStep;
    int substeps;
};

struct CapsuleCollider {
    glm::vec3 start;
    glm::vec3 end;
    float radius;
};

class HairData {
public:
    HairData();
    ~HairData();
    
    void initialize(const HairParams& params);
    void reset();
    
    int getNumGuideCurves() const { return m_numGuideCurves; }
    int getNumRenderStrands() const { return m_numRenderStrands; }
    int getPointsPerStrand() const { return m_pointsPerStrand; }
    
    const std::vector<GuideCurve>& getGuideCurves() const { return m_guideCurves; }
    std::vector<GuideCurve>& getGuideCurves() { return m_guideCurves; }
    
    const std::vector<RenderStrand>& getRenderStrands() const { return m_renderStrands; }
    std::vector<RenderStrand>& getRenderStrands() { return m_renderStrands; }
    
    const std::vector<CapsuleCollider>& getColliders() const { return m_colliders; }
    std::vector<CapsuleCollider>& getColliders() { return m_colliders; }
    
    const HairParams& getParams() const { return m_params; }
    HairParams& getParams() { return m_params; }
    
    void updateRenderStrandsFromGuides();
    
private:
    void generateGuideCurves();
    void generateRenderStrands();
    void computeRestLengths();
    
    int m_numGuideCurves;
    int m_numRenderStrands;
    int m_pointsPerStrand;
    
    HairParams m_params;
    std::vector<GuideCurve> m_guideCurves;
    std::vector<RenderStrand> m_renderStrands;
    std::vector<CapsuleCollider> m_colliders;
    
    std::vector<int> m_guideIndices;
    std::vector<float> m_interpWeights;
};
