#include "common/hair_data.h"
#include <glm/gtc/random.hpp>
#include <cmath>
#include <algorithm>

HairData::HairData()
    : m_numGuideCurves(0)
    , m_numRenderStrands(0)
    , m_pointsPerStrand(0)
{
}

HairData::~HairData()
{
}

void HairData::initialize(const HairParams& params)
{
    m_params = params;
    m_numGuideCurves = params.numGuideCurves;
    m_numRenderStrands = params.numRenderStrands;
    m_pointsPerStrand = params.pointsPerStrand;
    
    generateGuideCurves();
    generateRenderStrands();
    computeRestLengths();
}

void HairData::reset()
{
    generateGuideCurves();
    computeRestLengths();
    updateRenderStrandsFromGuides();
}

void HairData::generateGuideCurves()
{
    m_guideCurves.resize(m_numGuideCurves);
    
    const float headRadius = 0.3f * m_params.simulationScale;
    const float strandLength = m_params.strandLength * m_params.simulationScale;
    
    for (int i = 0; i < m_numGuideCurves; ++i) {
        GuideCurve& curve = m_guideCurves[i];
        curve.strandId = i;
        curve.points.resize(m_pointsPerStrand);
        curve.restLengths.resize(m_pointsPerStrand - 1);
        
        float theta = 2.0f * 3.14159f * float(i) / float(m_numGuideCurves);
        float phi = 3.14159f * 0.25f + 0.5f * (float(i) / float(m_numGuideCurves)) * 3.14159f * 0.5f;
        
        glm::vec3 rootPos(
            headRadius * sinf(phi) * cosf(theta),
            headRadius * cosf(phi) + headRadius,
            headRadius * sinf(phi) * sinf(theta)
        );
        
        glm::vec3 direction = glm::normalize(rootPos);
        direction.y += glm::vec3(0.0f, 0.1f, 0.0f);
        direction = glm::normalize(direction);
        
        float segmentLength = strandLength / float(m_pointsPerStrand - 1);
        
        for (int j = 0; j < m_pointsPerStrand; ++j) {
            curve.points[j] = rootPos + direction * segmentLength * float(j);
        }
        
        for (int j = 0; j < m_pointsPerStrand - 1; ++j) {
            curve.restLengths[j] = segmentLength;
        }
    }
}

void HairData::generateRenderStrands()
{
    m_renderStrands.resize(m_numRenderStrands);
    m_guideIndices.resize(m_numRenderStrands * 3);
    m_interpWeights.resize(m_numRenderStrands * 3);
    
    for (int i = 0; i < m_numRenderStrands; ++i) {
        RenderStrand& strand = m_renderStrands[i];
        strand.points.resize(m_pointsPerStrand);
        strand.tangents.resize(m_pointsPerStrand);
        strand.thickness = m_params.hairThickness * m_params.simulationScale;
        strand.color = m_params.hairColor;
        
        float t = float(i) / float(m_numRenderStrands - 1);
        int guideIdx0 = int(t * float(m_numGuideCurves - 1));
        guideIdx0 = std::min(guideIdx0, m_numGuideCurves - 2);
        float frac = t * float(m_numGuideCurves - 1) - float(guideIdx0);
        
        m_guideIndices[i * 3] = guideIdx0;
        m_guideIndices[i * 3 + 1] = guideIdx0 + 1;
        m_guideIndices[i * 3 + 2] = (guideIdx0 + 2) % m_numGuideCurves;
        
        m_interpWeights[i * 3] = 1.0f - frac;
        m_interpWeights[i * 3 + 1] = frac * 0.7f;
        m_interpWeights[i * 3 + 2] = frac * 0.3f;
    }
    
    updateRenderStrandsFromGuides();
}

void HairData::computeRestLengths()
{
    for (int i = 0; i < m_numGuideCurves; ++i) {
        GuideCurve& curve = m_guideCurves[i];
        for (int j = 0; j < m_pointsPerStrand - 1; ++j) {
            curve.restLengths[j] = glm::length(curve.points[j + 1] - curve.points[j]);
        }
    }
}

void HairData::updateRenderStrandsFromGuides()
{
    for (int i = 0; i < m_numRenderStrands; ++i) {
        RenderStrand& strand = m_renderStrands[i];
        
        int gi0 = m_guideIndices[i * 3];
        int gi1 = m_guideIndices[i * 3 + 1];
        int gi2 = m_guideIndices[i * 3 + 2];
        
        float w0 = m_interpWeights[i * 3];
        float w1 = m_interpWeights[i * 3 + 1];
        float w2 = m_interpWeights[i * 3 + 2];
        
        float totalWeight = w0 + w1 + w2;
        w0 /= totalWeight;
        w1 /= totalWeight;
        w2 /= totalWeight;
        
        const GuideCurve& g0 = m_guideCurves[gi0];
        const GuideCurve& g1 = m_guideCurves[gi1];
        const GuideCurve& g2 = m_guideCurves[gi2];
        
        for (int j = 0; j < m_pointsPerStrand; ++j) {
            strand.points[j] = g0.points[j] * w0 + g1.points[j] * w1 + g2.points[j] * w2;
        }
        
        for (int j = 0; j < m_pointsPerStrand - 1; ++j) {
            strand.tangents[j] = glm::normalize(strand.points[j + 1] - strand.points[j]);
        }
        if (m_pointsPerStrand > 1) {
            strand.tangents[m_pointsPerStrand - 1] = strand.tangents[m_pointsPerStrand - 2];
        }
    }
}
