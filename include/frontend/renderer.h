#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>
#include "common/hair_data.h"

class HairRenderer {
public:
    HairRenderer();
    ~HairRenderer();
    
    bool initialize();
    void shutdown();
    
    void setViewport(int width, int height);
    void render(const HairData& hairData);
    
    void setViewMatrix(const glm::mat4& view) { m_viewMatrix = view; }
    void setProjectionMatrix(const glm::mat4& proj) { m_projectionMatrix = proj; }
    void setLightDirection(const glm::vec3& dir) { m_lightDirection = glm::normalize(dir); }
    
    void setShowGuideCurves(bool show) { m_showGuideCurves = show; }
    void setShowColliders(bool show) { m_showColliders = show; }
    
    GLuint getShaderProgram() const { return m_hairShader; }
    
private:
    bool compileShaders();
    void createBuffers();
    void updateBuffers(const HairData& hairData);
    
    void renderGuideCurves(const HairData& hairData);
    void renderHairStrands(const HairData& hairData);
    void renderColliders(const HairData& hairData);
    
    GLuint compileShader(GLenum type, const std::string& source);
    std::string loadShaderSource(const std::string& filename);
    
    bool m_initialized;
    int m_viewportWidth;
    int m_viewportHeight;
    
    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;
    glm::vec3 m_lightDirection;
    
    GLuint m_hairShader;
    GLuint m_guideShader;
    GLuint m_solidShader;
    
    GLuint m_hairVAO;
    GLuint m_hairVBO;
    GLuint m_hairColorVBO;
    GLuint m_hairIndexVBO;
    
    GLuint m_guideVAO;
    GLuint m_guideVBO;
    
    int m_numHairPoints;
    int m_numHairIndices;
    int m_numGuidePoints;
    
    bool m_showGuideCurves;
    bool m_showColliders;
    
    static const std::string HAIR_VERTEX_SHADER;
    static const std::string HAIR_FRAGMENT_SHADER;
    static const std::string GUIDE_VERTEX_SHADER;
    static const std::string GUIDE_FRAGMENT_SHADER;
    static const std::string SOLID_VERTEX_SHADER;
    static const std::string SOLID_FRAGMENT_SHADER;
};
