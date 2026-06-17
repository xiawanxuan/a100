#include "frontend/renderer.h"
#include <iostream>
#include <sstream>

const std::string HairRenderer::HAIR_VERTEX_SHADER = R"(
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

void main() {
    vec4 viewPos = uViewMatrix * vec4(aPosition, 1.0);
    gl_Position = uProjectionMatrix * viewPos;
    
    vColor = aColor;
    vSegmentId = aSegmentId;
    
    vec3 normal = normalize(vec3(0.0, 1.0, 0.0));
    vNormal = normal;
    vViewDir = normalize(-viewPos.xyz);
}
)";

const std::string HairRenderer::HAIR_FRAGMENT_SHADER = R"(
#version 430 core

in vec3 vColor;
in vec3 vNormal;
in vec3 vViewDir;
in float vSegmentId;

uniform vec3 uLightDir;
uniform vec3 uHairColor;

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
    
    vec3 ambient = baseColor * 0.3;
    vec3 diffuse = baseColor * ndotl * 0.7;
    vec3 spec = vec3(0.3) * specular;
    
    vec3 finalColor = ambient + diffuse + spec;
    
    finalColor = pow(finalColor, vec3(1.0 / 2.2));
    
    fragColor = vec4(finalColor, 1.0);
}
)";

const std::string HairRenderer::GUIDE_VERTEX_SHADER = R"(
#version 430 core

layout(location = 0) in vec3 aPosition;

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

void main() {
    gl_Position = uProjectionMatrix * uViewMatrix * vec4(aPosition, 1.0);
}
)";

const std::string HairRenderer::GUIDE_FRAGMENT_SHADER = R"(
#version 430 core

uniform vec4 uColor;

out vec4 fragColor;

void main() {
    fragColor = uColor;
}
)";

const std::string HairRenderer::SOLID_VERTEX_SHADER = R"(
#version 430 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

out vec3 vNormal;
out vec3 vViewDir;

void main() {
    vec4 viewPos = uViewMatrix * vec4(aPosition, 1.0);
    gl_Position = uProjectionMatrix * viewPos;
    
    vNormal = mat3(uViewMatrix) * aNormal;
    vViewDir = normalize(-viewPos.xyz);
}
)";

const std::string HairRenderer::SOLID_FRAGMENT_SHADER = R"(
#version 430 core

in vec3 vNormal;
in vec3 vViewDir;

uniform vec3 uColor;
uniform vec3 uLightDir;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);
    
    float ndotl = max(dot(normal, lightDir), 0.0);
    
    vec3 ambient = uColor * 0.3;
    vec3 diffuse = uColor * ndotl * 0.7;
    
    vec3 finalColor = ambient + diffuse;
    finalColor = pow(finalColor, vec3(1.0 / 2.2));
    
    fragColor = vec4(finalColor, 0.8);
}
)";

HairRenderer::HairRenderer()
    : m_initialized(false)
    , m_viewportWidth(800)
    , m_viewportHeight(600)
    , m_viewMatrix(1.0f)
    , m_projectionMatrix(1.0f)
    , m_lightDirection(0.5f, 1.0f, 0.3f)
    , m_hairShader(0)
    , m_guideShader(0)
    , m_solidShader(0)
    , m_hairVAO(0)
    , m_hairVBO(0)
    , m_hairColorVBO(0)
    , m_hairIndexVBO(0)
    , m_guideVAO(0)
    , m_guideVBO(0)
    , m_numHairPoints(0)
    , m_numHairIndices(0)
    , m_numGuidePoints(0)
    , m_showGuideCurves(false)
    , m_showColliders(false)
{
}

HairRenderer::~HairRenderer()
{
    shutdown();
}

bool HairRenderer::initialize()
{
    if (m_initialized) return true;
    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }
    
    if (!compileShaders()) {
        std::cerr << "Failed to compile shaders" << std::endl;
        return false;
    }
    
    createBuffers();
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    m_initialized = true;
    return true;
}

void HairRenderer::shutdown()
{
    if (!m_initialized) return;
    
    if (m_hairVAO) glDeleteVertexArrays(1, &m_hairVAO);
    if (m_hairVBO) glDeleteBuffers(1, &m_hairVBO);
    if (m_hairColorVBO) glDeleteBuffers(1, &m_hairColorVBO);
    if (m_hairIndexVBO) glDeleteBuffers(1, &m_hairIndexVBO);
    
    if (m_guideVAO) glDeleteVertexArrays(1, &m_guideVAO);
    if (m_guideVBO) glDeleteBuffers(1, &m_guideVBO);
    
    if (m_hairShader) glDeleteProgram(m_hairShader);
    if (m_guideShader) glDeleteProgram(m_guideShader);
    if (m_solidShader) glDeleteProgram(m_solidShader);
    
    m_initialized = false;
}

bool HairRenderer::compileShaders()
{
    GLuint hairVS = compileShader(GL_VERTEX_SHADER, HAIR_VERTEX_SHADER);
    GLuint hairFS = compileShader(GL_FRAGMENT_SHADER, HAIR_FRAGMENT_SHADER);
    
    if (hairVS == 0 || hairFS == 0) return false;
    
    m_hairShader = glCreateProgram();
    glAttachShader(m_hairShader, hairVS);
    glAttachShader(m_hairShader, hairFS);
    glLinkProgram(m_hairShader);
    
    GLint success;
    glGetProgramiv(m_hairShader, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(m_hairShader, 512, NULL, infoLog);
        std::cerr << "Hair shader link error: " << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(hairVS);
    glDeleteShader(hairFS);
    
    GLuint guideVS = compileShader(GL_VERTEX_SHADER, GUIDE_VERTEX_SHADER);
    GLuint guideFS = compileShader(GL_FRAGMENT_SHADER, GUIDE_FRAGMENT_SHADER);
    
    if (guideVS == 0 || guideFS == 0) return false;
    
    m_guideShader = glCreateProgram();
    glAttachShader(m_guideShader, guideVS);
    glAttachShader(m_guideShader, guideFS);
    glLinkProgram(m_guideShader);
    
    glGetProgramiv(m_guideShader, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(m_guideShader, 512, NULL, infoLog);
        std::cerr << "Guide shader link error: " << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(guideVS);
    glDeleteShader(guideFS);
    
    GLuint solidVS = compileShader(GL_VERTEX_SHADER, SOLID_VERTEX_SHADER);
    GLuint solidFS = compileShader(GL_FRAGMENT_SHADER, SOLID_FRAGMENT_SHADER);
    
    if (solidVS == 0 || solidFS == 0) return false;
    
    m_solidShader = glCreateProgram();
    glAttachShader(m_solidShader, solidVS);
    glAttachShader(m_solidShader, solidFS);
    glLinkProgram(m_solidShader);
    
    glGetProgramiv(m_solidShader, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(m_solidShader, 512, NULL, infoLog);
        std::cerr << "Solid shader link error: " << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(solidVS);
    glDeleteShader(solidFS);
    
    return true;
}

GLuint HairRenderer::compileShader(GLenum type, const std::string& source)
{
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compile error: " << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

void HairRenderer::createBuffers()
{
    glGenVertexArrays(1, &m_hairVAO);
    glGenBuffers(1, &m_hairVBO);
    glGenBuffers(1, &m_hairColorVBO);
    glGenBuffers(1, &m_hairIndexVBO);
    
    glGenVertexArrays(1, &m_guideVAO);
    glGenBuffers(1, &m_guideVBO);
}

void HairRenderer::updateBuffers(const HairData& hairData)
{
    const auto& strands = hairData.getRenderStrands();
    int numStrands = strands.size();
    int pointsPerStrand = hairData.getPointsPerStrand();
    int totalPoints = numStrands * pointsPerStrand;
    
    std::vector<glm::vec3> positions(totalPoints);
    std::vector<glm::vec3> colors(totalPoints);
    std::vector<GLuint> indices;
    
    for (int i = 0; i < numStrands; i++) {
        for (int j = 0; j < pointsPerStrand; j++) {
            int idx = i * pointsPerStrand + j;
            positions[idx] = strands[i].points[j];
            colors[idx] = strands[i].color;
        }
    }
    
    for (int i = 0; i < numStrands; i++) {
        int base = i * pointsPerStrand;
        for (int j = 0; j < pointsPerStrand - 1; j++) {
            indices.push_back(base + j);
            indices.push_back(base + j + 1);
        }
    }
    
    m_numHairPoints = totalPoints;
    m_numHairIndices = indices.size();
    
    glBindVertexArray(m_hairVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_hairVBO);
    glBufferData(GL_ARRAY_BUFFER, totalPoints * sizeof(glm::vec3), positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_hairColorVBO);
    glBufferData(GL_ARRAY_BUFFER, totalPoints * sizeof(glm::vec3), colors.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_hairIndexVBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);
    
    glBindVertexArray(0);
    
    const auto& guideCurves = hairData.getGuideCurves();
    int numGuides = guideCurves.size();
    int totalGuidePoints = numGuides * pointsPerStrand;
    
    std::vector<glm::vec3> guidePositions(totalGuidePoints);
    
    for (int i = 0; i < numGuides; i++) {
        for (int j = 0; j < pointsPerStrand; j++) {
            guidePositions[i * pointsPerStrand + j] = guideCurves[i].points[j];
        }
    }
    
    m_numGuidePoints = totalGuidePoints;
    
    glBindVertexArray(m_guideVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_guideVBO);
    glBufferData(GL_ARRAY_BUFFER, totalGuidePoints * sizeof(glm::vec3), guidePositions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void HairRenderer::setViewport(int width, int height)
{
    m_viewportWidth = width;
    m_viewportHeight = height;
    glViewport(0, 0, width, height);
}

void HairRenderer::render(const HairData& hairData)
{
    if (!m_initialized) return;
    
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    updateBuffers(hairData);
    
    renderHairStrands(hairData);
    
    if (m_showGuideCurves) {
        renderGuideCurves(hairData);
    }
    
    if (m_showColliders) {
        renderColliders(hairData);
    }
}

void HairRenderer::renderHairStrands(const HairData& hairData)
{
    glUseProgram(m_hairShader);
    
    glUniformMatrix4fv(glGetUniformLocation(m_hairShader, "uViewMatrix"), 1, GL_FALSE, &m_viewMatrix[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(m_hairShader, "uProjectionMatrix"), 1, GL_FALSE, &m_projectionMatrix[0][0]);
    glUniform3fv(glGetUniformLocation(m_hairShader, "uLightDir"), 1, &m_lightDirection[0]);
    glUniform1f(glGetUniformLocation(m_hairShader, "uHairThickness"), hairData.getParams().hairThickness);
    glUniform3fv(glGetUniformLocation(m_hairShader, "uHairColor"), 1, &hairData.getParams().hairColor[0]);
    
    glBindVertexArray(m_hairVAO);
    glDrawElements(GL_LINES, m_numHairIndices, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    glUseProgram(0);
}

void HairRenderer::renderGuideCurves(const HairData& hairData)
{
    glUseProgram(m_guideShader);
    
    glUniformMatrix4fv(glGetUniformLocation(m_guideShader, "uViewMatrix"), 1, GL_FALSE, &m_viewMatrix[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(m_guideShader, "uProjectionMatrix"), 1, GL_FALSE, &m_projectionMatrix[0][0]);
    glUniform4f(glGetUniformLocation(m_guideShader, "uColor"), 0.0f, 1.0f, 0.0f, 1.0f);
    
    glBindVertexArray(m_guideVAO);
    
    int pointsPerStrand = hairData.getPointsPerStrand();
    int numGuides = hairData.getNumGuideCurves();
    
    for (int i = 0; i < numGuides; i++) {
        glDrawArrays(GL_LINE_STRIP, i * pointsPerStrand, pointsPerStrand);
    }
    
    glBindVertexArray(0);
    glUseProgram(0);
}

void HairRenderer::renderColliders(const HairData& hairData)
{
    const auto& colliders = hairData.getColliders();
    
    glUseProgram(m_solidShader);
    
    glUniformMatrix4fv(glGetUniformLocation(m_solidShader, "uViewMatrix"), 1, GL_FALSE, &m_viewMatrix[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(m_solidShader, "uProjectionMatrix"), 1, GL_FALSE, &m_projectionMatrix[0][0]);
    glUniform3fv(glGetUniformLocation(m_solidShader, "uLightDir"), 1, &m_lightDirection[0]);
    glUniform3f(glGetUniformLocation(m_solidShader, "uColor"), 0.8f, 0.2f, 0.2f);
    
    for (const auto& collider : colliders) {
        glm::vec3 dir = collider.end - collider.start;
        float length = glm::length(dir);
        if (length < 0.001f) continue;
        
        glm::mat4 model = glm::mat4(1.0f);
        glm::vec3 mid = (collider.start + collider.end) * 0.5f;
        model = glm::translate(model, mid);
        
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 dirNorm = glm::normalize(dir);
        float angle = acosf(glm::dot(up, dirNorm));
        glm::vec3 axis = glm::cross(up, dirNorm);
        if (glm::length(axis) > 0.001f) {
            model = glm::rotate(model, angle, glm::normalize(axis));
        }
        model = glm::scale(model, glm::vec3(collider.radius, length * 0.5f, collider.radius));
        
        GLint modelLoc = glGetUniformLocation(m_solidShader, "uModelMatrix");
        
        const int segments = 16;
        const int stacks = 8;
        
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<GLuint> indices;
        
        for (int i = 0; i <= stacks; i++) {
            float y = -1.0f + (2.0f * i) / float(stacks);
            float r = 1.0f;
            
            for (int j = 0; j <= segments; j++) {
                float angle = 2.0f * 3.14159f * j / float(segments);
                float x = cosf(angle) * r;
                float z = sinf(angle) * r;
                
                vertices.push_back(glm::vec3(x, y, z));
                normals.push_back(glm::normalize(glm::vec3(x, 0.0f, z)));
            }
        }
        
        for (int i = 0; i < stacks; i++) {
            for (int j = 0; j < segments; j++) {
                int a = i * (segments + 1) + j;
                int b = a + 1;
                int c = a + segments + 1;
                int d = c + 1;
                
                indices.push_back(a);
                indices.push_back(c);
                indices.push_back(b);
                indices.push_back(b);
                indices.push_back(c);
                indices.push_back(d);
            }
        }
        
        GLuint tempVAO, tempVBO, tempNBO, tempIBO;
        glGenVertexArrays(1, &tempVAO);
        glGenBuffers(1, &tempVBO);
        glGenBuffers(1, &tempNBO);
        glGenBuffers(1, &tempIBO);
        
        glBindVertexArray(tempVAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, tempVBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        
        glBindBuffer(GL_ARRAY_BUFFER, tempNBO);
        glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempIBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
        
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        
        glDeleteBuffers(1, &tempVBO);
        glDeleteBuffers(1, &tempNBO);
        glDeleteBuffers(1, &tempIBO);
        glDeleteVertexArrays(1, &tempVAO);
    }
    
    glUseProgram(0);
}
