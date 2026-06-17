#include "frontend/viewport.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Viewport::Viewport()
    : m_width(800)
    , m_height(600)
    , m_fov(45.0f)
    , m_nearPlane(0.1f)
    , m_farPlane(1000.0f)
    , m_cameraPos(0.0f, 1.0f, 3.0f)
    , m_target(0.0f, 0.5f, 0.0f)
    , m_distance(3.0f)
    , m_yaw(-90.0f)
    , m_pitch(20.0f)
    , m_viewMatrix(1.0f)
    , m_projectionMatrix(1.0f)
    , m_isRotating(false)
    , m_isPanning(false)
    , m_isZooming(false)
    , m_lastMouseX(0.0f)
    , m_lastMouseY(0.0f)
    , m_rotationSpeed(0.5f)
    , m_panSpeed(0.01f)
    , m_zoomSpeed(0.05f)
{
    updateView();
    updateProjection();
}

Viewport::~Viewport()
{
}

void Viewport::setSize(int width, int height)
{
    m_width = width;
    m_height = height;
    updateProjection();
}

void Viewport::onMouseDown(int button, float x, float y)
{
    m_lastMouseX = x;
    m_lastMouseY = y;
    
    if (button == 0) {
        m_isRotating = true;
    } else if (button == 1) {
        m_isPanning = true;
    } else if (button == 2) {
        m_isZooming = true;
    }
}

void Viewport::onMouseUp(int button, float x, float y)
{
    if (button == 0) m_isRotating = false;
    if (button == 1) m_isPanning = false;
    if (button == 2) m_isZooming = false;
}

void Viewport::onMouseMove(float x, float y)
{
    float deltaX = x - m_lastMouseX;
    float deltaY = y - m_lastMouseY;
    
    if (m_isRotating) {
        m_yaw += deltaX * m_rotationSpeed;
        m_pitch -= deltaY * m_rotationSpeed;
        m_pitch = std::max(-89.0f, std::min(89.0f, m_pitch));
        updateView();
    }
    
    if (m_isPanning) {
        float panAmount = m_distance * m_panSpeed;
        
        glm::vec3 forward = glm::normalize(m_target - m_cameraPos);
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 up = glm::normalize(glm::cross(right, forward));
        
        m_target -= right * deltaX * panAmount;
        m_target += up * deltaY * panAmount;
        updateView();
    }
    
    if (m_isZooming) {
        m_distance += deltaY * m_zoomSpeed * m_distance;
        m_distance = std::max(0.1f, m_distance);
        updateView();
    }
    
    m_lastMouseX = x;
    m_lastMouseY = y;
}

void Viewport::onMouseWheel(float delta)
{
    m_distance -= delta * m_zoomSpeed * m_distance;
    m_distance = std::max(0.1f, m_distance);
    updateView();
}

void Viewport::updateView()
{
    float yawRad = glm::radians(m_yaw);
    float pitchRad = glm::radians(m_pitch);
    
    m_cameraPos.x = m_target.x + m_distance * cosf(pitchRad) * sinf(yawRad);
    m_cameraPos.y = m_target.y + m_distance * sinf(pitchRad);
    m_cameraPos.z = m_target.z + m_distance * cosf(pitchRad) * cosf(yawRad);
    
    m_viewMatrix = glm::lookAt(m_cameraPos, m_target, glm::vec3(0.0f, 1.0f, 0.0f));
}

void Viewport::updateProjection()
{
    float aspect = float(m_width) / float(std::max(1, m_height));
    m_projectionMatrix = glm::perspective(glm::radians(m_fov), aspect, m_nearPlane, m_farPlane);
}
