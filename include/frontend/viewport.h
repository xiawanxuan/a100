#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Viewport {
public:
    Viewport();
    ~Viewport();
    
    void setSize(int width, int height);
    
    void onMouseDown(int button, float x, float y);
    void onMouseUp(int button, float x, float y);
    void onMouseMove(float x, float y);
    void onMouseWheel(float delta);
    
    glm::mat4 getViewMatrix() const { return m_viewMatrix; }
    glm::mat4 getProjectionMatrix() const { return m_projectionMatrix; }
    glm::vec3 getCameraPosition() const { return m_cameraPos; }
    
    float getFov() const { return m_fov; }
    void setFov(float fov) { m_fov = fov; updateProjection(); }
    
    float getNearPlane() const { return m_nearPlane; }
    void setNearPlane(float near) { m_nearPlane = near; updateProjection(); }
    
    float getFarPlane() const { return m_farPlane; }
    void setFarPlane(float far) { m_farPlane = far; updateProjection(); }
    
    glm::vec3 getTarget() const { return m_target; }
    void setTarget(const glm::vec3& target) { m_target = target; updateView(); }
    
    float getDistance() const { return m_distance; }
    void setDistance(float distance) { m_distance = distance; updateView(); }
    
    float getYaw() const { return m_yaw; }
    void setYaw(float yaw) { m_yaw = yaw; updateView(); }
    
    float getPitch() const { return m_pitch; }
    void setPitch(float pitch) { m_pitch = glm::clamp(pitch, -89.0f, 89.0f); updateView(); }

private:
    void updateView();
    void updateProjection();
    
    int m_width;
    int m_height;
    
    float m_fov;
    float m_nearPlane;
    float m_farPlane;
    
    glm::vec3 m_cameraPos;
    glm::vec3 m_target;
    float m_distance;
    float m_yaw;
    float m_pitch;
    
    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;
    
    bool m_isRotating;
    bool m_isPanning;
    bool m_isZooming;
    float m_lastMouseX;
    float m_lastMouseY;
    
    float m_rotationSpeed;
    float m_panSpeed;
    float m_zoomSpeed;
};
