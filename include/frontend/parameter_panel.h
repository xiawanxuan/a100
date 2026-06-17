#pragma once

#include "common/hair_data.h"

class ParameterPanel {
public:
    ParameterPanel();
    ~ParameterPanel();
    
    void render(HairParams& params, bool& showGuides, bool& showColliders);
    
    void setPosition(float x, float y) { m_x = x; m_y = y; }
    void setSize(float width, float height) { m_width = width; m_height = height; }
    
    bool isParamsChanged() const { return m_paramsChanged; }
    void clearParamsChanged() { m_paramsChanged = false; }

private:
    void renderPhysicsSection(HairParams& params);
    void renderWindSection(HairParams& params);
    void renderCollisionSection(HairParams& params);
    void renderDisplaySection(HairParams& params, bool& showGuides, bool& showColliders);
    
    float m_x;
    float m_y;
    float m_width;
    float m_height;
    
    bool m_paramsChanged;
    bool m_physicsExpanded;
    bool m_windExpanded;
    bool m_collisionExpanded;
    bool m_displayExpanded;
    bool m_vortexExpanded;
};
