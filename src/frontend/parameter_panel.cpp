#include "frontend/parameter_panel.h"
#include <imgui.h>

ParameterPanel::ParameterPanel()
    : m_x(0.0f)
    , m_y(0.0f)
    , m_width(300.0f)
    , m_height(600.0f)
    , m_paramsChanged(false)
    , m_physicsExpanded(true)
    , m_windExpanded(true)
    , m_collisionExpanded(true)
    , m_displayExpanded(true)
    , m_vortexExpanded(false)
{
}

ParameterPanel::~ParameterPanel()
{
}

void ParameterPanel::render(HairParams& params, bool& showGuides, bool& showColliders)
{
    m_paramsChanged = false;
    
    ImGui::Begin("Parameters", nullptr, 
        ImGuiWindowFlags_NoMove | 
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse);
    
    ImGui::SetWindowPos(ImVec2(m_x, m_y), ImGuiCond_Always);
    ImGui::SetWindowSize(ImVec2(m_width, m_height), ImGuiCond_Always);
    
    ImGui::Text("Hair Physics System v1.0");
    ImGui::Separator();
    
    renderPhysicsSection(params);
    renderWindSection(params);
    renderCollisionSection(params);
    renderDisplaySection(params, showGuides, showColliders);
    
    ImGui::End();
}

void ParameterPanel::renderPhysicsSection(HairParams& params)
{
    if (ImGui::CollapsingHeader("Physics", m_physicsExpanded ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
        m_physicsExpanded = true;
        
        float oldStiffness = params.stiffness;
        if (ImGui::SliderFloat("Stiffness", &params.stiffness, 0.0f, 1.0f, "%.3f")) {
            m_paramsChanged = true;
        }
        
        float oldBend = params.bendStiffness;
        if (ImGui::SliderFloat("Bend Stiffness", &params.bendStiffness, 0.0f, 10.0f, "%.3f")) {
            m_paramsChanged = true;
        }
        
        float oldTwist = params.twistStiffness;
        if (ImGui::SliderFloat("Twist Stiffness", &params.twistStiffness, 0.0f, 5.0f, "%.3f")) {
            m_paramsChanged = true;
        }
        
        float oldDamping = params.damping;
        if (ImGui::SliderFloat("Damping", &params.damping, 0.0f, 2.0f, "%.3f")) {
            m_paramsChanged = true;
        }
        
        float oldGravity = params.gravity;
        if (ImGui::SliderFloat("Gravity", &params.gravity, -20.0f, 20.0f, "%.2f")) {
            m_paramsChanged = true;
        }
        
        float oldMass = params.mass;
        if (ImGui::InputFloat("Mass", &params.mass, 0.001f, 0.01f, "%.4f")) {
            m_paramsChanged = true;
        }
        
        int oldSubsteps = params.substeps;
        if (ImGui::SliderInt("Substeps", &params.substeps, 1, 10)) {
            m_paramsChanged = true;
        }
        
        float oldTimeStep = params.timeStep;
        if (ImGui::InputFloat("Time Step", &params.timeStep, 0.001f, 0.01f, "%.4f")) {
            m_paramsChanged = true;
        }
    } else {
        m_physicsExpanded = false;
    }
}

void ParameterPanel::renderWindSection(HairParams& params)
{
    if (ImGui::CollapsingHeader("Wind", m_windExpanded ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
        m_windExpanded = true;
        
        float oldStrength = params.windStrength;
        if (ImGui::SliderFloat("Wind Strength", &params.windStrength, 0.0f, 50.0f, "%.2f")) {
            m_paramsChanged = true;
        }
        
        float oldDir[3] = { params.windDirection.x, params.windDirection.y, params.windDirection.z };
        if (ImGui::SliderFloat3("Wind Direction", oldDir, -1.0f, 1.0f, "%.2f")) {
            params.windDirection = glm::normalize(glm::vec3(oldDir[0], oldDir[1], oldDir[2]));
            m_paramsChanged = true;
        }
        
        float oldTurbulence = params.turbulence;
        if (ImGui::SliderFloat("Turbulence", &params.turbulence, 0.0f, 5.0f, "%.2f")) {
            m_paramsChanged = true;
        }
        
        if (ImGui::CollapsingHeader("Vortex Sources", m_vortexExpanded ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            m_vortexExpanded = true;
            
            int oldNumVortex = params.numVortexSources;
            if (ImGui::SliderInt("Num Vortex", &params.numVortexSources, 0, 5)) {
                m_paramsChanged = true;
            }
            
            for (int i = 0; i < params.numVortexSources; i++) {
                char label[32];
                sprintf(label, "Vortex %d", i + 1);
                
                if (ImGui::TreeNode(label)) {
                    float pos[3] = { params.vortexPositions[i].x, params.vortexPositions[i].y, params.vortexPositions[i].z };
                    if (ImGui::InputFloat3("Position", pos, "%.2f")) {
                        params.vortexPositions[i] = glm::vec3(pos[0], pos[1], pos[2]);
                        m_paramsChanged = true;
                    }
                    
                    if (ImGui::SliderFloat("Strength", &params.vortexStrengths[i], 0.0f, 20.0f, "%.2f")) {
                        m_paramsChanged = true;
                    }
                    
                    if (ImGui::SliderFloat("Radius", &params.vortexRadii[i], 0.1f, 5.0f, "%.2f")) {
                        m_paramsChanged = true;
                    }
                    
                    ImGui::TreePop();
                }
            }
        } else {
            m_vortexExpanded = false;
        }
    } else {
        m_windExpanded = false;
    }
}

void ParameterPanel::renderCollisionSection(HairParams& params)
{
    if (ImGui::CollapsingHeader("Collision", m_collisionExpanded ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
        m_collisionExpanded = true;
        
        float oldRadius = params.collisionRadius;
        if (ImGui::SliderFloat("Collision Radius", &params.collisionRadius, 0.001f, 0.1f, "%.4f")) {
            m_paramsChanged = true;
        }
        
        float oldFriction = params.friction;
        if (ImGui::SliderFloat("Friction", &params.friction, 0.0f, 1.0f, "%.3f")) {
            m_paramsChanged = true;
        }
        
        ImGui::Text("Capsule colliders: auto-generated");
    } else {
        m_collisionExpanded = false;
    }
}

void ParameterPanel::renderDisplaySection(HairParams& params, bool& showGuides, bool& showColliders)
{
    if (ImGui::CollapsingHeader("Display", m_displayExpanded ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
        m_displayExpanded = true;
        
        float oldThickness = params.hairThickness;
        if (ImGui::SliderFloat("Hair Thickness", &params.hairThickness, 0.001f, 0.02f, "%.4f")) {
            m_paramsChanged = true;
        }
        
        float oldColor[3] = { params.hairColor.r, params.hairColor.g, params.hairColor.b };
        if (ImGui::ColorEdit3("Hair Color", oldColor)) {
            params.hairColor = glm::vec3(oldColor[0], oldColor[1], oldColor[2]);
            m_paramsChanged = true;
        }
        
        float oldScale = params.simulationScale;
        if (ImGui::SliderFloat("Scale", &params.simulationScale, 0.1f, 10.0f, "%.2f")) {
            m_paramsChanged = true;
        }
        
        if (ImGui::Checkbox("Show Guide Curves", &showGuides)) {
            m_paramsChanged = true;
        }
        
        if (ImGui::Checkbox("Show Colliders", &showColliders)) {
            m_paramsChanged = true;
        }
        
        ImGui::Text("Guide curves: %d", params.numGuideCurves);
        ImGui::Text("Render strands: %d", params.numRenderStrands);
        ImGui::Text("Points/strand: %d", params.pointsPerStrand);
    } else {
        m_displayExpanded = false;
    }
}
