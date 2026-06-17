#include "frontend/timeline.h"
#include <imgui.h>
#include <cstdio>

Timeline::Timeline()
    : m_x(0.0f)
    , m_y(0.0f)
    , m_width(800.0f)
    , m_height(120.0f)
    , m_currentTime(0.0f)
    , m_duration(10.0f)
    , m_fps(30.0f)
    , m_isPlaying(false)
    , m_isRecording(false)
    , m_loop(true)
    , m_exportRequested(false)
    , m_isDragging(false)
{
    std::snprintf(m_exportFilename, sizeof(m_exportFilename), "hair_cache.abc");
}

Timeline::~Timeline()
{
}

void Timeline::update(float deltaTime)
{
    if (m_isPlaying) {
        m_currentTime += deltaTime;
        
        if (m_currentTime >= m_duration) {
            if (m_loop) {
                m_currentTime = 0.0f;
            } else {
                m_currentTime = m_duration;
                m_isPlaying = false;
            }
        }
    }
}

void Timeline::render()
{
    ImGui::Begin("Timeline", nullptr,
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoTitleBar);
    
    ImGui::SetWindowPos(ImVec2(m_x, m_y), ImGuiCond_Always);
    ImGui::SetWindowSize(ImVec2(m_width, m_height), ImGuiCond_Always);
    
    renderPlaybackControls();
    renderTimelineTrack();
    renderRecordingControls();
    
    ImGui::End();
}

void Timeline::renderPlaybackControls()
{
    ImGui::BeginGroup();
    
    if (ImGui::Button(m_isPlaying ? "Pause" : "Play", ImVec2(60, 24))) {
        m_isPlaying = !m_isPlaying;
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("Stop", ImVec2(60, 24))) {
        m_isPlaying = false;
        m_currentTime = 0.0f;
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("<<", ImVec2(30, 24))) {
        m_currentTime = 0.0f;
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button(">>", ImVec2(30, 24))) {
        m_currentTime = m_duration;
    }
    
    ImGui::SameLine();
    
    ImGui::Checkbox("Loop", &m_loop);
    
    ImGui::SameLine();
    ImGui::Text("FPS: %.1f", m_fps);
    
    ImGui::SameLine();
    char timeStr[64];
    std::snprintf(timeStr, sizeof(timeStr), "%.2f / %.2f s", m_currentTime, m_duration);
    ImGui::Text("%s", timeStr);
    
    ImGui::SameLine();
    int curFrame = getCurrentFrame();
    int totalFrames = (int)(m_duration * m_fps);
    char frameStr[64];
    std::snprintf(frameStr, sizeof(frameStr), "Frame %d / %d", curFrame, totalFrames);
    ImGui::Text("%s", frameStr);
    
    ImGui::EndGroup();
}

void Timeline::renderTimelineTrack()
{
    float trackHeight = 30.0f;
    float trackY = ImGui::GetCursorPosY() + 5.0f;
    float trackX = 10.0f;
    float trackWidth = m_width - 20.0f;
    
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 trackStart = ImVec2(canvasPos.x + trackX, canvasPos.y + trackY);
    ImVec2 trackEnd = ImVec2(canvasPos.x + trackX + trackWidth, canvasPos.y + trackY + trackHeight);
    
    drawList->AddRectFilled(trackStart, trackEnd, IM_COL32(40, 40, 50, 255));
    drawList->AddRect(trackStart, trackEnd, IM_COL32(80, 80, 100, 255));
    
    int numMarkers = (int)m_duration + 1;
    for (int i = 0; i <= numMarkers; i++) {
        float x = trackStart.x + (float(i) / m_duration) * trackWidth;
        drawList->AddLine(
            ImVec2(x, trackStart.y),
            ImVec2(x, trackStart.y + 5.0f),
            IM_COL32(100, 100, 120, 255)
        );
    }
    
    float playheadX = trackStart.x + (m_currentTime / m_duration) * trackWidth;
    drawList->AddLine(
        ImVec2(playheadX, trackStart.y - 5.0f),
        ImVec2(playheadX, trackEnd.y + 5.0f),
        IM_COL32(255, 100, 100, 255),
        2.0f
    );
    
    drawList->AddTriangleFilled(
        ImVec2(playheadX - 5.0f, trackStart.y - 10.0f),
        ImVec2(playheadX + 5.0f, trackStart.y - 10.0f),
        ImVec2(playheadX, trackStart.y - 2.0f),
        IM_COL32(255, 100, 100, 255)
    );
    
    ImGui::Dummy(ImVec2(trackWidth, trackHeight + 15.0f));
    
    if (ImGui::IsMouseClicked(0) && ImGui::IsItemHovered()) {
        m_isDragging = true;
    }
    
    if (m_isDragging && !ImGui::IsMouseDown(0)) {
        m_isDragging = false;
    }
    
    if (m_isDragging) {
        ImVec2 mousePos = ImGui::GetMousePos();
        float relX = mousePos.x - trackStart.x;
        m_currentTime = (relX / trackWidth) * m_duration;
        m_currentTime = fmaxf(0.0f, fminf(m_duration, m_currentTime));
    }
}

void Timeline::renderRecordingControls()
{
    ImGui::BeginGroup();
    
    ImGui::PushStyleColor(ImGuiCol_Button, m_isRecording ? IM_COL32(200, 50, 50, 255) : IM_COL32(100, 100, 100, 255));
    if (ImGui::Button("REC", ImVec2(60, 24))) {
        m_isRecording = !m_isRecording;
    }
    ImGui::PopStyleColor();
    
    ImGui::SameLine();
    
    ImGui::Text("Duration:");
    ImGui::SameLine();
    if (ImGui::InputFloat("##duration", &m_duration, 1.0f, 5.0f, "%.1f")) {
        m_duration = fmaxf(0.1f, m_duration);
    }
    
    ImGui::SameLine();
    
    ImGui::Text("FPS:");
    ImGui::SameLine();
    if (ImGui::InputFloat("##fps", &m_fps, 1.0f, 5.0f, "%.0f")) {
        m_fps = fmaxf(1.0f, m_fps);
    }
    
    ImGui::SameLine();
    
    ImGui::Text("Export:");
    ImGui::SameLine();
    ImGui::InputText("##filename", m_exportFilename, sizeof(m_exportFilename));
    
    ImGui::SameLine();
    
    if (ImGui::Button("Export Alembic")) {
        m_exportRequested = true;
    }
    
    ImGui::EndGroup();
}

void Timeline::reset()
{
    m_currentTime = 0.0f;
    m_isPlaying = false;
    m_isRecording = false;
}
