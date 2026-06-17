#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>

#include "common/hair_data.h"
#include "common/shared_memory.h"
#include "backend/cuda_solver.h"
#include "backend/alembic_exporter.h"
#include "frontend/renderer.h"
#include "frontend/viewport.h"
#include "frontend/parameter_panel.h"
#include "frontend/timeline.h"

static const int WINDOW_WIDTH = 1280;
static const int WINDOW_HEIGHT = 720;
static const int PARAM_PANEL_WIDTH = 320;
static const int TIMELINE_HEIGHT = 140;

static HairData g_hairData;
static CudaHairSolver g_cudaSolver;
static HairRenderer g_renderer;
static Viewport g_viewport;
static ParameterPanel g_paramPanel;
static Timeline g_timeline;
static SharedMemoryManager g_sharedMemory;
static AlembicExporter g_exporter;

static bool g_showGuideCurves = false;
static bool g_showColliders = false;
static bool g_useSharedMemory = false;
static bool g_initialized = false;

static double g_lastFrameTime = 0.0;
static float g_deltaTime = 0.0f;
static float g_fps = 0.0f;
static int g_frameCount = 0;
static float g_fpsTimer = 0.0f;

static void glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    int viewportHeight = WINDOW_HEIGHT - TIMELINE_HEIGHT;
    int viewportWidth = WINDOW_WIDTH - PARAM_PANEL_WIDTH;
    
    if (x < viewportWidth && y < viewportHeight) {
        if (action == GLFW_PRESS) {
            g_viewport.onMouseDown(button, (float)x, (float)y);
        } else if (action == GLFW_RELEASE) {
            g_viewport.onMouseUp(button, (float)x, (float)y);
        }
    }
}

static void cursorPositionCallback(GLFWwindow* window, double x, double y) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    int viewportHeight = WINDOW_HEIGHT - TIMELINE_HEIGHT;
    int viewportWidth = WINDOW_WIDTH - PARAM_PANEL_WIDTH;
    
    if (x < viewportWidth && y < viewportHeight) {
        g_viewport.onMouseMove((float)x, (float)y);
    }
}

static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    
    int viewportHeight = WINDOW_HEIGHT - TIMELINE_HEIGHT;
    int viewportWidth = WINDOW_WIDTH - PARAM_PANEL_WIDTH;
    
    if (x < viewportWidth && y < viewportHeight) {
        g_viewport.onMouseWheel((float)yoffset * 10.0f);
    }
}

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void initializeHairSystem() {
    HairParams params;
    params.numGuideCurves = 300;
    params.numRenderStrands = 100000;
    params.pointsPerStrand = 20;
    
    params.stiffness = 0.85f;
    params.bendStiffness = 3.0f;
    params.twistStiffness = 1.0f;
    params.damping = 0.05f;
    params.gravity = 9.81f;
    params.mass = 0.01f;
    
    params.windStrength = 2.0f;
    params.windDirection = glm::vec3(1.0f, 0.2f, 0.5f);
    params.turbulence = 1.5f;
    
    params.numVortexSources = 2;
    params.vortexPositions[0] = glm::vec3(0.5f, 1.0f, 0.0f);
    params.vortexStrengths[0] = 5.0f;
    params.vortexRadii[0] = 1.0f;
    params.vortexPositions[1] = glm::vec3(-0.5f, 1.5f, 0.5f);
    params.vortexStrengths[1] = 3.0f;
    params.vortexRadii[1] = 0.8f;
    
    params.collisionRadius = 0.01f;
    params.friction = 0.3f;
    
    params.strandLength = 0.5f;
    params.hairThickness = 0.003f;
    params.hairColor = glm::vec3(0.6f, 0.45f, 0.3f);
    
    params.simulationScale = 1.0f;
    params.timeStep = 1.0f / 60.0f;
    params.substeps = 2;
    
    g_hairData.initialize(params);
    
    CapsuleCollider headCollider;
    headCollider.start = glm::vec3(0.0f, 0.2f, 0.0f);
    headCollider.end = glm::vec3(0.0f, 0.6f, 0.0f);
    headCollider.radius = 0.3f;
    g_hairData.getColliders().push_back(headCollider);
    
    CapsuleCollider bodyCollider;
    bodyCollider.start = glm::vec3(0.0f, -0.5f, 0.0f);
    bodyCollider.end = glm::vec3(0.0f, 0.2f, 0.0f);
    bodyCollider.radius = 0.25f;
    g_hairData.getColliders().push_back(bodyCollider);
    
    g_cudaSolver.initialize(params);
    g_cudaSolver.uploadHairData(g_hairData);
    g_cudaSolver.setColliders(g_hairData.getColliders());
    
    g_initialized = true;
}

void resetSimulation() {
    g_hairData.reset();
    g_cudaSolver.uploadHairData(g_hairData);
    g_cudaSolver.reset();
    g_timeline.reset();
}

void exportAlembic(const std::string& filename) {
    std::cout << "Exporting Alembic: " << filename << std::endl;
    
    g_exporter.beginExport(filename, g_hairData.getParams());
    
    float exportTime = 0.0f;
    float exportDuration = g_timeline.getDuration();
    float fps = g_timeline.getFPS();
    float frameTime = 1.0f / fps;
    
    resetSimulation();
    
    while (exportTime < exportDuration) {
        g_cudaSolver.step(frameTime);
        g_cudaSolver.downloadHairData(g_hairData);
        
        g_exporter.addFrame(g_hairData, exportTime);
        
        exportTime += frameTime;
    }
    
    g_exporter.endExport();
    
    resetSimulation();
    
    std::cout << "Export complete: " << g_exporter.getNumFrames() << " frames" << std::endl;
}

void updateSimulation() {
    if (!g_initialized) return;
    
    if (g_timeline.isPlaying()) {
        g_cudaSolver.step(g_deltaTime);
        g_cudaSolver.downloadHairData(g_hairData);
    }
    
    if (g_timeline.isRecording()) {
        static float lastRecordTime = 0.0f;
        float recordInterval = 1.0f / g_timeline.getFPS();
        
        if (g_timeline.getCurrentTime() - lastRecordTime >= recordInterval) {
            lastRecordTime = g_timeline.getCurrentTime();
        }
    }
}

void render() {
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
    
    int viewportWidth = width - PARAM_PANEL_WIDTH;
    int viewportHeight = height - TIMELINE_HEIGHT;
    
    glViewport(0, TIMELINE_HEIGHT, viewportWidth, viewportHeight);
    
    g_renderer.setViewport(viewportWidth, viewportHeight);
    g_renderer.setViewMatrix(g_viewport.getViewMatrix());
    g_renderer.setProjectionMatrix(g_viewport.getProjectionMatrix());
    g_renderer.setShowGuideCurves(g_showGuideCurves);
    g_renderer.setShowColliders(g_showColliders);
    
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, TIMELINE_HEIGHT, viewportWidth, viewportHeight);
    g_renderer.render(g_hairData);
    glDisable(GL_SCISSOR_TEST);
}

void renderUI(int windowWidth, int windowHeight) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    int viewportWidth = windowWidth - PARAM_PANEL_WIDTH;
    int viewportHeight = windowHeight - TIMELINE_HEIGHT;
    
    g_paramPanel.setPosition((float)viewportWidth, 0.0f);
    g_paramPanel.setSize((float)PARAM_PANEL_WIDTH, (float)viewportHeight);
    
    HairParams params = g_hairData.getParams();
    g_paramPanel.render(params, g_showGuideCurves, g_showColliders);
    
    if (g_paramPanel.isParamsChanged()) {
        g_cudaSolver.updateParams(params);
        g_hairData.getParams() = params;
    }
    
    g_timeline.setPosition(0.0f, 0.0f);
    g_timeline.setSize((float)windowWidth, (float)TIMELINE_HEIGHT);
    
    ImGui::SetNextWindowPos(ImVec2(0, windowHeight - TIMELINE_HEIGHT));
    ImGui::SetNextWindowSize(ImVec2(windowWidth, TIMELINE_HEIGHT));
    
    g_timeline.render();
    
    if (g_timeline.isExportRequested()) {
        g_timeline.clearExportRequest();
        exportAlembic(g_timeline.getExportFilename());
    }
    
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("Stats", nullptr, 
        ImGuiWindowFlags_NoMove | 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("FPS: %.1f", g_fps);
    ImGui::Text("Frame time: %.2f ms", g_deltaTime * 1000.0f);
    ImGui::Text("Guide curves: %d", g_hairData.getNumGuideCurves());
    ImGui::Text("Render strands: %d", g_hairData.getNumRenderStrands());
    ImGui::End();
    
    ImGui::SetNextWindowPos(ImVec2(10, 90));
    ImGui::Begin("Controls", nullptr,
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_AlwaysAutoResize);
    
    if (ImGui::Button("Reset Simulation")) {
        resetSimulation();
    }
    
    ImGui::Checkbox("Shared Memory", &g_useSharedMemory);
    
    ImGui::End();
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main() {
    glfwSetErrorCallback(glfwErrorCallback);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        "Hair Physics System - GPU Accelerated",
        NULL, NULL
    );
    
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    
    glfwSwapInterval(0);
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");
    
    if (!g_renderer.initialize()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return -1;
    }
    
    g_viewport.setSize(WINDOW_WIDTH - PARAM_PANEL_WIDTH, WINDOW_HEIGHT - TIMELINE_HEIGHT);
    g_viewport.setDistance(3.0f);
    g_viewport.setTarget(glm::vec3(0.0f, 0.5f, 0.0f));
    
    initializeHairSystem();
    
    g_sharedMemory.create();
    
    g_lastFrameTime = glfwGetTime();
    
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        g_deltaTime = (float)(currentTime - g_lastFrameTime);
        g_lastFrameTime = currentTime;
        
        g_frameCount++;
        g_fpsTimer += g_deltaTime;
        if (g_fpsTimer >= 1.0f) {
            g_fps = (float)g_frameCount / g_fpsTimer;
            g_frameCount = 0;
            g_fpsTimer = 0.0f;
        }
        
        glfwPollEvents();
        
        updateSimulation();
        g_timeline.update(g_deltaTime);
        
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        
        glViewport(0, 0, width, height);
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        render();
        renderUI(width, height);
        
        if (g_useSharedMemory && g_initialized) {
            g_sharedMemory.writeFrameData(g_hairData, g_timeline.getCurrentFrame(), g_timeline.getCurrentTime());
            g_sharedMemory.setPlaying(g_timeline.isPlaying());
            g_sharedMemory.setRecording(g_timeline.isRecording());
        }
        
        glfwSwapBuffers(window);
    }
    
    g_sharedMemory.close();
    
    g_cudaSolver.shutdown();
    g_renderer.shutdown();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
