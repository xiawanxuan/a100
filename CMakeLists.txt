cmake_minimum_required(VERSION 3.20)
project(HairPhysicsSystem VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenGL REQUIRED)

add_subdirectory(thirdparty/glfw)
add_subdirectory(thirdparty/glew)
add_subdirectory(thirdparty/imgui)
add_subdirectory(thirdparty/glm)

set(COMMON_SOURCES
    src/common/hair_data.cpp
    src/common/shared_memory.cpp
)

set(BACKEND_SOURCES
    src/backend/kernel.cu
    src/backend/wind_field.cu
    src/backend/collision.cu
    src/backend/alembic_exporter.cpp
    src/backend/cuda_solver.cpp
)

set(FRONTEND_SOURCES
    src/frontend/renderer.cpp
    src/frontend/viewport.cpp
    src/frontend/parameter_panel.cpp
    src/frontend/timeline.cpp
)

set(MAIN_SOURCE src/main.cpp)

add_executable(HairPhysicsEditor
    ${MAIN_SOURCE}
    ${COMMON_SOURCES}
    ${BACKEND_SOURCES}
    ${FRONTEND_SOURCES}
)

target_include_directories(HairPhysicsEditor PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    include
    thirdparty/glfw/include
    thirdparty/glew/include
    thirdparty/imgui
    thirdparty/glm
    thirdparty/alembic/lib
    thirdparty/Imath/src
)

target_link_libraries(HairPhysicsEditor PRIVATE
    OpenGL::GL
    glfw
    libglew_static
    imgui
    glm
    ${CUDA_LIBRARIES}
    cudart
)

set_target_properties(HairPhysicsEditor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(WIN32)
    target_compile_definitions(HairPhysicsEditor PRIVATE WINDOWS_LEAN_AND_MEAN NOMINMAX)
endif()
