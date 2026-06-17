#pragma once

#define HAIR_PHYSICS_VERSION_MAJOR 1
#define HAIR_PHYSICS_VERSION_MINOR 0
#define HAIR_PHYSICS_VERSION_PATCH 0

#define HAIR_PHYSICS_VERSION "1.0.0"

namespace HairPhysics {

static const int MAX_GUIDE_CURVES = 500;
static const int MAX_RENDER_STRANDS = 500000;
static const int MAX_POINTS_PER_STRAND = 100;
static const int MAX_VORTEX_SOURCES = 5;
static const int MAX_COLLIDERS = 20;

static const int DEFAULT_GUIDE_CURVES = 300;
static const int DEFAULT_RENDER_STRANDS = 100000;
static const int DEFAULT_POINTS_PER_STRAND = 20;

static const float DEFAULT_STIFFNESS = 0.85f;
static const float DEFAULT_BEND_STIFFNESS = 3.0f;
static const float DEFAULT_TWIST_STIFFNESS = 1.0f;
static const float DEFAULT_DAMPING = 0.05f;
static const float DEFAULT_GRAVITY = 9.81f;
static const float DEFAULT_MASS = 0.01f;

static const float DEFAULT_WIND_STRENGTH = 2.0f;
static const float DEFAULT_TURBULENCE = 1.5f;

static const float DEFAULT_COLLISION_RADIUS = 0.01f;
static const float DEFAULT_FRICTION = 0.3f;

static const float DEFAULT_STRAND_LENGTH = 0.5f;
static const float DEFAULT_HAIR_THICKNESS = 0.003f;
static const float DEFAULT_SIMULATION_SCALE = 1.0f;

static const float DEFAULT_TIMESTEP = 1.0f / 60.0f;
static const int DEFAULT_SUBSTEPS = 2;

static const float DEFAULT_FPS = 30.0f;
static const float DEFAULT_DURATION = 10.0f;

enum class SimulationMode {
    Realtime,
    Record,
    Bake
};

enum class WindType {
    Directional,
    Vortex,
    Turbulence
};

enum class CollisionType {
    Capsule,
    Sphere,
    Plane,
    Mesh
};

}
