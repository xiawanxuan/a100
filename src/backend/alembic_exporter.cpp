#include "backend/alembic_exporter.h"
#include <iostream>
#include <fstream>
#include <cstring>

#pragma pack(push, 1)
struct AlembicSimpleHeader {
    char magic[8];
    uint32_t version;
    uint32_t numStrands;
    uint32_t pointsPerStrand;
    uint32_t numFrames;
    float totalTime;
    float fps;
};

struct FrameHeader {
    float time;
    uint32_t pointCount;
};
#pragma pack(pop)

AlembicExporter::AlembicExporter()
    : m_isExporting(false)
    , m_numFrames(0)
    , m_totalTime(0.0f)
    , m_exportGuideCurves(false)
    , m_exportRenderStrands(true)
    , m_archive(nullptr)
    , m_topObject(nullptr)
    , m_hairObject(nullptr)
    , m_curvesSchema(nullptr)
    , m_renderSchema(nullptr)
{
}

AlembicExporter::~AlembicExporter()
{
    if (m_isExporting) {
        endExport();
    }
}

bool AlembicExporter::beginExport(const std::string& filename, const HairParams& params)
{
    if (m_isExporting) return false;
    
    m_filename = filename;
    m_params = params;
    m_numFrames = 0;
    m_totalTime = 0.0f;
    
    int totalStrands = m_exportRenderStrands ? params.numRenderStrands : params.numGuideCurves;
    int pointsPerStrand = params.pointsPerStrand;
    
    m_curveCounts.resize(totalStrands, pointsPerStrand);
    m_curveWidths.resize(totalStrands * pointsPerStrand, params.hairThickness * params.simulationScale);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for export: " << filename << std::endl;
        return false;
    }
    
    AlembicSimpleHeader header;
    std::memcpy(header.magic, "HAIRCACHE", 8);
    header.version = 1;
    header.numStrands = totalStrands;
    header.pointsPerStrand = pointsPerStrand;
    header.numFrames = 0;
    header.totalTime = 0.0f;
    header.fps = 30.0f;
    
    file.write(reinterpret_cast<char*>(&header), sizeof(header));
    file.close();
    
    m_isExporting = true;
    return true;
}

void AlembicExporter::addFrame(const HairData& hairData, float time)
{
    if (!m_isExporting) return;
    
    std::ofstream file(m_filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) return;
    
    const std::vector<RenderStrand>& strands = hairData.getRenderStrands();
    int totalStrands = strands.size();
    int pointsPerStrand = hairData.getPointsPerStrand();
    int totalPoints = totalStrands * pointsPerStrand;
    
    FrameHeader frameHeader;
    frameHeader.time = time;
    frameHeader.pointCount = totalPoints;
    
    file.write(reinterpret_cast<char*>(&frameHeader), sizeof(frameHeader));
    
    std::vector<glm::vec3> positions(totalPoints);
    std::vector<glm::vec3> colors(totalPoints);
    
    for (int i = 0; i < totalStrands; i++) {
        for (int j = 0; j < pointsPerStrand; j++) {
            int idx = i * pointsPerStrand + j;
            positions[idx] = strands[i].points[j];
            colors[idx] = strands[i].color;
        }
    }
    
    file.write(reinterpret_cast<char*>(positions.data()), totalPoints * sizeof(glm::vec3));
    file.write(reinterpret_cast<char*>(colors.data()), totalPoints * sizeof(glm::vec3));
    
    file.close();
    
    m_numFrames++;
    m_totalTime = time;
    
    std::ofstream headerUpdate(m_filename, std::ios::binary | std::ios::in);
    if (headerUpdate.is_open()) {
        AlembicSimpleHeader header;
        headerUpdate.read(reinterpret_cast<char*>(&header), sizeof(header));
        header.numFrames = m_numFrames;
        header.totalTime = m_totalTime;
        headerUpdate.seekp(0);
        headerUpdate.write(reinterpret_cast<char*>(&header), sizeof(header));
        headerUpdate.close();
    }
}

void AlembicExporter::endExport()
{
    if (!m_isExporting) return;
    
    m_isExporting = false;
    
    std::cout << "Alembic export complete: " << m_filename << std::endl;
    std::cout << "  Frames: " << m_numFrames << std::endl;
    std::cout << "  Total time: " << m_totalTime << "s" << std::endl;
}

void AlembicExporter::createHairSchema()
{
}

void AlembicExporter::writeCurves(const std::vector<GuideCurve>& curves, float time)
{
}

void AlembicExporter::writeRenderStrands(const std::vector<RenderStrand>& strands, float time)
{
}
