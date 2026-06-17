#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "common/hair_data.h"

class AlembicExporter {
public:
    AlembicExporter();
    ~AlembicExporter();
    
    bool beginExport(const std::string& filename, const HairParams& params);
    void addFrame(const HairData& hairData, float time);
    void endExport();
    
    bool isExporting() const { return m_isExporting; }
    
    int getNumFrames() const { return m_numFrames; }
    float getTotalTime() const { return m_totalTime; }
    
    void setExportGuideCurves(bool exportGuides) { m_exportGuideCurves = exportGuides; }
    void setExportRenderStrands(bool exportRender) { m_exportRenderStrands = exportRender; }
    
private:
    void createHairSchema();
    void writeCurves(const std::vector<GuideCurve>& curves, float time);
    void writeRenderStrands(const std::vector<RenderStrand>& strands, float time);
    
    bool m_isExporting;
    std::string m_filename;
    HairParams m_params;
    
    int m_numFrames;
    float m_totalTime;
    
    bool m_exportGuideCurves;
    bool m_exportRenderStrands;
    
    void* m_archive;
    void* m_topObject;
    void* m_hairObject;
    void* m_curvesSchema;
    void* m_renderSchema;
    
    std::vector<int> m_curveCounts;
    std::vector<float> m_curveWidths;
    std::vector<glm::vec3> m_curveUVs;
};
