#pragma once

#include <string>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "common/hair_data.h"

#define SHARED_MEMORY_NAME "HairPhysicsSharedMem"
#define SHARED_MEMORY_SIZE (64 * 1024 * 1024)

struct SharedMemoryHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t dataSize;
    uint32_t frameNumber;
    double   timestamp;
    bool     isNewFrame;
    bool     isPlaying;
    bool     isRecording;
    uint32_t padding;
    
    HairParams params;
    int numGuideCurves;
    int pointsPerGuide;
    int numRenderStrands;
    int pointsPerStrand;
    
    uint64_t guidePointsOffset;
    uint64_t renderPointsOffset;
    uint64_t collidersOffset;
    int numColliders;
};

class SharedMemoryManager {
public:
    SharedMemoryManager();
    ~SharedMemoryManager();
    
    bool create(const std::string& name = SHARED_MEMORY_NAME);
    bool open(const std::string& name = SHARED_MEMORY_NAME);
    void close();
    
    bool isValid() const { return m_isValid; }
    
    SharedMemoryHeader* getHeader() { return m_header; }
    void* getData() { return m_data; }
    
    void writeFrameData(const HairData& hairData, int frameNumber, double timestamp);
    bool readFrameData(HairData& hairData, int& frameNumber, double& timestamp);
    
    void setPlaying(bool playing);
    void setRecording(bool recording);
    
    bool isNewFrameAvailable();
    void markFrameRead();

private:
    bool mapMemory(size_t size);
    void unmapMemory();
    
    std::string m_name;
    bool m_isValid;
    bool m_isCreator;
    
    SharedMemoryHeader* m_header;
    void* m_data;
    size_t m_size;
    
#ifdef _WIN32
    HANDLE m_hMapFile;
#else
    int m_fd;
#endif
};
