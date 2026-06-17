#include "common/shared_memory.h"
#include <cstring>
#include <iostream>

SharedMemoryManager::SharedMemoryManager()
    : m_isValid(false)
    , m_isCreator(false)
    , m_header(nullptr)
    , m_data(nullptr)
    , m_size(0)
#ifdef _WIN32
    , m_hMapFile(NULL)
#else
    , m_fd(-1)
#endif
{
}

SharedMemoryManager::~SharedMemoryManager()
{
    close();
}

bool SharedMemoryManager::create(const std::string& name)
{
    m_name = name;
    m_size = SHARED_MEMORY_SIZE;
    
#ifdef _WIN32
    m_hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        (DWORD)m_size,
        name.c_str()
    );
    
    if (m_hMapFile == NULL) {
        std::cerr << "Failed to create shared memory: " << GetLastError() << std::endl;
        return false;
    }
    
    m_data = MapViewOfFile(
        m_hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        m_size
    );
    
    if (m_data == nullptr) {
        std::cerr << "Failed to map shared memory view" << std::endl;
        CloseHandle(m_hMapFile);
        m_hMapFile = NULL;
        return false;
    }
#else
    m_fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (m_fd == -1) {
        std::cerr << "Failed to create shared memory" << std::endl;
        return false;
    }
    
    if (ftruncate(m_fd, m_size) == -1) {
        std::cerr << "Failed to set shared memory size" << std::endl;
        close();
        return false;
    }
    
    m_data = mmap(NULL, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
    if (m_data == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        close();
        return false;
    }
#endif
    
    m_header = static_cast<SharedMemoryHeader*>(m_data);
    std::memset(m_data, 0, m_size);
    
    m_header->magic = 0x48414952;
    m_header->version = 1;
    m_header->dataSize = m_size - sizeof(SharedMemoryHeader);
    m_header->isNewFrame = false;
    m_header->isPlaying = false;
    m_header->isRecording = false;
    
    m_header->guidePointsOffset = sizeof(SharedMemoryHeader);
    m_header->renderPointsOffset = m_header->guidePointsOffset;
    m_header->collidersOffset = m_header->renderPointsOffset;
    
    m_isCreator = true;
    m_isValid = true;
    
    return true;
}

bool SharedMemoryManager::open(const std::string& name)
{
    m_name = name;
    m_size = SHARED_MEMORY_SIZE;
    
#ifdef _WIN32
    m_hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        name.c_str()
    );
    
    if (m_hMapFile == NULL) {
        return false;
    }
    
    m_data = MapViewOfFile(
        m_hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        m_size
    );
    
    if (m_data == nullptr) {
        CloseHandle(m_hMapFile);
        m_hMapFile = NULL;
        return false;
    }
#else
    m_fd = shm_open(name.c_str(), O_RDWR, 0666);
    if (m_fd == -1) {
        return false;
    }
    
    m_data = mmap(NULL, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
    if (m_data == MAP_FAILED) {
        close();
        return false;
    }
#endif
    
    m_header = static_cast<SharedMemoryHeader*>(m_data);
    m_isCreator = false;
    m_isValid = true;
    
    return true;
}

void SharedMemoryManager::close()
{
    if (!m_isValid) return;
    
#ifdef _WIN32
    if (m_data) {
        UnmapViewOfFile(m_data);
        m_data = nullptr;
    }
    if (m_hMapFile) {
        CloseHandle(m_hMapFile);
        m_hMapFile = NULL;
    }
#else
    if (m_data && m_data != MAP_FAILED) {
        munmap(m_data, m_size);
        m_data = nullptr;
    }
    if (m_fd != -1) {
        if (m_isCreator) {
            shm_unlink(m_name.c_str());
        }
        close(m_fd);
        m_fd = -1;
    }
#endif
    
    m_header = nullptr;
    m_isValid = false;
}

void SharedMemoryManager::writeFrameData(const HairData& hairData, int frameNumber, double timestamp)
{
    if (!m_isValid || !m_header) return;
    
    const HairParams& params = hairData.getParams();
    m_header->params = params;
    m_header->frameNumber = frameNumber;
    m_header->timestamp = timestamp;
    
    m_header->numGuideCurves = hairData.getNumGuideCurves();
    m_header->pointsPerGuide = hairData.getPointsPerStrand();
    m_header->numRenderStrands = hairData.getNumRenderStrands();
    m_header->pointsPerStrand = hairData.getPointsPerStrand();
    m_header->numColliders = (int)hairData.getColliders().size();
    
    uint64_t offset = sizeof(SharedMemoryHeader);
    
    m_header->guidePointsOffset = offset;
    int guidePointCount = m_header->numGuideCurves * m_header->pointsPerGuide;
    glm::vec3* guidePoints = reinterpret_cast<glm::vec3*>(reinterpret_cast<char*>(m_data) + offset);
    
    const auto& guideCurves = hairData.getGuideCurves();
    for (int i = 0; i < m_header->numGuideCurves; ++i) {
        for (int j = 0; j < m_header->pointsPerGuide; ++j) {
            guidePoints[i * m_header->pointsPerGuide + j] = guideCurves[i].points[j];
        }
    }
    offset += guidePointCount * sizeof(glm::vec3);
    
    m_header->renderPointsOffset = offset;
    int renderPointCount = m_header->numRenderStrands * m_header->pointsPerStrand;
    glm::vec3* renderPoints = reinterpret_cast<glm::vec3*>(reinterpret_cast<char*>(m_data) + offset);
    
    const auto& renderStrands = hairData.getRenderStrands();
    for (int i = 0; i < m_header->numRenderStrands; ++i) {
        for (int j = 0; j < m_header->pointsPerStrand; ++j) {
            renderPoints[i * m_header->pointsPerStrand + j] = renderStrands[i].points[j];
        }
    }
    offset += renderPointCount * sizeof(glm::vec3);
    
    m_header->collidersOffset = offset;
    CapsuleCollider* colliders = reinterpret_cast<CapsuleCollider*>(reinterpret_cast<char*>(m_data) + offset);
    
    const auto& colliderList = hairData.getColliders();
    for (size_t i = 0; i < colliderList.size(); ++i) {
        colliders[i] = colliderList[i];
    }
    
    m_header->isNewFrame = true;
}

bool SharedMemoryManager::readFrameData(HairData& hairData, int& frameNumber, double& timestamp)
{
    if (!m_isValid || !m_header || !m_header->isNewFrame) return false;
    
    frameNumber = m_header->frameNumber;
    timestamp = m_header->timestamp;
    
    HairParams params = m_header->params;
    params.numGuideCurves = m_header->numGuideCurves;
    params.numRenderStrands = m_header->numRenderStrands;
    params.pointsPerStrand = m_header->pointsPerGuide;
    
    hairData.initialize(params);
    
    glm::vec3* guidePoints = reinterpret_cast<glm::vec3*>(
        reinterpret_cast<char*>(m_data) + m_header->guidePointsOffset);
    
    auto& guideCurves = hairData.getGuideCurves();
    for (int i = 0; i < m_header->numGuideCurves; ++i) {
        for (int j = 0; j < m_header->pointsPerGuide; ++j) {
            guideCurves[i].points[j] = guidePoints[i * m_header->pointsPerGuide + j];
        }
    }
    
    hairData.updateRenderStrandsFromGuides();
    
    CapsuleCollider* colliders = reinterpret_cast<CapsuleCollider*>(
        reinterpret_cast<char*>(m_data) + m_header->collidersOffset);
    
    auto& colliderList = hairData.getColliders();
    colliderList.clear();
    for (int i = 0; i < m_header->numColliders; ++i) {
        colliderList.push_back(colliders[i]);
    }
    
    return true;
}

void SharedMemoryManager::setPlaying(bool playing)
{
    if (m_isValid && m_header) {
        m_header->isPlaying = playing;
    }
}

void SharedMemoryManager::setRecording(bool recording)
{
    if (m_isValid && m_header) {
        m_header->isRecording = recording;
    }
}

bool SharedMemoryManager::isNewFrameAvailable()
{
    if (!m_isValid || !m_header) return false;
    return m_header->isNewFrame;
}

void SharedMemoryManager::markFrameRead()
{
    if (m_isValid && m_header) {
        m_header->isNewFrame = false;
    }
}
