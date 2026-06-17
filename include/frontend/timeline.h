#pragma once

class Timeline {
public:
    Timeline();
    ~Timeline();
    
    void render();
    
    void setPosition(float x, float y) { m_x = x; m_y = y; }
    void setSize(float width, float height) { m_width = width; m_height = height; }
    
    float getCurrentTime() const { return m_currentTime; }
    void setCurrentTime(float time) { m_currentTime = time; }
    
    float getDuration() const { return m_duration; }
    void setDuration(float duration) { m_duration = duration; }
    
    float getFPS() const { return m_fps; }
    void setFPS(float fps) { m_fps = fps; }
    
    bool isPlaying() const { return m_isPlaying; }
    void setPlaying(bool playing) { m_isPlaying = playing; }
    
    bool isRecording() const { return m_isRecording; }
    void setRecording(bool recording) { m_isRecording = recording; }
    
    int getCurrentFrame() const { return (int)(m_currentTime * m_fps); }
    
    bool isExportRequested() const { return m_exportRequested; }
    void clearExportRequest() { m_exportRequested = false; }
    
    const char* getExportFilename() const { return m_exportFilename; }
    
    void update(float deltaTime);
    
    void reset();

private:
    void renderPlaybackControls();
    void renderTimelineTrack();
    void renderRecordingControls();
    
    float m_x;
    float m_y;
    float m_width;
    float m_height;
    
    float m_currentTime;
    float m_duration;
    float m_fps;
    
    bool m_isPlaying;
    bool m_isRecording;
    bool m_loop;
    
    bool m_exportRequested;
    char m_exportFilename[256];
    
    bool m_isDragging;
};
