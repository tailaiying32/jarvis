#ifndef TTS_H
#define TTS_H

#include <atomic>
#include <condition_variable>
#include <string>
#include <vector>
#include <cstdint>
#include <queue>
#include <thread>
#include <mutex>

#include "sherpa-onnx/c-api/c-api.h"
#include "miniaudio.h"

enum class TTSEngine {
    Piper,
    Kokoro
};

struct TTSConfig {
    TTSEngine engine;
    std::string model_path;
    std::string tokens_path;
    std::string data_dir;
    std::string voices_path;  // Kokoro only
};

// Pre-generated audio buffer
struct AudioBuffer {
    std::vector<int16_t> samples;
    int32_t sample_rate;
};

class TextToSpeech {
public:
    TextToSpeech() : tts_(nullptr), sample_rate_(0), deviceInitialized_(false) {}
    ~TextToSpeech() { shutdown(); }

    bool init(const TTSConfig& config);

    // blocking: generates and plays audio, returns when done
    void speak(const std::string& text, float speed = 1.0f);

    // streaming for real-time audio generation
    void startStreaming();
    void queueText(const std::string& text);
    void finishStreaming();

    void shutdown();

private:
    const SherpaOnnxOfflineTts* tts_;
    int32_t sample_rate_;

    void playAudio(const float* samples, int32_t n, int32_t sample_rate);
    AudioBuffer generateAudio(const std::string& text, float speed);

    // Text queue (input)
    std::queue<std::string> textQueue_;
    std::mutex textMutex_;
    std::condition_variable textCv_;

    // Audio queue (pre-generated, ready to play)
    std::queue<AudioBuffer> audioQueue_;
    std::mutex audioMutex_;
    std::condition_variable audioCv_;

    // Persistent audio device
    ma_device device_;
    bool deviceInitialized_;

    // Current playback state (accessed by callback)
    std::vector<int16_t> currentBuffer_;
    std::atomic<size_t> playbackPos_{0};
    std::mutex playbackMutex_;

    // Threads
    std::thread generatorThread_;

    std::atomic<bool> streaming_{false};
    std::atomic<bool> textDone_{false};
    std::atomic<bool> allDone_{false};

    void generatorLoop();
    static void audioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);
    void fillAudioBuffer(int16_t* output, ma_uint32 frameCount);
};

#endif
