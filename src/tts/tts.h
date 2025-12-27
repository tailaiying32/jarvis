#ifndef TTS_H
#define TTS_H

#include <atomic>
#include <condition_variable>
#include <string>
#include <vector>
#include <cstdint>
#include <queue>
#include <thread>

#include "sherpa-onnx/c-api/c-api.h"
#include "miniaudio.h"

class TextToSpeech {
public:
    TextToSpeech() : tts_(nullptr), sample_rate_(0) {}
    ~TextToSpeech() { shutdown(); }

    bool init(const std::string& model_path,
              const std::string& tokens_path,
              const std::string& data_dir);

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

    // streaming state
    std::queue<std::string> textQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCv_;
    std::thread workerThread_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> done_{false};

    void workerLoop();
};

#endif
