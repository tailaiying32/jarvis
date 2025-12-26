#ifndef TTS_H
#define TTS_H

#include <string>
#include <vector>
#include <cstdint>
#include "sherpa-onnx/c-api/c-api.h"
#include "miniaudio.h"

class TextToSpeech {
public:
    TextToSpeech() : tts_(nullptr), sample_rate_(0) {}
    ~TextToSpeech() { shutdown(); }

    bool init(const std::string& model_path,
              const std::string& voices_path,
              const std::string& tokens_path,
              const std::string& data_dir);

    // Blocking: generates and plays audio, returns when done
    void speak(const std::string& text, float speed = 1.0f);

    void shutdown();

private:
    const SherpaOnnxOfflineTts* tts_;
    int32_t sample_rate_;

    void playAudio(const float* samples, int32_t n, int32_t sample_rate);
};

#endif
