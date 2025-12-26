#ifndef PIPELINE_H
#define PIPELINE_H

#include <string>
#include "../audio/audio_capture.h"
#include "../transcribe/transcribe.h"
#include "../llm/text_inference.h"
#include "../tts/tts.h"

class Transcribe;

class Assistant {

public:
    Assistant(): audio_(nullptr), stt_(nullptr), llm_(nullptr), tts_(nullptr) {}
    ~Assistant() { shutdown(); }

    bool init(const std::string& whisper_model,
              const std::string& llama_model,
              const std::string& tts_model,
              const std::string& tts_voices,
              const std::string& tts_tokens,
              const std::string& tts_data_dir);
    void run();
    void shutdown();

private:
    AudioCapture* audio_;
    Transcribe* stt_;
    TextInference* llm_;
    TextToSpeech* tts_;
};

#endif