#include <string>

#include "src/pipeline/assistant.h"

int main() {
    Assistant jarvis;

    // Model paths
    const std::string whisper_model = "models/ggml-large-v3-turbo-q8_0.bin";
    const std::string llama_model   = "models/Qwen3-VL-4B-Instruct-Q4_1.gguf";

    // TTS model paths (Kokoro)
    const std::string tts_model    = "models/kokoro-int8-multi-lang-v1_1/model.int8.onnx";
    const std::string tts_voices   = "models/kokoro-int8-multi-lang-v1_1/voices.bin";
    const std::string tts_tokens   = "models/kokoro-int8-multi-lang-v1_1/tokens.txt";
    const std::string tts_data_dir = "models/kokoro-int8-multi-lang-v1_1/espeak-ng-data";

    if (!jarvis.init(whisper_model, llama_model, tts_model, tts_voices, tts_tokens, tts_data_dir)) {
        return 1;
    }

    jarvis.run();
    jarvis.shutdown();
    return 0;
}
