#include <string>

#include "src/pipeline/assistant.h"

int main() {
    Assistant jarvis;

    // Model paths
    const std::string whisper_model = "models/ggml-large-v3-turbo-q8_0.bin";
    const std::string llama_model   = "models/Qwen3-VL-4B-Instruct-Q4_1.gguf";

    // TTS model paths (Piper)
    const std::string tts_model    = "models/vits-piper-en_US-glados/en_US-glados.onnx";
    const std::string tts_tokens   = "models/vits-piper-en_US-glados/tokens.txt";
    const std::string tts_data_dir = "models/vits-piper-en_US-glados/espeak-ng-data";

    if (!jarvis.init(whisper_model, llama_model, tts_model, tts_tokens, tts_data_dir)) {
        return 1;
    }

    jarvis.run();
    jarvis.shutdown();
    return 0;
}
