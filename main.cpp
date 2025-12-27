#include <string>

#include "src/pipeline/assistant.h"

int main() {
    Assistant jarvis;

    // Model paths
    const std::string whisper_model = "models/ggml-medium-q8_0.bin";
    const std::string llama_model   = "models/Qwen3-VL-4B-Instruct-Q4_1.gguf";

    // Piper TTS config
    PiperConfig piper;
    piper.model    = "models/vits-piper-en_US-glados/en_US-glados.onnx";
    piper.tokens   = "models/vits-piper-en_US-glados/tokens.txt";
    piper.data_dir = "models/vits-piper-en_US-glados/espeak-ng-data";

    // Kokoro TTS config
    KokoroConfig kokoro;
    kokoro.model    = "models/kokoro-int8-multi-lang-v1_0/model.int8.onnx";
    kokoro.tokens   = "models/kokoro-int8-multi-lang-v1_0/tokens.txt";
    kokoro.data_dir = "models/kokoro-int8-multi-lang-v1_0/espeak-ng-data";
    kokoro.voices   = "models/kokoro-int8-multi-lang-v1_0/voices.bin";

    if (!jarvis.init(whisper_model, llama_model, piper, kokoro)) {
        return 1;
    }

    jarvis.run();
    jarvis.shutdown();
    return 0;
}
