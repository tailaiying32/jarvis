#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <mutex>
#include <string>

#include "transcribe.h"
#include <whisper.h>

bool Transcribe::init(const std::string &model_path) {
    #ifdef _WIN32
        system("chcp 65001");
    #endif
        //=== 1. Initialize Whisper ===
        std::cout << "Loading Whisper model...\n";

        std::cout << "Whisper system info: " << whisper_print_system_info() << "\n";

        const std::string& model = model_path;

        const whisper_context_params context_params = whisper_context_default_params();

        whisper_log_set([](enum ggml_log_level level, const char* text, void* user_data) {
            // Silent
        }, nullptr);
        ctx_ = whisper_init_from_file_with_params(
            model.c_str(),
            context_params
        );

        if (!ctx_) {
            std::cerr << "Failed to load Whisper model!\n";
            std::cerr << "Make sure ggml-base.en.bin exists in models/ folder\n";
            return false;
        }
        std::cout << "Model loaded successfully!\n";
        return true;
}

std::string Transcribe::transcribe(const std::vector<float> &audio) {
    if (audio.size() < 1600) {
        return "Not enough audio captured (need at least 0.1 seconds)\n";
    }

    std::cout << "Transcribing...\n";

    whisper_full_params full_params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    full_params.print_progress   = false;
    full_params.print_timestamps = false;
    full_params.print_realtime   = false;
    full_params.single_segment   = false;
    full_params.language         = "en";
    full_params.n_threads        = 8;
    full_params.vad = true;
    full_params.vad_model_path = "models/silero-v6.2.0-ggml.bin";

    // Run inference
    if (whisper_full(ctx_, full_params, audio.data(), static_cast<int>(audio.size())) != 0) {
        return "Whisper inference failed!\n";
    }

    // Get the transcribed text
    int numSegments = whisper_full_n_segments(ctx_);
    std::cout << "\n\n=== Transcription ===\n\n";

    std::string fullText;
    for (int i = 0; i < numSegments; i++) {
        const char* text = whisper_full_get_segment_text(ctx_, i);
        fullText += text;
        std::cout << text;
    }
    std::cout << "\n=====================\n";

    return fullText;
}

void Transcribe::shutdown() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}


