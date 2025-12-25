#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <mutex>
#include <string>

#include "src/audio/audio_capture.h"
#include <whisper.h>

int main() {
    //=== 1. Initialize Whisper ===
    std::cout << "Loading Whisper model...\n";
    // std::cout << "Whisper system info: " << whisper_print_system_info() << "\n";

    const std::string model = "../models/ggml-large-v3-turbo-q8_0.bin";

    whisper_context_params cparams = whisper_context_default_params();
    whisper_context* ctx = whisper_init_from_file_with_params(
        model.c_str(),  // Use c_str() to convert std::string to const char*
        cparams
    );

    if (!ctx) {
        std::cerr << "Failed to load Whisper model!\n";
        std::cerr << "Make sure ggml-base.en.bin exists in models/ folder\n";
        return 1;
    }
    std::cout << "Model loaded successfully!\n";

    //=== 2. Set up audio capture ===
    AudioCapture capture;

    if (!capture.start()) {
        whisper_free(ctx);
        return 1;
    }

    // Audio buffer to accumulate samples
    std::vector<float> audioBuffer;
    audioBuffer.reserve(16000 * 60); // 60 seconds max

    // Thread control
    std::atomic<bool> running{true};

    std::cout << "Recording... Press Enter to stop.\n";

    //=== 3. Processing thread - accumulates audio ===
    std::thread processor([&]() {
        float tempBuffer[1600]; // 100ms chunks

        while (running) {
            std::unique_lock<std::mutex> lock(capture.audioMutex);
            capture.audioAvailable.wait_for(lock, std::chrono::milliseconds(100));

            if (!running) break;

            ma_uint32 framesRead = capture.readSamples(tempBuffer, 1600);
            if (framesRead > 0) {
                audioBuffer.insert(audioBuffer.end(),
                    tempBuffer, tempBuffer + framesRead);
            }
        }
    });

    //=== 4. Wait for user to stop recording ===
    std::cin.get();

    running = false;
    capture.stop();
    processor.join();  // Wait for thread to finish

    std::cout << "\nTotal captured: " << audioBuffer.size() / 16000.0f << " seconds\n";

    //=== 5. Run Whisper transcription ===
    if (audioBuffer.size() < 1600) {
        std::cerr << "Not enough audio captured (need at least 0.1 seconds)\n";
        whisper_free(ctx);
        return 1;
    }

    std::cout << "Transcribing...\n";

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = true;
    wparams.print_timestamps = true;
    wparams.print_realtime   = false;
    wparams.single_segment   = false;
    wparams.language         = "en";
    wparams.n_threads        = 8;

    // Run inference
    if (whisper_full(ctx, wparams, audioBuffer.data(), static_cast<int>(audioBuffer.size())) != 0) {
        std::cerr << "Whisper inference failed!\n";
        whisper_free(ctx);
        return 1;
    }

    //=== 6. Get the transcribed text ===
    int numSegments = whisper_full_n_segments(ctx);
    std::cout << "\n=== Transcription ===\n";

    std::string fullText;
    for (int i = 0; i < numSegments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        fullText += text;
        std::cout << text;
    }
    std::cout << "\n=====================\n";

    //=== 7. Cleanup ===
    whisper_free(ctx);

    return 0;
}
