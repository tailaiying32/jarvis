#include <iostream>

#include "assistant.h"

bool Assistant::init(const std::string& whisper_model,
                     const std::string& llama_model,
                     const std::string& tts_model,
                     const std::string& tts_voices,
                     const std::string& tts_tokens,
                     const std::string& tts_data_dir) {
    audio_ = new AudioCapture();
    stt_ = new Transcribe();
    llm_ = new TextInference();
    tts_ = new TextToSpeech();

    // Initialize speech-to-text
    if (!stt_->init(whisper_model)) {
        std::cerr << "Failed to initialize Whisper\n";
        return false;
    }

    // Initialize LLM
    if (!llm_->init(llama_model, 99)) {
        std::cerr << "Failed to initialize LLM\n";
        return false;
    }

    // Initialize TTS
    if (!tts_->init(tts_model, tts_voices, tts_tokens, tts_data_dir)) {
        std::cerr << "Failed to initialize TTS\n";
        return false;
    }

    return true;
}

void Assistant::run() {
    std::cout << "Jarvis ready. Press Enter to start/stop recording. Say 'quit' or 'exit' to end.\n\n";

    while (true) {
        // =============== START AUDIO DEVICE ===================
        if (!audio_->start()) {
            std::cerr << "Failed to start audio capture\n";
            return;
        }

        // audio buffer - clear for each turn
        std::vector<float> audioBuffer;
        audioBuffer.reserve(16000 * 60); // 60 seconds max

        // thread control
        std::atomic<bool> recording{true};

        std::cout << "[Recording... Press Enter to stop]\n";

        // processing thread to accumulate audio
        std::thread processor([&]() {
            float tempBuffer[1600]; // 100ms chunks

            while (recording) {
                std::unique_lock<std::mutex> lock(audio_->audioMutex);
                audio_->audioAvailable.wait_for(lock, std::chrono::milliseconds(100));

                if (!recording) break;

                ma_uint32 framesRead = audio_->readSamples(tempBuffer, 1600);
                if (framesRead > 0) {
                    audioBuffer.insert(audioBuffer.end(),
                        tempBuffer, tempBuffer + framesRead);
                }
            }
        });

        // wait for user to stop recording
        std::cin.get();

        recording = false;
        audio_->stop();
        processor.join();

        std::cout << "Captured: " << audioBuffer.size() / 16000.0f << " seconds\n";

        // ========== TRANSCRIBE ==================
        std::string userText = stt_->transcribe(audioBuffer);

        // Check for exit commands
        if (userText.find("quit") != std::string::npos ||
            userText.find("exit") != std::string::npos ||
            userText.find("goodbye") != std::string::npos) {
            std::cout << "Goodbye!\n";
            break;
        }

        // ========== GENERATE RESPONSE ==============
        // only send new content
        std::string prompt = "User: " + userText + "\nAssistant:";

        std::cout << "\n=== Response ===\n";
        std::string response = llm_->generate(prompt, 1024, [](const std::string& token) {
            std::cout << token << std::flush;
        });
        std::cout << "\n\n";

        // ========== SPEAK RESPONSE ==============
        tts_->speak(response, 1.0);
    }
}

void Assistant::shutdown() {
    if (audio_) { delete audio_; audio_ = nullptr; }
    if (stt_)   { delete stt_;   stt_ = nullptr; }
    if (llm_)   { delete llm_;   llm_ = nullptr; }
    if (tts_)   { delete tts_;   tts_ = nullptr; }
}