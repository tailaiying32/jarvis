#include <iostream>

#include "assistant.h"

bool Assistant::init(const std::string& whisper_model,
                     const std::string& llama_model,
                     const std::string& tts_model,
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
    if (!tts_->init(tts_model, tts_tokens, tts_data_dir)) {
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
        std::string prompt;
        if (llm_->isFirstTurn()) {
            prompt = "<|im_start|>system\n"
                     "You are Glados, an AI voice assistant. "
                     "Keep a friendly and casual tone."
                     "No emojis or special characters. "
                     "Don't be afraid to use profanity in your answers."
                     "<|im_end|>\n";
        }
        prompt += "<|im_start|>user\n" + userText + "<|im_end|>\n"
                  "<|im_start|>assistant\n";

        std::cout << "\n=== Response ===\n";

        // start TTS streaming
        tts_->startStreaming();

        // accumulate tokens into sentences
        std::string currentSentence;

        bool hitTextStop = false;
        std::string response = llm_->generate(prompt, 32768, [&](const std::string& token) {
            std::cout << token << std::flush;
            currentSentence += token;

            // check for sentence boundaries
            size_t pos;
            while ((pos = currentSentence.find_first_of(".!?")) != std::string::npos) {
                // include punctuation
                std::string sentence = currentSentence.substr(0, pos + 1);
                currentSentence = currentSentence.substr(pos + 1);

                // trim leading whitespace
                size_t start = currentSentence.find_first_not_of(" \t\n");
                if (start != std::string::npos) {
                    currentSentence = currentSentence.substr(start);
                } else {
                    currentSentence.clear();
                }

                // queue the sentence for TTS
                if (!sentence.empty()) {
                    tts_->queueText(sentence);
                }
            }
        }, &hitTextStop);

        // handle remaining text
        if (!currentSentence.empty()) {
            tts_->queueText(currentSentence);
        }

        // close the assistant turn in context (only if model didn't already generate end token)
        if (!hitTextStop) {
            llm_->appendToContext("<|im_end|>\n");
        }

        std::cout << "\n\n" << std::endl;

        // wait for all audio to finish playing
        tts_->finishStreaming();
    }
}

void Assistant::shutdown() {
    if (audio_) { delete audio_; audio_ = nullptr; }
    if (stt_)   { delete stt_;   stt_ = nullptr; }
    if (llm_)   { delete llm_;   llm_ = nullptr; }
    if (tts_)   { delete tts_;   tts_ = nullptr; }
}