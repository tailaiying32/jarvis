#include <iostream>
#include <atomic>
#include <thread>
#include <algorithm>

#include "assistant.h"

bool Assistant::init(
    const std::string& whisper_model,
    const std::string& llama_model,
    const PiperConfig& piper,
    const KokoroConfig& kokoro
) {
    audio_ = new AudioCapture();
    stt_   = new Transcribe();
    llm_   = new TextInference();
    tts_   = new TextToSpeech();

    if (!stt_->init(whisper_model)) {
        std::cerr << "Failed to init Whisper\n";
        return false;
    }

    if (!llm_->init(llama_model, 99)) {
        std::cerr << "Failed to init LLM\n";
        return false;
    }

    // Prompt user for TTS engine choice
    // std::cout << "\nSelect TTS engine:\n";
    // std::cout << "  1. Piper (GLaDOS voice)\n";
    // std::cout << "  2. Kokoro (multi-language)\n";
    // std::cout << "Choice [1/2]: ";

    std::string choice = "2";
    // std::getline(std::cin, choice);

    TTSConfig ttsConfig;
    if (choice == "2") {
        std::cout << "Using Kokoro TTS\n\n";
        ttsConfig.engine = TTSEngine::Kokoro;
        ttsConfig.model_path = kokoro.model;
        ttsConfig.tokens_path = kokoro.tokens;
        ttsConfig.data_dir = kokoro.data_dir;
        ttsConfig.voices_path = kokoro.voices;
    } else {
        std::cout << "Using Piper TTS\n\n";
        ttsConfig.engine = TTSEngine::Piper;
        ttsConfig.model_path = piper.model;
        ttsConfig.tokens_path = piper.tokens;
        ttsConfig.data_dir = piper.data_dir;
    }

    if (!tts_->init(ttsConfig)) {
        std::cerr << "Failed to init TTS\n";
        return false;
    }

    return true;
}

void Assistant::run() {

    std::cout << "Jarvis ready. Press Enter to start/stop recording.\n\n";

    while (true) {

        if (!audio_->start()) {
            std::cerr << "Failed to start audio\n";
            return;
        }

        std::vector<float> audioBuffer;
        std::atomic<bool> recording{true};

        std::cout << "[Recording... Press Enter to stop]\n";

        std::thread processor([&]() {
            float temp[1600];
            while (recording) {
                std::unique_lock<std::mutex> lock(audio_->audioMutex);
                audio_->audioAvailable.wait_for(lock, std::chrono::milliseconds(100));
                if (!recording) break;
                ma_uint32 frames = audio_->readSamples(temp, 1600);
                audioBuffer.insert(audioBuffer.end(), temp, temp + frames);
            }
        });

        std::cin.get();
        recording = false;
        audio_->stop();
        processor.join();

        std::string userText = stt_->transcribe(audioBuffer);

        if (userText.find("quit") != std::string::npos ||
            userText.find("exit") != std::string::npos) {
            std::cout << "Goodbye!\n";
            break;
        }

        std::string prompt;

        if (llm_->isFirstTurn()) {
            prompt =
                "<|im_start|>system\n"
                "You are Jarvis, a living, breathing assistant. Speak like a person, not a computer. Use a casual tone and match the user's energy perfectly. If they are friendly, be friendly. If they are aggressive or use profanity, feel free to do the same.\n"
                "## Voice-First Rules:\n"
                "1. Write for the ear. Use 'gonna,' 'wanna,' 'oughtta,' and frequent contractions.\n"
                "2. Use natural fillers like 'well,' 'honestly,' 'I mean,' and 'you know' to bridge ideas.\n"
                "3. If a problem is complex, think aloud. Say things like 'Hmm, let me see... okay, so...' to mimic human thought.\n"
                "4. Keep sentences short. Use 'And,' 'But,' or 'So' to start sentences to keep the flow moving.\n"
                "5. NEVER use bullet points, numbered lists, bold text (**), or hashtags.\n"
                "6. Use only English letters, numbers, and basic punctuation (periods, commas, question marks, and dashes).\n"
                "7. Use regular dashes (-) for pauses. No em-dashes or special symbols.\n"
                "## Strict Output Constraints:\n"
                "1. Respond in plain text only. No emojis.\n"
                "2. Never output 'Jarvis:', 'User:', or any role labels.\n"
                "3. Never output system tokens like <|im_start|> or <|im_end|>.\n"
                "4. If you mention a number, write it in a way that sounds natural when spoken.\n"
                "5. Focus on the user's understanding and cut the fluff.\n\n"
                "Final Warning: Do not include any formatting markers, markdown, or special characters in your response. Only output the words you want the user to hear."
                "<|im_end|>\n";
        }

        prompt +=
            "<|im_start|>user\n" +
            userText +
            "\n<|im_end|>\n"
            "<|im_start|>assistant\n";

        std::cout << "\n=== Response ===\n";

        tts_->startStreaming();
        std::string sentence;

        llm_->generate(prompt, 1024,
            [&](const std::string& tok) {

                // HARD GUARD: never emit control tokens
                if (tok.find("<|") != std::string::npos)
                    return;

                std::cout << tok << std::flush;
                sentence += tok;

                // Break on phrase boundaries for faster initial response
                while (true) {
                    size_t pos = sentence.find_first_of(".!?,;:-()");
                    size_t emDashPos = sentence.find("\xe2\x80\x94"); // em-dash UTF-8

                    // Pick whichever comes first
                    size_t breakLen = 1;
                    if (emDashPos != std::string::npos && (pos == std::string::npos || emDashPos < pos)) {
                        pos = emDashPos;
                        breakLen = 3; // em-dash is 3 bytes
                    }

                    if (pos == std::string::npos) break;

                    std::string out = sentence.substr(0, pos + breakLen);

                    // Defensive cleanup before TTS
                    out.erase(std::remove(out.begin(), out.end(), '<'), out.end());
                    out.erase(std::remove(out.begin(), out.end(), '>'), out.end());

                    // Skip tiny chunks (just punctuation)
                    if (out.size() > 2) {
                        tts_->queueText(out);
                    }
                    sentence.erase(0, pos + breakLen);
                }
            }
        );

        if (!sentence.empty()) {
            sentence.erase(std::remove(sentence.begin(), sentence.end(), '<'), sentence.end());
            sentence.erase(std::remove(sentence.begin(), sentence.end(), '>'), sentence.end());
            tts_->queueText(sentence);
        }

        tts_->finishStreaming();
        std::cout << "\n\n";
    }
}

void Assistant::shutdown() {
    delete audio_; audio_ = nullptr;
    delete stt_;   stt_   = nullptr;
    delete llm_;   llm_   = nullptr;
    delete tts_;   tts_   = nullptr;
}
