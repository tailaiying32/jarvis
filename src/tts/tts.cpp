#include "tts.h"
#include <iostream>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>

#include "sherpa-onnx/c-api/c-api.h"

bool TextToSpeech::init(const std::string& model_path, const std::string& tokens_path, const std::string& data_dir) {
    SherpaOnnxOfflineTtsConfig config;
    memset(&config, 0, sizeof(config));

    // Piper/VITS model configuration
    config.model.vits.model    = model_path.c_str();
    config.model.vits.tokens   = tokens_path.c_str();
    config.model.vits.data_dir = data_dir.c_str();
    config.model.vits.length_scale = 1.0f;
    config.model.num_threads = 4;
    config.model.provider = "cpu";

    tts_ = SherpaOnnxCreateOfflineTts(&config);
    if (!tts_) {
        std::cerr << "Failed to create TTS engine\n";
        return false;
    }

    sample_rate_ = SherpaOnnxOfflineTtsSampleRate(tts_);
    return true;
}


// helper function to clean response of emojis and non-valid characters
static std::string sanitizeForTTS(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = text[i];
        if (c < 0x80) {
            result += c;
            ++i;
        }
        else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < text.size()) {
                result += text.substr(i, 2);
            }
            i += 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            i += 3;
        }
        else if ((c & 0xF8) == 0xF0) {
            i += 4;
        }
        else {
            ++i;
        }
    }

    return result;
}


void TextToSpeech::speak(const std::string& text, float speed) {
    if (!tts_) {
        std::cerr << "TTS not initialized\n";
        return;
    }

    // Sanitize text - remove emojis and problematic unicode
    std::string cleanText = sanitizeForTTS(text);
    // if the text is empty, just return
    if (text.empty()) { return; }

    // generate audio
    const SherpaOnnxGeneratedAudio* audio = SherpaOnnxOfflineTtsGenerate(tts_, text.c_str(), /*sid=*/0, speed);

    if (!audio || audio->n == 0) {
        std::cerr << "Failed to generate audio\n";
        if (audio) {
            SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        }
        return;
    }

    // play the audio
    playAudio(audio->samples, audio->n, audio->sample_rate);

    // cleanup
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
}

// playback state passed to callback
struct PlaybackData {
    const int16_t* samples;
    int32_t totalFrames;
    std::atomic<int32_t> currentFrame;
};

// miniaudio callback for playback
static void playbackCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    (void)pInput;
    PlaybackData* data = static_cast<PlaybackData*>(pDevice->pUserData);
    int16_t* output = static_cast<int16_t*>(pOutput);

    int32_t current = data->currentFrame.load();
    int32_t framesRemaining = data->totalFrames - current;
    ma_uint32 framesToCopy = (static_cast<int32_t>(frameCount) < framesRemaining)
                              ? frameCount
                              : static_cast<ma_uint32>(framesRemaining);

    if (framesToCopy > 0) {
        memcpy(output, data->samples + current, framesToCopy * sizeof(int16_t));
        data->currentFrame.fetch_add(static_cast<int32_t>(framesToCopy));
    }

    // Fill remaining with silence
    if (framesToCopy < frameCount) {
        memset(output + framesToCopy, 0, (frameCount - framesToCopy) * sizeof(int16_t));
    }
}

void TextToSpeech::playAudio(const float* samples, int32_t n, int32_t sample_rate) {
    // convert float [-1,1] to int16 for playback
    std::vector<int16_t> pcm(n);
    for (int32_t i = 0; i < n; i++) {
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        pcm[i] = static_cast<int16_t>(sample * 32767.0f);
    }

    // playback state
    PlaybackData playbackData;
    playbackData.samples = pcm.data();
    playbackData.totalFrames = n;
    playbackData.currentFrame.store(0);

    // set up playback device with callback
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = ma_format_s16;
    deviceConfig.playback.channels = 1;
    deviceConfig.sampleRate        = static_cast<ma_uint32>(sample_rate);
    deviceConfig.pUserData         = &playbackData;
    deviceConfig.dataCallback      = playbackCallback;

    ma_device device;
    if (ma_device_init(nullptr, &deviceConfig, &device) != MA_SUCCESS) {
        std::cerr << "Failed to initialize playback device\n";
        return;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Failed to start playback device\n";
        ma_device_uninit(&device);
        return;
    }

    // wait until playback is done
    while (playbackData.currentFrame.load() < playbackData.totalFrames) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // small delay to let final buffer drain
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ma_device_stop(&device);
    ma_device_uninit(&device);
}

void TextToSpeech::startStreaming() {
    done_ = false;
    streaming_ = true;
    workerThread_ = std::thread(&TextToSpeech::workerLoop, this);
}

void TextToSpeech::queueText(const std::string& text) {
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        textQueue_.push(text.c_str());
    }
    queueCv_.notify_one();
}

void TextToSpeech::finishStreaming() {
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        done_ = true;
    }
    queueCv_.notify_one();

    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    streaming_ = false;
}


void TextToSpeech::workerLoop() {
    while (true) {
        std::string text;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCv_.wait(lock, [this] {
                return !textQueue_.empty() || done_;
            });

            if (textQueue_.empty() && done_) {
                break;  // no more text incoming
            }

            if (!textQueue_.empty()) {
                text = textQueue_.front();
                textQueue_.pop();
            }
        }

        if (!text.empty()) {
            // generate and play this chunk
            speak(text, 1.2);
        }
    }
}

void TextToSpeech::shutdown() {
    if (tts_) {
        SherpaOnnxDestroyOfflineTts(tts_);
        tts_ = nullptr;
    }
}
