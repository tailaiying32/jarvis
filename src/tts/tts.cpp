#include "tts.h"
#include <iostream>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>

bool TextToSpeech::init(const std::string& model_path, const std::string& voices_path, const std::string& tokens_path, const std::string& data_dir) {
    SherpaOnnxOfflineTtsConfig config;
    memset(&config, 0, sizeof(config));

    // Kokoro model configuration
    config.model.kokoro.model    = model_path.c_str();
    config.model.kokoro.voices   = voices_path.c_str();
    config.model.kokoro.tokens   = tokens_path.c_str();
    config.model.kokoro.data_dir = data_dir.c_str();
    config.model.kokoro.length_scale = 1.0f;
    config.model.kokoro.lang = "en";
    config.model.num_threads = 8;
    config.model.provider      = "cpu";

    tts_ = SherpaOnnxCreateOfflineTts(&config);
    if (!tts_) {
        std::cerr << "Failed to create TTS engine\n";
        return false;
    }

    sample_rate_ = SherpaOnnxOfflineTtsSampleRate(tts_);
    std::cout << "TTS initialized. Sample rate: " << sample_rate_ << " Hz\n";

    return true;
}

void TextToSpeech::speak(const std::string& text, float speed) {
    if (!tts_) {
        std::cerr << "TTS not initialized\n";
        return;
    }

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

void TextToSpeech::shutdown() {
    if (tts_) {
        SherpaOnnxDestroyOfflineTts(tts_);
        tts_ = nullptr;
    }
}
