#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include "miniaudio.h"
#include <mutex>
#include <condition_variable>
#include <atomic>

class AudioCapture {
public:
    // constructor and destructor
    AudioCapture();
    ~AudioCapture();

    // initialize and start audio capture
    bool start();

    // stop capture
    void stop();

    // get available samples and returns number of frames read
    ma_uint32 readSamples(float* outputBuffer, ma_uint32 maxFrames);

    // check if audio is available
    ma_uint32 availableFrames();

    // condition variable and mutex for synchronization
    std::condition_variable audioAvailable;
    std::mutex audioMutex;


private:
    ma_device device{};
    ma_pcm_rb rb{};
    std::atomic<bool> isRunning{ false };

    // config
    static constexpr ma_uint32 CHANNELS = 1;        // Mono for Whisper
    static constexpr ma_uint32 SAMPLE_RATE = 16000; // 16kHz for Whisper
    static constexpr ma_uint32 BUFFER_SECONDS = 5;

    // static callback
    static void dataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);
};

#endif // AUDIO_CAPTURE_H