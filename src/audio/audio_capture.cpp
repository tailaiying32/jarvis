#include "audio_capture.h"

#include <iostream>
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

AudioCapture::AudioCapture() {
	// ring buffer initialized in start()
}

AudioCapture::~AudioCapture() {
	stop();
}

bool AudioCapture::start() {
	// Initialize ring buffer
	ma_uint32 rbSize = SAMPLE_RATE * BUFFER_SECONDS * sizeof(float);
	ma_result result = ma_pcm_rb_init(ma_format_f32, CHANNELS, rbSize, nullptr, nullptr, &rb);
	if (result != MA_SUCCESS) {
		std::cerr << "Failed to initialize ring buffer." << std::endl;
		return false;

	}
	// Configure device
	ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
	deviceConfig.capture.format   = ma_format_f32;
	deviceConfig.capture.channels = CHANNELS;
	deviceConfig.sampleRate       = SAMPLE_RATE;
	deviceConfig.dataCallback     = dataCallback;
	deviceConfig.pUserData        = this;

	// Initialize device
	if (ma_device_init(nullptr, &deviceConfig, &device) != MA_SUCCESS) {
		std::cerr << "Failed to initialize audio device." << std::endl;
		ma_device_uninit(&device);
		ma_pcm_rb_uninit(&rb);
		return false;
	}

	// Start device
	if (ma_device_start(&device) != MA_SUCCESS) {
		std::cerr << "Failed to start audio device." << std::endl;
		ma_device_uninit(&device);
		ma_pcm_rb_uninit(&rb);
		return false;
	}

	isRunning = true;
	std::cout << "Audio capture started.\n" << std::endl;
	return true;
}

void AudioCapture::stop() {
	if (isRunning) {
		ma_device_stop(&device);
		ma_device_uninit(&device);
		ma_pcm_rb_uninit(&rb);
		isRunning = false;
		audioAvailable.notify_all(); // wake up any waiting threads
		std::cout << "Audio capture stopped.\n" << std::endl;
	}
}

// STATIC callback - called by miniaudio from audio thread
void AudioCapture::dataCallback(ma_device* pDevice, void* pOutput,
	const void* pInput, ma_uint32 frameCount) {
	// Get 'this' pointer from user data
	AudioCapture* capture = static_cast<AudioCapture*>(pDevice->pUserData);

	if (pInput == nullptr) return;

	// Write incoming audio to ring buffer
	void* pWriteBuffer;
	ma_uint32 framesToWrite = frameCount;

	ma_pcm_rb_acquire_write(&capture->rb, &framesToWrite, &pWriteBuffer);

	if (framesToWrite > 0) {
		memcpy(pWriteBuffer, pInput,
			framesToWrite * ma_get_bytes_per_frame(ma_format_f32, CHANNELS));
		ma_pcm_rb_commit_write(&capture->rb, framesToWrite);
	}

	// Signal that new audio is available
	capture->audioAvailable.notify_one();

	(void)pOutput; // Unused for capture
}

ma_uint32 AudioCapture::readSamples(float* outputBuffer, ma_uint32 maxFrames) {
	void* pReadBuffer;
	ma_uint32 framesToRead = maxFrames;

	ma_pcm_rb_acquire_read(&rb, &framesToRead, &pReadBuffer);

	if (framesToRead > 0) {
		memcpy(outputBuffer, pReadBuffer,
			framesToRead * ma_get_bytes_per_frame(ma_format_f32, CHANNELS));
		ma_pcm_rb_commit_read(&rb, framesToRead);
	}

	return framesToRead;
}

ma_uint32 AudioCapture::availableFrames() {
	return ma_pcm_rb_available_read(&rb);
}
