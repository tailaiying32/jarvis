#ifndef VAD_H
#define VAD_H

class VoiceActivityDetector {
public:
    // returns true if speech is detected in the audio frame
    bool isSpeech(const float* audioFrame, size_t frameSize);

    // calculate rms energy of the audio frame
    float calculateRMS(const float* audioFrame, size_t frameSize);

    // threshold for speech detection
    float speechThreshold = 0.01f;

    // track silence duration for end of speech detection
    int silenceDuration = 0;
    int silenceThreshold = 30; // number of frames to consider as silence
};

#endif