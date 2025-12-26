#ifndef TRANSCRIBE_H
#define TRANSCRIBE_H

#include <string>
#include <vector>
#include <whisper.h>

class Transcribe {

public:
    Transcribe() {
        ctx_ = nullptr;
    };
    ~Transcribe() {
        shutdown();
    }

    bool init(const std::string& model_path);
    std::string transcribe(const std::vector<float> &audio);
    void shutdown();

private:
    whisper_context*ctx_;

};

#endif
