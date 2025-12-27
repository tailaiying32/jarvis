#ifndef TEXT_INFERENCE_H
#define TEXT_INFERENCE_H

#include <string>
#include <llama.h>
#include <vector>
#include <functional>

using TokenCallback = std::function<void(const std::string& token)>;

class TextInference {

public:
    TextInference()
        : model_(nullptr),
          ctx_(nullptr),
          sampler_(nullptr),
          im_start_token_(-1),
          im_end_token_(-1),
          n_past_(0) {}

    ~TextInference() {
        shutdown();
    }

    bool init(const std::string& model_path, int gpu_layers);

    std::string generate(
        const std::string& prompt,
        int max_tokens,
        TokenCallback on_token,
        bool* hit_text_stop = nullptr
    );

    void appendToContext(const std::string& text);
    void clearHistory();
    bool isFirstTurn() const { return n_past_ == 0; }

    void shutdown();

private:
    llama_model* model_;
    llama_context* ctx_;
    llama_sampler* sampler_;

    // Qwen ChatML control tokens
    llama_token im_start_token_;
    llama_token im_end_token_;

    int n_past_;
};

#endif
