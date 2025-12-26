#ifndef TEXT_INFERENCE_H
#define TEXT_INFERENCE_H

#include <string>
#include <llama.h>
#include <vector>
#include <functional>

using TokenCallback = std::function<void(const std::string&token)>;

class TextInference {

public:
    TextInference(): model_(nullptr), ctx_(nullptr), sampler_(nullptr) {}
    ~TextInference() {
        shutdown();
    };

    bool init(const std::string& model_path, int gpu_layers);
    std::string generate(const std::string& prompt, int max_tokens, TokenCallback on_token);
    void clearHistory();

    void llama_batch_add(llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits);
    void llama_batch_clear(llama_batch & batch);

    void shutdown();

private:
    llama_model* model_;
    struct llama_context *ctx_;
    llama_sampler* sampler_;
    int n_past_ = 0;
};

#endif