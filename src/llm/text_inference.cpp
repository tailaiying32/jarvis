#include "text_inference.h"
#include <cstdio>

bool TextInference::init(const std::string& model_path, int gpu_layers) {

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;

    llama_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) return false;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.9f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return true;
}

std::string TextInference::generate(
    const std::string& prompt,
    int max_tokens,
    TokenCallback on_token,
    bool* hit_text_stop
) {
    if (hit_text_stop) *hit_text_stop = false;

    const llama_vocab* vocab = llama_model_get_vocab(model_);

    std::vector<llama_token> tokens(prompt.size() + 1);
    bool add_bos = (n_past_ == 0);

    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        prompt.size(),
        tokens.data(),
        tokens.size(),
        add_bos,
        false
    );

    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]   = tokens[i];
        batch.pos[i]     = n_past_ + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]  = (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;

    n_past_ += n_tokens;

    std::string result;
    std::string rolling;

    llama_token eos = llama_vocab_eos(vocab);

    for (int i = 0; i < max_tokens; i++) {

        if (llama_decode(ctx_, batch) != 0) {
            fprintf(stderr, "llama_decode failed\n");
            break;
        }

        llama_token token = llama_sampler_sample(sampler_, ctx_, -1);

        if (token == eos) {
            if (hit_text_stop) *hit_text_stop = true;
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(
            vocab,
            token,
            buf,
            sizeof(buf),
            0,
            true
        );

        if (n > 0) {
            std::string piece(buf, n);
            rolling += piece;

            // HARD STOP: ChatML / control tokens
            if (rolling.find("<|") != std::string::npos) {
                if (hit_text_stop) *hit_text_stop = true;
                break;
            }

            // Keep rolling buffer short
            if (rolling.size() > 32)
                rolling.erase(0, rolling.size() - 32);

            result += piece;
            if (on_token) on_token(piece);
        }

        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = n_past_;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        n_past_++;
    }

    llama_batch_free(batch);
    return result;
}

void TextInference::appendToContext(const std::string& text) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(text.size() + 1);

    int n = llama_tokenize(
        vocab,
        text.c_str(),
        text.size(),
        tokens.data(),
        tokens.size(),
        false,
        false
    );

    tokens.resize(n);

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = n_past_ + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = n;

    llama_decode(ctx_, batch);
    n_past_ += n;
    llama_batch_free(batch);
}

void TextInference::clearHistory() {
    llama_memory_clear(llama_get_memory(ctx_), true);
    n_past_ = 0;
}

void TextInference::shutdown() {
    if (sampler_) { llama_sampler_free(sampler_); sampler_ = nullptr; }
    if (ctx_)     { llama_free(ctx_); ctx_ = nullptr; }
    if (model_)   { llama_model_free(model_); model_ = nullptr; }
}
