#include "text_inference.h"

#include <vector>
#include "llama.h"

bool TextInference::init(const std::string &model_path, const int gpu_layers) {
    // set model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;

    // load model from file
    llama_log_set([](enum ggml_log_level level, const char* text, void* user_data) {
        // Silent - do nothing
    }, nullptr);

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        return false;
    }

    // set context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;

    // initialize model
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    // initialize sampler chain with typical samplers
    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return true;
}

std::string TextInference::generate(const std::string& prompt, int max_tokens, TokenCallback on_token, bool* hit_text_stop) {
    if (hit_text_stop) *hit_text_stop = false;

    // tokenize prompt (don't add BOS if continuing conversation)
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens_list(prompt.size() + 1);
    bool add_bos = (n_past_ == 0);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), add_bos, false);
    tokens_list.resize(n_tokens);

    // prepare batch - positions start at n_past_
    llama_batch batch = llama_batch_init(std::max(n_tokens, 1), 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens_list[i], n_past_ + i, { 0 }, i == n_tokens - 1);
    }
    n_past_ += n_tokens;

    std::string result;
    std::string buffer;  // buffer to detect stop sequences before streaming
    const size_t BUFFER_SIZE = 12;  // length of "<|im_start|>"
    llama_token eos_token = llama_vocab_eos(vocab);
    llama_token eot_token = llama_vocab_eot(vocab);

    for (int i = 0; i < max_tokens; i++) {
        // evaluate the batch
        if (llama_decode(ctx_, batch) != 0) {
            fprintf(stderr, "Failed to evaluate\n");
            break;
        }

        // sample next token
        llama_token new_token = llama_sampler_sample(sampler_, ctx_, -1);

        // check for eos or eot (end-of-turn, used by ChatML models like Qwen)
        if (new_token == eos_token || new_token == eot_token) break;

        // convert output to string
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        std::string piece(buf, n);

        buffer.append(piece);

        // check for ChatML markers in buffer
        size_t stop_pos = buffer.find("<|im_end|>");
        if (stop_pos == std::string::npos) {
            stop_pos = buffer.find("<|im_start|>");
        }
        if (stop_pos != std::string::npos) {
            // emit everything before the stop marker
            if (stop_pos > 0 && on_token) {
                on_token(buffer.substr(0, stop_pos));
            }
            result.append(buffer.substr(0, stop_pos));
            if (hit_text_stop) *hit_text_stop = true;
            break;
        }

        // emit safe content (keep BUFFER_SIZE chars in case stop sequence is split)
        if (buffer.size() > BUFFER_SIZE) {
            size_t safe_len = buffer.size() - BUFFER_SIZE;
            std::string safe_content = buffer.substr(0, safe_len);
            if (on_token) {
                on_token(safe_content);
            }
            result.append(safe_content);
            buffer = buffer.substr(safe_len);
        }

        // prepare batch for next token
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_past_, { 0 }, true);
        n_past_++;
    }

    // emit any remaining buffered content (no stop sequence found)
    if (!buffer.empty() && on_token) {
        on_token(buffer);
    }
    result.append(buffer);

    llama_batch_free(batch);
    return result;
}

void TextInference::appendToContext(const std::string& text) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(text.size() + 1);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), false, false);
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], n_past_ + i, { 0 }, false);
    }
    llama_decode(ctx_, batch);
    n_past_ += n_tokens;
    llama_batch_free(batch);
}

void TextInference::clearHistory() {
    llama_memory_clear(llama_get_memory(ctx_), true);
    n_past_ = 0;
}

void TextInference::llama_batch_add(llama_batch &batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> &seq_ids, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens]   = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

void TextInference::llama_batch_clear(llama_batch &batch) {
    batch.n_tokens = 0;
}


void TextInference::shutdown() {
    if (sampler_) { llama_sampler_free(sampler_); sampler_ = nullptr; }
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
}
