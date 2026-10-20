// src/poly/gemma4.cpp - Gemma 4's layers.  The reference for the math is llama.cpp's gemma4 graph.
#include "gemma4.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time

// A value a file gives once for the model or once a layer.
int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

void Gemma4::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    freq_ = Dev{};
}

void Gemma4::load_geometry() {
    // --draft-model is a model of its own here, not a file of this one with a block in it, which is what
    // load_shape() would look for
    std::unique_ptr<Shards> other = std::move(dg_);
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };
    gelu_ = true;
    window_ = shape_i("attention.sliding_window");
    dim_full_ = shape_i("attention.key_length");
    dim_swa_ = shape_i("attention.key_length_swa");
    const MetaValue* rot = find("rope.dimension_count");
    const MetaValue* rot_swa = find("rope.dimension_count_swa");
    rot_full_ = rot ? (int) rot->u : dim_full_;
    rot_swa_ = rot_swa ? (int) rot_swa->u : dim_swa_;
    base_full_ = shape_f("rope.freq_base");
    const MetaValue* base_swa = find("rope.freq_base_swa");
    base_swa_ = base_swa ? (float) base_swa->f : base_full_;
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->f : 0.f;

    const MetaValue& pattern = must("attention.sliding_window_pattern");
    const MetaValue& heads = must("attention.head_count");
    const MetaValue& kv = must("attention.head_count_kv");
    swa_.resize((size_t) n_layer_);
    kv_.resize((size_t) n_layer_);
    n_head_ = (int) of_layer(heads, 0);
    for (int il = 0; il < n_layer_; ++il) {
        swa_[(size_t) il] = of_layer(pattern, il) != 0;
        kv_[(size_t) il] = (int) of_layer(kv, il);
        need(of_layer(heads, il) == n_head_, "layers with different numbers of heads");
        need(kv_[(size_t) il] > 0 && n_head_ % kv_[(size_t) il] == 0, "bad attention.head_count_kv");
    }
    need(n_head_ > 0 && n_head_ <= 256, "bad attention.head_count");
    need(window_ > 0 && dim_full_ > 0 && dim_swa_ > 0, "bad attention.sliding_window or key lengths");
    need(shape_i("attention.value_length") == dim_full_ && shape_i("attention.value_length_swa") == dim_swa_,
         "the attention's keys and values differ in length");
    need(rot_full_ % 2 == 0 && rot_full_ <= dim_full_ && rot_swa_ % 2 == 0 && rot_swa_ <= dim_swa_,
         "bad rope.dimension_count");
    const MetaValue* shared = find("attention.shared_kv_layers");
    need(!shared || shared->u == 0, "layers that read another layer's keys and values");
    const MetaValue* per_layer = find("embedding_length_per_layer_input");
    need(!per_layer || per_layer->u == 0, "token embeddings of a layer's own");

    // the dense MLP's width is its matrix's: a file may give it once a layer
    const TensorInfo* gate = tensor(0, "ffn_gate.weight");
    need(gate && gate->shape.size() == 2, "no blk.0.ffn_gate.weight");
    n_ff_ = (int) gate->shape[1];

    // GELU has no limits to clamp to
    clamp_exp_.assign((size_t) n_layer_, INFINITY);
    clamp_shexp_ = clamp_exp_;

    if (other) {
        dg_ = std::move(other);
        load_draft_shape();
        block_ = dg_.get();   // the file its tensors are read from (home())
        draft_ = true;
    }
}

// The draft model's shape, which must fit this model's: it takes the model's output and its token embeddings, its
// queries have the model's heads, and each of its layers reads the rows of one of the model's.  llama.cpp gives a
// window layer the model's layer before the last, and a full layer the last.
void Gemma4::load_draft_shape() {
    const Shards& g = *dg_;
    const std::string file = opt_.draft_model;
    auto need = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(file + " is not a draft model for this model: " + what);
    };
    const MetaValue* arch = g.get("general.architecture");
    need(arch && arch->s == "gemma4-assistant",
         "its architecture is " + (arch ? "'" + arch->s + "'" : std::string("missing")) + ", not 'gemma4-assistant'");
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = meta_in(g, key);
        need(v != nullptr, std::string("no gemma4-assistant.") + key);
        return *v;
    };
    d_embd_ = (int) must("embedding_length").u;
    d_ff_ = (int) must("feed_forward_length").u;
    d_eps_ = (float) must("attention.layer_norm_rms_epsilon").f;
    const int n = (int) must("block_count").u;
    need((int64_t) must("embedding_length_out").u == n_embd_, "it is made for a model of another width");
    need(d_embd_ > 0 && d_embd_ % 32 == 0 && d_embd_ <= n_embd_ && d_ff_ > 0 && d_ff_ % 32 == 0 && n > 0 && n <= 64,
         "bad embedding_length, feed_forward_length or block_count");
    need((int) must("attention.sliding_window").u == window_, "its window is not the model's");
    const MetaValue& pattern = must("attention.sliding_window_pattern");
    const MetaValue& heads = must("attention.head_count");
    const MetaValue& kv = must("attention.head_count_kv");
    const int dim_full = (int) must("attention.key_length").u, dim_swa = (int) must("attention.key_length_swa").u;
    const MetaValue* rot = meta_in(g, "rope.dimension_count");
    const MetaValue* rot_swa = meta_in(g, "rope.dimension_count_swa");
    const float base_full = (float) must("rope.freq_base").f;
    const MetaValue* base_swa = meta_in(g, "rope.freq_base_swa");
    need(n_layer_ >= 2 && swa_[(size_t) n_layer_ - 2] && !swa_[(size_t) n_layer_ - 1],
         "the model does not end on a window layer and a full layer");
    guess_.assign((size_t) n, Guess{});
    for (int il = 0; il < n; ++il) {
        Guess& G = guess_[(size_t) il];
        G.swa = of_layer(pattern, il) != 0;
        G.of = G.swa ? n_layer_ - 2 : n_layer_ - 1;
        G.dim = G.swa ? dim_swa : dim_full;
        G.n_rot = G.swa ? (rot_swa ? (int) rot_swa->u : dim_swa) : (rot ? (int) rot->u : dim_full);
        need(G.n_rot > 0 && G.n_rot % 2 == 0 && G.n_rot <= G.dim, "bad rope.dimension_count");
        G.theta_scale = std::pow(G.swa && base_swa ? (float) base_swa->f : base_full, -2.f / (float) G.n_rot);
        need(of_layer(heads, il) == n_head_ && G.dim == (G.swa ? dim_swa_ : dim_full_) &&
                 of_layer(kv, il) == kv_[(size_t) G.of],
             "layer " + std::to_string(il) + "'s heads are not those of the model's layer " + std::to_string(G.of));
    }
}

// Twice, like the model's weights: the first time adds up the arena.
void Gemma4::load_drafter() {
    const Shards& g = *dg_;
    auto up = [&](const std::string& name, int64_t a, int64_t b, Want w) { return upload(g.find(name), name, a, b, w); };
    d_pre_ = up("nextn.pre_projection.weight", 2 * n_embd_, d_embd_, kAnyMat);
    d_post_ = up("nextn.post_projection.weight", d_embd_, n_embd_, kAnyMat);
    d_out_norm_ = up("output_norm.weight", d_embd_, 1, kF32);
    d_table_ = up("token_embd.weight", d_embd_, n_vocab_, kGuessMat);
    d_freq_ = Dev{};
    for (int il = 0; il < (int) guess_.size(); ++il) {
        Guess& G = guess_[(size_t) il];
        auto f32 = [&](const char* n, int64_t a) { return up(blk(il, n), a, 1, kF32); };
        auto mat = [&](const char* n, int64_t a, int64_t b) { return up(blk(il, n), a, b, kAnyMat); };
        G.attn_norm = f32("attn_norm.weight", d_embd_);
        G.wq = mat("attn_q.weight", d_embd_, (int64_t) n_head_ * G.dim);
        G.wo = mat("attn_output.weight", (int64_t) n_head_ * G.dim, d_embd_);
        G.q_norm = f32("attn_q_norm.weight", G.dim);
        G.post_attn = f32("post_attention_norm.weight", d_embd_);
        G.ffn_norm = f32("ffn_norm.weight", d_embd_);
        G.ffn_gate = mat("ffn_gate.weight", d_embd_, d_ff_);
        G.ffn_up = mat("ffn_up.weight", d_embd_, d_ff_);
        G.ffn_down = mat("ffn_down.weight", d_ff_, d_embd_);
        G.post_ffn = f32("post_ffw_norm.weight", d_embd_);
        if (!G.swa && !d_freq_.p) d_freq_ = up("rope_freqs.weight", G.n_rot / 2, 1, kF32);
        const TensorInfo* s = g.find(blk(il, "layer_output_scale.weight"));
        if (!s) continue;
        if (s->type != kTypeF32 || s->elements() != 1 || g.payload(*s) != sizeof(float))
            throw std::runtime_error(s->name + " is " + shape_of(*s) + ", expected F32 [1]");
        std::memcpy(&G.out_scale, g.data(*s), sizeof(float));
    }
}

void Gemma4::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    auto has = [&](const char* n) { return tensor(il, n) != nullptr; };

    B.swa = swa_[(size_t) il] != 0;
    B.dim = B.swa ? dim_swa_ : dim_full_;
    B.n_kv = kv_[(size_t) il];
    B.n_rot = B.swa ? rot_swa_ : rot_full_;
    B.theta_scale = std::pow(B.swa ? base_swa_ : base_full_, -2.f / (float) B.n_rot);
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * B.dim);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) B.n_kv * B.dim);
    if (has("attn_v.weight")) B.wv = mat("attn_v.weight", n_embd_, (int64_t) B.n_kv * B.dim);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * B.dim, n_embd_);
    B.q_norm = f32("attn_q_norm.weight", B.dim);
    B.k_norm = f32("attn_k_norm.weight", B.dim);
    B.post_attn = f32("post_attention_norm.weight", n_embd_);
    // the full layers' rotary pairs turn by frequencies the file scales; the tensor is the model's, not a layer's
    if (!B.swa && !freq_.p)
        freq_ = upload(gg_->find("rope_freqs.weight"), "rope_freqs.weight", B.n_rot / 2, 1, kF32);

    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
    B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
    B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
    B.post_ffn = f32("post_ffw_norm.weight", n_embd_);
    if (const TensorInfo* s = tensor(il, "layer_output_scale.weight")) {
        if (s->type != kTypeF32 || s->elements() != 1 || home(*s).payload(*s) != sizeof(float))
            throw std::runtime_error(s->name + " is " + shape_of(*s) + ", expected F32 [1]");
        std::memcpy(&B.out_scale, home(*s).data(*s), sizeof(float));
    }
    if (!has("ffn_gate_inp.weight")) return;   // a layer with the MLP alone

    B.rscale = f32("ffn_gate_inp.scale", n_embd_);
    B.post_mlp = f32("post_ffw_norm_1.weight", n_embd_);
    B.pre_moe = f32("pre_ffw_norm_2.weight", n_embd_);
    B.post_moe = f32("post_ffw_norm_2.weight", n_embd_);
    load_moe(il, false);
    B.escale.assign((size_t) n_expert_, 1.f);
    if (const TensorInfo* s = tensor(il, "ffn_down_exps.scale")) {
        B.escale_dev = upload(s, blk(il, "ffn_down_exps.scale"), n_expert_, 1, kF32);
        if (s->type != kTypeF32 || (int64_t) s->elements() != n_expert_ ||
            home(*s).payload(*s) != (size_t) n_expert_ * sizeof(float))
            throw std::runtime_error(s->name + " is " + shape_of(*s) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        std::memcpy(B.escale.data(), home(*s).data(*s), (size_t) n_expert_ * sizeof(float));
    }
    for (const char* other : {"ffn_gate_exps.scale", "ffn_up_exps.scale", "ffn_gate_up_exps.scale"})
        if (has(other)) throw std::runtime_error(blk(il, other) + ": a scale this engine does not apply");
}

// The draft model's inputs count: the model's output beside a token's embedding, and its MLPs' hidden values.
int Gemma4::widest() const {
    return std::max({n_head_ * std::max(dim_full_, dim_swa_), guess_.empty() ? 0 : 2 * n_embd_, d_ff_});
}

// The caches: a full layer keeps a row a position, a window layer a ring of the rows a call can read.
void Gemma4::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    size_t rings = 0;   // a window layer's rows are a ring, whatever the sequence's length
    for (const Block& B : blocks_)
        if (B.swa) rings += 2 * (size_t) ring_ * B.n_kv * B.dim;
    uint16_t* kv = rings ? (uint16_t*) dmalloc(rings * sizeof(uint16_t), "the attention cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        const size_t row = (size_t) B.n_kv * B.dim;
        if (B.swa) {
            B.keys = kv;
            kv += ring_ * row;
            B.vals = kv;
            kv += ring_ * row;
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Gemma4::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * widest();
    size_t kv = 0;
    for (const Block& B : blocks_) kv = std::max(kv, n * B.n_kv * B.dim);
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    vraw_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    mlp_ = fbuf(n * n_embd_);
    rin_ = fbuf(n * n_embd_);
    if (!guess_.empty()) {   // one token at a time
        dxh_ = fbuf(2 * (size_t) n_embd_);
        dg_gate_ = fbuf((size_t) d_ff_);
        dg_up_ = fbuf((size_t) d_ff_);
        dg_h_ = fbuf((size_t) d_ff_);
    }
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

// A sequence's rows are the full layers': a window layer's are part of its state (save_layer()).
size_t Gemma4::card_row_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_)
        if (!B.swa) b += 2 * (size_t) B.n_kv * B.dim * sizeof(uint16_t);
    return b;
}

void Gemma4::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) {
        const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t), bytes = (size_t) n * row;
        if (B.swa || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + (size_t) first * row, bytes, cudaMemcpyDeviceToHost),
               "save rows");
            dst += bytes;
        }
    }
}

void Gemma4::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) {
        const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t), bytes = (size_t) n * row;
        if (B.swa || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + (size_t) first * row, src + (size_t) (first - from) * row, bytes,
                          cudaMemcpyHostToDevice),
               "load rows");
            src += (size_t) count * row;
        }
    }
}

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
template <class F>
void Gemma4::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What a window layer has of the sequence: the rows the next token attends.
size_t Gemma4::extra_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_)
        if (B.swa) b += 2 * (size_t) keep_ * B.n_kv * B.dim * sizeof(uint16_t);
    return b;
}

// A window layer's rows, behind the recurrent state (which this architecture has none of) and the window layers
// before it.
void Gemma4::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < il; ++j) {
        const Block& B = blocks_[(size_t) j];
        if (B.swa) dst += 2 * (size_t) keep_ * B.n_kv * B.dim * sizeof(uint16_t);
    }
    const Block& B = blocks_[(size_t) il];
    const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t);
    for (const uint16_t* cache : {B.keys, B.vals}) {
        std::memset(dst, 0, (size_t) keep_ * row);   // the places of positions before the sequence's start
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(dst + slot * row, (const uint8_t*) cache + at * row, m * row, cudaMemcpyDeviceToHost),
               "save state");
        });
        dst += (size_t) keep_ * row;
    }
}

void Gemma4::load_extra(const uint8_t* src, int64_t pos) {
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        const size_t row = (size_t) B.n_kv * B.dim * sizeof(uint16_t);
        for (uint16_t* cache : {B.keys, B.vals}) {
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy((uint8_t*) cache + at * row, src + slot * row, m * row, cudaMemcpyHostToDevice),
                   "load state");
            });
            src += (size_t) keep_ * row;
        }
    }
}

// The recordings of a full layer's head are numbered by the rows they are sized for; a window layer's never
// reads more than its window.
int Gemma4::head_key(int il, int n) const { return swa_[(size_t) il] ? 0 : room_key(n); }

// n tokens' rows from position `pos` into a layer's cache.  A ring's rows may end and go on from its start.  A
// recorded launch stores them from the position at_ holds when it runs.
void Gemma4::store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n) {
    const int width = B.n_kv * B.dim;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// A picture's rows see each other in the window layers (llama.cpp reads a picture without the causal mask there,
// and with it in the full layers), so a picture goes into the model in a call of its own.
int64_t Gemma4::whole(int64_t at, int64_t m) const {
    for (const Image& im : images_) {
        if (at >= im.at + im.n) continue;
        if (at < im.at) return std::min(m, im.at - at);
        if (at != im.at) throw std::logic_error("a sequence cannot go on from the middle of a picture");
        if (im.n > rows_)
            throw std::runtime_error("a picture of " + std::to_string(im.n) + " rows is more than a prompt batch (" +
                                     std::to_string(rows_) + "): raise --prompt-batch or lower the picture's tokens");
        return im.n;
    }
    return m;
}

// Queries and keys are normed a head and turned by the position; values are normed without weights.  The scores
// are not scaled: the norms set their size.
void Gemma4::attn(Block& B, int64_t pos, int n) {
    // a call is a picture or has no row of one (whole())
    const Image* im = image_at(pos);
    for (const Image& o : images_)
        if (o.at < pos + n && o.at + o.n > pos && (&o != im || o.at != pos || o.n != n))
            throw std::logic_error("a picture's rows in a call with other rows");
    const bool sees_all = B.swa && im != nullptr;
    const float* freq = B.swa ? nullptr : fp(freq_);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    lin(B.wq, xn_, qraw_, n, xq_);
    head_norm_rope(qraw_, B.dim, fp(B.q_norm), q_, B.dim, n_head_, B.n_rot, B.theta_scale, pos, n, rms_eps_, st_,
                   freq, nullptr, 1, at);
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, B.dim, fp(B.k_norm), k_, B.dim, B.n_kv, B.n_rot, B.theta_scale, pos, n, rms_eps_, st_,
                   freq, nullptr, 1, at);
    const float* values = kraw_;   // a layer without a value projection takes the keys'
    if (B.wv.p) {
        lin(B.wv, xn_, vraw_, n, xq_);
        values = vraw_;
    }
    head_norm_rope(values, B.dim, nullptr, v_, B.dim, B.n_kv, 0, 1.f, pos, n, rms_eps_, st_);
    store(k_, B.keys, B, pos, n);
    store(v_, B.vals, B, pos, n);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, B.dim, n_head_, B.n_kv, 1.f, st_, window, ring, sees_all);
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, B.dim, n_head_, B.n_kv, 1.f, st_, window, ring,
               sees_all, at, !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n));
    lin(B.wo, att_, ao_, n);
}

// A layer from its input to its router's copy (a layer without routed experts: to its end).  r_ holds the
// residual with the attention added from the middle on, which the MLP, the router and the experts each norm
// their own way.  A prompt batch's MoE (moe_rows) does its own routing.
void Gemma4::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int all = n * n_embd_;
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, all, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    rms_norm(ao_, fp(B.post_attn), cur_, n_embd_, n, rms_eps_, st_);
    dbg("attn", il, cur_, n_embd_);
    axpy(cur_, 1.f, r_, all, true, st_);

    if (!L.moe) {
        mlp(L, il, n, fo_);
        rms_norm(fo_, fp(B.post_ffn), cur_, n_embd_, n, rms_eps_, st_);
        dbg("ffn", il, cur_, n_embd_);
        axpy(cur_, 1.f, r_, all, true, st_);
        if (B.out_scale != 1.f) axpy(r_, B.out_scale, r_, all, false, st_);
        return;
    }
    // the MLP is beside the experts, and goes after the router, while the host routes (moe_shared())
    rms_norm(r_, fp(B.rscale), rin_, n_embd_, n, rms_eps_, st_);
    rms_norm(r_, fp(B.pre_moe), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, all, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

// The router reads the residual, normed and scaled a value at a time (rin_).
void Gemma4::router_scores(const Layer& L, int n) { lin(L.gate_inp, rin_, rl_, n); }

// The dense MLP of the layer's normed residual, into `out`.
void Gemma4::mlp(const Layer& L, int il, int n, float* out) {
    const Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.ffn_norm), cur_, n_embd_, n, rms_eps_, st_);
    lin(B.ffn_gate, cur_, g_, n);
    lin(B.ffn_up, cur_, u_, n);
    geglu(g_, u_, h_, n * n_ff_, st_);
    lin(B.ffn_down, h_, out, n);
}

// No expert is shared: the routed experts' sum starts from nothing.  The MLP beside them goes here, after the
// router's copy: the card has it to do while the host routes and computes its experts.
void Gemma4::moe_shared(const Layer& L, int il, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
    mlp(L, il, n, cur_);
    rms_norm(cur_, fp(blocks_[(size_t) il].post_mlp), mlp_, n_embd_, n, rms_eps_, st_);
    dbg("mlp", il, mlp_, n_embd_);
}

// The routed experts' sum is normed, the MLP's output is added, their sum is normed again and goes into the
// residual, which the layer then scales.
void Gemma4::layer_tail(Layer&, int il, int n) {
    const Block& B = blocks_[(size_t) il];
    const int all = n * n_embd_;
    rms_norm(fo_, fp(B.post_moe), cur_, n_embd_, n, rms_eps_, st_);
    axpy(mlp_, 1.f, cur_, all, true, st_);
    rms_norm(cur_, fp(B.post_ffn), fo_, n_embd_, n, rms_eps_, st_);
    axpy(fo_, 1.f, r_, all, true, st_);
    if (B.out_scale != 1.f) axpy(r_, B.out_scale, r_, all, false, st_);
}

// The MLP's output is the head's, and the tail adds it to the experts'.
void Gemma4::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    mlp_ += by * n_embd_;
}

void Gemma4::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Gemma4::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, softcap_, st_);
}

// One guess of the draft model.  Its input is the token as the model embeds it and, beside it, the model's output
// for the token before: the model's own for the first guess, and from then on the draft model's guess at it (its
// second output).  Every guess asks from the position of the first, pos(), and reads the rows before it; the
// guessed tokens leave no rows.  The model's scratch is free between its calls, and a guess is narrower than a
// token of the model's.
void Gemma4::draft_step(int32_t token, int i) {
    if (pos_ < 1) throw std::logic_error("draft: the sequence has no token yet");
    const int e = n_embd_, a = d_embd_;
    float* own = hn_ + e;
    embed(token, h_r_);
    const float scale = embed_scale();
    for (int k = 0; k < e; ++k) h_r_[k] *= scale;
    copy2(h_r_, dxh_, e, i == 0 ? hn_ : own, dxh_ + e, e, st_);
    float* x = r_;
    lin(d_pre_, dxh_, x, 1);
    for (const Guess& G : guess_) {
        const Block& B = blocks_[(size_t) G.of];
        rms_norm(x, fp(G.attn_norm), xn_, a, 1, d_eps_, st_);
        lin(G.wq, xn_, qraw_, 1);
        head_norm_rope(qraw_, G.dim, fp(G.q_norm), q_, G.dim, n_head_, G.n_rot, G.theta_scale, pos_, 1, d_eps_, st_,
                       G.swa ? nullptr : fp(d_freq_));
        // a window layer's token attends itself and the window_ - 1 before it; the asking token has no row
        attend(q_, B.keys, B.vals, pos_ - 1, 1, score_, chunk_, att_, G.dim, n_head_, B.n_kv, 1.f, st_,
               G.swa ? window_ - 1 : 0, G.swa ? ring_ : 0);
        lin(G.wo, att_, ao_, 1);
        rms_norm(ao_, fp(G.post_attn), cur_, a, 1, d_eps_, st_);
        axpy(cur_, 1.f, x, a, true, st_);

        rms_norm(x, fp(G.ffn_norm), xn_, a, 1, d_eps_, st_);
        lin(G.ffn_gate, xn_, dg_gate_, 1);
        lin(G.ffn_up, xn_, dg_up_, 1);
        geglu(dg_gate_, dg_up_, dg_h_, d_ff_, st_);
        lin(G.ffn_down, dg_h_, cur_, 1);
        rms_norm(cur_, fp(G.post_ffn), fo_, a, 1, d_eps_, st_);
        axpy(fo_, 1.f, x, a, true, st_);
        if (G.out_scale != 1.f) axpy(x, G.out_scale, x, a, false, st_);
    }
    rms_norm(x, fp(d_out_norm_), xn_, a, 1, d_eps_, st_);
    Dev table = d_table_;   // the matrix's first rows are the common tokens'
    table.n_out = draft_rows();
    lin(table, xn_, logits_, 1);
    lin(d_post_, xn_, own, 1);
}

float Gemma4::embed_scale() const { return std::sqrt((float) n_embd_); }

bool Gemma4::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 0, 1.f / std::sqrt((float) n_embd_), nullptr, fp(B.escale_dev), true, 0.f,
               ids, w, st_);
    return true;
}

// What ggml's MoE graph does for this model: the router's scores over the root of the width, a softmax over every
// expert, the n_used most likely with their weights made to sum to one, and each expert's own scale on its output.
void Gemma4::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& escale = blocks_[(size_t) il].escale;
    const float s = 1.f / std::sqrt((float) n_embd_);
    float mx = logits[0] * s;
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e] * s);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] * s - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] = w[k] / denom * escale[(size_t) ids[k]];
}

}  // namespace strata::poly
