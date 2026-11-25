#include "emeans.cuh"
#include "ggml-cuda.h"

#if defined(GGML_USE_HIP) || defined(__CUDACC__) || defined(GGML_USE_CUDA)

#ifdef GGML_USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define EM_LAUNCH_BOUNDS      __attribute__((amdgpu_flat_work_group_size(256, 256)))
#define EM_LAUNCH_BOUNDS_GEMV __attribute__((amdgpu_flat_work_group_size(256, 256), amdgpu_waves_per_eu(16, 16)))
#define EM_SHFL(v, lane, w)  __shfl((v), (lane), (w))
#define EM_SHFL_XOR(v, off)  __shfl_xor((v), (off), 32)
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#define hipStream_t            cudaStream_t
#define hipError_t             cudaError_t
#define hipSuccess             cudaSuccess
#define hipMalloc              cudaMalloc
#define hipFree                cudaFree
#define hipMemcpy              cudaMemcpy
#define hipMemcpyHostToDevice  cudaMemcpyHostToDevice
#define hipGetErrorString      cudaGetErrorString
#define EM_LAUNCH_BOUNDS       __launch_bounds__(256)
#define EM_LAUNCH_BOUNDS_GEMV  __launch_bounds__(256)
#define EM_SHFL(v, lane, w)    __shfl_sync(0xFFFFFFFFu, (v), (lane), (w))
#define EM_SHFL_XOR(v, off)    __shfl_xor_sync(0xFFFFFFFFu, (v), (off), 32)
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Minimal embedded .emns format & streaming VRAM loader (matches emeans_format.h)
// ============================================================================
static constexpr char EMNS_MAGIC[8] = {'E','M','N','S','0','0','0','1'};

enum EmFormat : uint32_t {
    EM_FMT_FP32          = 0,
    EM_FMT_FP16          = 1,
    EM_FMT_BF16          = 2,
    EM_FMT_EM1D_Q4_G64   = 3,
    EM_FMT_EM1D_Q3_G64   = 4,
    EM_FMT_EM2D_K32_G128 = 5,
    EM_FMT_EM2D_K16_G128 = 6,
};

struct EmPlane {
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    void *   d_ptr  = nullptr;
};

struct EmTensor {
    std::string name;
    uint32_t fmt = 0;
    bool is_w13 = false;
    std::vector<int64_t> shape;
    std::unordered_map<std::string, EmPlane> planes;

    const EmPlane * plane(const char * key) const {
        auto it = planes.find(key);
        return (it == planes.end()) ? nullptr : &it->second;
    }
    template <typename T>
    const T * ptr(const char * key) const {
        const EmPlane * p = plane(key);
        return p ? reinterpret_cast<const T *>(p->d_ptr) : nullptr;
    }
};

struct EmnsState {
    std::mutex mtx;
    bool loaded = false;
    std::string loaded_path;
    uint8_t * d_blob = nullptr;
    uint64_t data_bytes = 0;
    std::vector<uint8_t *> extra_blobs;
    std::vector<std::string> extra_paths;
    std::unordered_map<std::string, EmTensor> tensors;

    // Tiny GPU stub buffer (64 bytes per bound tensor) so llama.cpp assigns
    // overridden weights to HIP without allocating GGUF weight VRAM.
    static constexpr size_t STUB_STRIDE = 64;
    static constexpr size_t MAX_STUBS   = 2048;
    ggml_backend_buffer_t stub_buf = nullptr;
    uint8_t * stub_base = nullptr;
    size_t stub_used = 0;
    std::unordered_map<const void *, const EmTensor *> ptr_map;
};

static EmnsState & emns_state() {
    static EmnsState s;
    return s;
}

// --- Minimal JSON manifest parser for EmnsWriter ---
static void em_skip_ws(const char * s, size_t n, size_t & i) {
    while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}
static bool em_parse_str(const char * s, size_t n, size_t & i, std::string & out) {
    em_skip_ws(s, n, i);
    if (i >= n || s[i] != '"') return false;
    ++i;
    out.clear();
    while (i < n && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < n) { ++i; out.push_back(s[i++]); }
        else { out.push_back(s[i++]); }
    }
    if (i >= n || s[i] != '"') return false;
    ++i;
    return true;
}
static bool em_parse_u64(const char * s, size_t n, size_t & i, uint64_t & out) {
    em_skip_ws(s, n, i);
    if (i >= n || s[i] < '0' || s[i] > '9') return false;
    uint64_t v = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9') v = v * 10 + (uint64_t)(s[i++] - '0');
    out = v;
    return true;
}
static bool em_skip_value(const char * s, size_t n, size_t & i) {
    em_skip_ws(s, n, i);
    if (i >= n) return false;
    if (s[i] == '"') { std::string tmp; return em_parse_str(s, n, i, tmp); }
    if (s[i] == '{' || s[i] == '[') {
        char open = s[i], close = (open == '{') ? '}' : ']';
        int depth = 0;
        while (i < n) {
            if (s[i] == '"') { std::string tmp; if (!em_parse_str(s, n, i, tmp)) return false; continue; }
            if (s[i] == open) ++depth;
            else if (s[i] == close && --depth == 0) { ++i; return true; }
            ++i;
        }
        return false;
    }
    while (i < n && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
    return true;
}

static bool em_parse_manifest(const std::string & json, std::unordered_map<std::string, EmTensor> & out) {
    const char * s = json.c_str();
    const size_t n = json.size();
    const size_t pos = json.find("\"tensors\"");
    if (pos == std::string::npos) return false;
    size_t i = pos + 9;
    em_skip_ws(s, n, i);
    if (i >= n || s[i] != ':') return false;
    ++i;
    em_skip_ws(s, n, i);
    if (i >= n || s[i] != '[') return false;
    ++i;

    while (i < n) {
        em_skip_ws(s, n, i);
        if (i < n && s[i] == ']') { ++i; return true; }
        if (i >= n || s[i] != '{') return false;
        ++i;
        EmTensor t;
        while (i < n) {
            em_skip_ws(s, n, i);
            if (i < n && s[i] == '}') { ++i; break; }
            std::string key;
            if (!em_parse_str(s, n, i, key)) return false;
            em_skip_ws(s, n, i);
            if (i >= n || s[i] != ':') return false;
            ++i;
            if (key == "name") {
                if (!em_parse_str(s, n, i, t.name)) return false;
            } else if (key == "fmt") {
                uint64_t v = 0;
                if (!em_parse_u64(s, n, i, v)) return false;
                t.fmt = (uint32_t)v;
            } else if (key == "shape") {
                em_skip_ws(s, n, i);
                if (i >= n || s[i] != '[') return false;
                ++i;
                while (i < n) {
                    em_skip_ws(s, n, i);
                    if (s[i] == ']') { ++i; break; }
                    uint64_t d = 0;
                    if (!em_parse_u64(s, n, i, d)) return false;
                    t.shape.push_back((int64_t)d);
                    em_skip_ws(s, n, i);
                    if (i < n && s[i] == ',') ++i;
                }
            } else if (key == "planes") {
                em_skip_ws(s, n, i);
                if (i >= n || s[i] != '{') return false;
                ++i;
                while (i < n) {
                    em_skip_ws(s, n, i);
                    if (s[i] == '}') { ++i; break; }
                    std::string pkey;
                    if (!em_parse_str(s, n, i, pkey)) return false;
                    em_skip_ws(s, n, i);
                    if (i >= n || s[i] != ':') return false;
                    ++i;
                    em_skip_ws(s, n, i);
                    if (i >= n || s[i] != '[') return false;
                    ++i;
                    EmPlane pl;
                    if (!em_parse_u64(s, n, i, pl.offset)) return false;
                    em_skip_ws(s, n, i);
                    if (i >= n || s[i] != ',') return false;
                    ++i;
                    if (!em_parse_u64(s, n, i, pl.nbytes)) return false;
                    em_skip_ws(s, n, i);
                    if (i >= n || s[i] != ']') return false;
                    ++i;
                    t.planes[pkey] = pl;
                    em_skip_ws(s, n, i);
                    if (i < n && s[i] == ',') ++i;
                }
            } else {
                if (!em_skip_value(s, n, i)) return false;
            }
            em_skip_ws(s, n, i);
            if (i < n && s[i] == ',') ++i;
        }
        t.is_w13 = false;
        out[t.name] = std::move(t);
        em_skip_ws(s, n, i);
        if (i < n && s[i] == ',') ++i;
    }
    return false;
}

// --- Map llama.cpp GGUF tensor name -> .emns tensor name ---
static std::string map_gguf_to_emns(const char * gguf_name) {
    if (!gguf_name) return "";
    std::string g(gguf_name);
    if (g == "token_embd.weight") return "model.llm.embed.weight";
    if (g == "output.weight")     return "model.llm.unembed.weight";
    if (g.rfind("blk.", 0) != 0)  return "";

    size_t dot2 = g.find('.', 4);
    if (dot2 == std::string::npos) return "";
    std::string idx = g.substr(4, dot2 - 4);
    std::string suf = g.substr(dot2 + 1);
    std::string pfx = "model.llm.layers." + idx + ".";

    if (suf == "attn_q.weight")            return pfx + "attn.wq_du.weight";
    if (suf == "attn_k.weight")            return pfx + "attn.wk_dv.weight";
    if (suf == "attn_v.weight")            return pfx + "attn.wv_dv.weight";
    if (suf == "attn_output.weight")       return pfx + "attn.wo_ud.weight";
    if (suf == "attn_gate.weight")         return pfx + "attn.gate_attn.weight";
    if (suf == "attn_rel.weight")          return pfx + "attn.wr_du.weight";
    if (suf == "ffn_gate_up_exps.weight")  return pfx + "mlp.experts.w13_weight";
    if (suf == "ffn_down_exps.weight")     return pfx + "mlp.experts.w2_weight";

    // Loop controllers
    auto map_ctrl = [&](const char * g_ctrl, const char * e_ctrl) -> std::string {
        std::string base = std::string("loop_") + g_ctrl + "_ctrl_";
        if (suf == base + "down.weight")   return pfx + "controllers." + e_ctrl + ".down_weight";
        if (suf == base + "gain.weight")   return pfx + "controllers." + e_ctrl + ".gain_weight";
        if (suf == base + "forget.weight") return pfx + "controllers." + e_ctrl + ".forget_weight";
        return "";
    };
    if (auto r = map_ctrl("q",   "attn"); !r.empty()) return r;
    if (auto r = map_ctrl("k",   "k");    !r.empty()) return r;
    if (auto r = map_ctrl("v",   "v");    !r.empty()) return r;
    if (auto r = map_ctrl("ffn", "mlp");  !r.empty()) return r;
    return "";
}

// ============================================================================
// Public C API (called from llama-model.cpp / llama-model-loader.cpp)
// ============================================================================
extern "C" {

bool ggml_cuda_emeans_init(const char * emns_path) {
    if (!emns_path || !emns_path[0]) return false;
    auto & st = emns_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    if (st.loaded && st.loaded_path == emns_path) {
        return true;
    }

    FILE * f = fopen(emns_path, "rb");
    if (!f) {
        fprintf(stderr, "[emeans] ERROR: cannot open '%s'\n", emns_path);
        return false;
    }
    char magic[8];
    uint64_t manifest_len = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, EMNS_MAGIC, 8) != 0 ||
        fread(&manifest_len, 8, 1, f) != 1) {
        fprintf(stderr, "[emeans] ERROR: invalid .emns header in '%s'\n", emns_path);
        fclose(f);
        return false;
    }
    std::string mjson(manifest_len, '\0');
    if (fread(&mjson[0], 1, manifest_len, f) != manifest_len) {
        fclose(f);
        return false;
    }
    uint64_t hdr_end = 16 + manifest_len;
    uint64_t data_base = (hdr_end + 63ULL) & ~63ULL;
#if defined(_WIN32)
    _fseeki64(f, 0, SEEK_END);
    uint64_t file_size = (uint64_t)_ftelli64(f);
    _fseeki64(f, (long long)data_base, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END);
    uint64_t file_size = (uint64_t)ftello(f);
    fseeko(f, (off_t)data_base, SEEK_SET);
#endif
    uint64_t data_bytes = file_size - data_base;

    st.tensors.clear();
    if (!em_parse_manifest(mjson, st.tensors)) {
        fprintf(stderr, "[emeans] ERROR: failed to parse manifest\n");
        fclose(f);
        return false;
    }

    if (st.d_blob) { hipFree(st.d_blob); st.d_blob = nullptr; }
    hipError_t err = hipMalloc(&st.d_blob, data_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "[emeans] ERROR: hipMalloc(%.3f GiB) failed: %s\n",
                data_bytes / 1073741824.0, hipGetErrorString(err));
        fclose(f);
        return false;
    }
    st.data_bytes = data_bytes;

    const size_t CHUNK = 64ULL << 20;
    std::vector<uint8_t> h_buf(CHUNK);
    uint64_t done = 0;
    while (done < data_bytes) {
        size_t want = (size_t)((data_bytes - done < CHUNK) ? (data_bytes - done) : CHUNK);
        size_t got = fread(h_buf.data(), 1, want, f);
        if (got != want) {
            fprintf(stderr, "[emeans] ERROR: short read at %llu / %llu\n",
                    (unsigned long long)done, (unsigned long long)data_bytes);
            hipFree(st.d_blob); st.d_blob = nullptr;
            fclose(f);
            return false;
        }
        err = hipMemcpy(st.d_blob + done, h_buf.data(), got, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "[emeans] ERROR: hipMemcpy failed: %s\n", hipGetErrorString(err));
            hipFree(st.d_blob); st.d_blob = nullptr;
            fclose(f);
            return false;
        }
        done += got;
    }
    fclose(f);

    for (auto & kv : st.tensors) {
        for (auto & pk : kv.second.planes) {
            pk.second.d_ptr = st.d_blob + pk.second.offset;
        }
    }

    st.loaded = true;
    st.loaded_path = emns_path;
    st.ptr_map.clear();
    st.stub_used = 0;
    fprintf(stderr, "[emeans] Loaded '%s': %.4f GiB (%zu tensors) into contiguous HIP VRAM\n",
            emns_path, data_bytes / 1073741824.0, st.tensors.size());
    return true;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
bool ggml_cuda_emeans_init_extra(const char * emns_path) {
    if (!emns_path || !emns_path[0]) return false;
    auto & st = emns_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    for (const auto & p : st.extra_paths) {
        if (p == emns_path) return true;
    }

    FILE * f = fopen(emns_path, "rb");
    if (!f) {
        fprintf(stderr, "[emeans] ERROR: cannot open extra '%s'\n", emns_path);
        return false;
    }
    char magic[8];
    uint64_t manifest_len = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, EMNS_MAGIC, 8) != 0 ||
        fread(&manifest_len, 8, 1, f) != 1) {
        fprintf(stderr, "[emeans] ERROR: invalid .emns header in '%s'\n", emns_path);
        fclose(f);
        return false;
    }
    std::string mjson(manifest_len, '\0');
    if (fread(&mjson[0], 1, manifest_len, f) != manifest_len) {
        fclose(f);
        return false;
    }
    uint64_t hdr_end = 16 + manifest_len;
    uint64_t data_base = (hdr_end + 63ULL) & ~63ULL;
#if defined(_WIN32)
    _fseeki64(f, 0, SEEK_END);
    uint64_t file_size = (uint64_t)_ftelli64(f);
    _fseeki64(f, (long long)data_base, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END);
    uint64_t file_size = (uint64_t)ftello(f);
    fseeko(f, (off_t)data_base, SEEK_SET);
#endif
    uint64_t data_bytes = file_size - data_base;

    std::unordered_map<std::string, EmTensor> extra_tensors;
    if (!em_parse_manifest(mjson, extra_tensors)) {
        fprintf(stderr, "[emeans] ERROR: failed to parse extra manifest\n");
        fclose(f);
        return false;
    }

    uint8_t * d_extra = nullptr;
    hipError_t err = hipMalloc(&d_extra, data_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "[emeans] ERROR: hipMalloc(%.3f MiB) failed for extra: %s\n",
                data_bytes / 1048576.0, hipGetErrorString(err));
        fclose(f);
        return false;
    }

    const size_t CHUNK = 64ULL << 20;
    std::vector<uint8_t> h_buf(CHUNK);
    uint64_t done = 0;
    while (done < data_bytes) {
        size_t want = (size_t)((data_bytes - done < CHUNK) ? (data_bytes - done) : CHUNK);
        size_t got = fread(h_buf.data(), 1, want, f);
        if (got != want) {
            hipFree(d_extra);
            fclose(f);
            return false;
        }
        err = hipMemcpy(d_extra + done, h_buf.data(), got, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            hipFree(d_extra);
            fclose(f);
            return false;
        }
        done += got;
    }
    fclose(f);

    size_t n_added = 0;
    for (auto & kv : extra_tensors) {
        if (kv.second.fmt != EM_FMT_EM1D_Q4_G64 && kv.second.fmt != EM_FMT_EM1D_Q3_G64 &&
            kv.second.fmt != EM_FMT_EM2D_K32_G128 && kv.second.fmt != EM_FMT_EM2D_K16_G128) {
            continue;
        }
        for (auto & pk : kv.second.planes) {
            pk.second.d_ptr = d_extra + pk.second.offset;
        }
        st.tensors[kv.first] = std::move(kv.second);
        ++n_added;
    }

    st.extra_blobs.push_back(d_extra);
    st.extra_paths.push_back(emns_path);
    st.loaded = true;
    fprintf(stderr, "[emeans] Loaded extra '%s': %.2f MiB (%zu quantized tensors) into HIP VRAM\n",
            emns_path, data_bytes / 1048576.0, n_added);
    return true;
}

ggml_backend_buffer_t ggml_cuda_emeans_get_stub_buffer(ggml_backend_buffer_type_t buft) {
    auto & st = emns_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    if (!st.loaded) return nullptr;
    if (!st.stub_buf) {
        st.stub_buf = ggml_backend_buft_alloc_buffer(buft, EmnsState::MAX_STUBS * EmnsState::STUB_STRIDE);
        if (!st.stub_buf) return nullptr;
        ggml_backend_buffer_set_usage(st.stub_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        st.stub_base = (uint8_t *)ggml_backend_buffer_get_base(st.stub_buf);
    }
    return st.stub_buf;
}

void * ggml_cuda_emeans_bind_tensor(const char * gguf_name, int64_t ne0, int64_t ne1, int64_t ne2) {
    auto & st = emns_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    if (!st.loaded || !st.stub_base) return nullptr;

    std::string ename = map_gguf_to_emns(gguf_name);
    auto it = ename.empty() ? st.tensors.end() : st.tensors.find(ename);
    if (it == st.tensors.end() && gguf_name) {
        it = st.tensors.find(std::string(gguf_name));
    }
    if (it == st.tensors.end()) return nullptr;
    const EmTensor & em = it->second;

    // Only bind quantized formats (EM1D_Q4_G64, EM1D_Q3_G64, EM2D_K32_G128, EM2D_K16_G128)
    if (em.fmt != EM_FMT_EM1D_Q4_G64 && em.fmt != EM_FMT_EM1D_Q3_G64 &&
        em.fmt != EM_FMT_EM2D_K32_G128 && em.fmt != EM_FMT_EM2D_K16_G128) {
        return nullptr;
    }
    // Verify shape match (GGML [K, N, E] vs EMNS [N, K] or [E, N, K])
    if (em.shape.size() == 2) {
        if (em.shape[0] != ne1 || em.shape[1] != ne0 || ne2 != 1) {
            fprintf(stderr, "[emeans] WARN: shape mismatch for %s: ggml=[%lld,%lld,%lld] vs emns=[%lld,%lld]\n",
                    gguf_name, (long long)ne0, (long long)ne1, (long long)ne2,
                    (long long)em.shape[0], (long long)em.shape[1]);
            return nullptr;
        }
    } else if (em.shape.size() == 3) {
        if (em.shape[0] != ne2 || em.shape[1] != ne1 || em.shape[2] != ne0) {
            fprintf(stderr, "[emeans] WARN: shape mismatch for %s: ggml=[%lld,%lld,%lld] vs emns=[%lld,%lld,%lld]\n",
                    gguf_name, (long long)ne0, (long long)ne1, (long long)ne2,
                    (long long)em.shape[0], (long long)em.shape[1], (long long)em.shape[2]);
            return nullptr;
        }
    } else {
        return nullptr;
    }

    if (st.stub_used >= EmnsState::MAX_STUBS) return nullptr;
    void * stub_ptr = st.stub_base + st.stub_used * EmnsState::STUB_STRIDE;
    st.stub_used++;
    st.ptr_map[stub_ptr] = &em;
    return stub_ptr;
}

bool ggml_cuda_emeans_is_stub(const void * ptr) {
    auto & st = emns_state();
    if (!st.loaded || !ptr || !st.stub_base) return false;
    const uint8_t * p = (const uint8_t *)ptr;
    return p >= st.stub_base && p < st.stub_base + EmnsState::MAX_STUBS * EmnsState::STUB_STRIDE;
}

} // extern "C"

static inline const EmTensor * emns_lookup(const ggml_tensor * src0) {
    auto & st = emns_state();
    if (!st.loaded || !src0 || !src0->data) return nullptr;
    const uint8_t * p = (const uint8_t *)src0->data;
    if (p < st.stub_base || p >= st.stub_base + EmnsState::MAX_STUBS * EmnsState::STUB_STRIDE) {
        return nullptr;
    }
    auto it = st.ptr_map.find(src0->data);
    return (it == st.ptr_map.end()) ? nullptr : it->second;
}

// ============================================================================
// Wave32 / Warp32 Packed-LUT HIP + CUDA Kernels (VGPR<=59, 0 spills)
// ============================================================================
#ifdef GGML_USE_HIP
typedef _Float16 em_f16x2_t __attribute__((ext_vector_type(2)));
#endif

#ifdef GGML_USE_HIP
typedef float em_gacc_t;
static __device__ __forceinline__ em_gacc_t em_gacc_zero() { return 0.0f; }
static __device__ __forceinline__ em_gacc_t em_gacc_fdot2(uint32_t a_bits, uint32_t b_bits, em_gacc_t acc) {
    union { uint32_t u; em_f16x2_t v; } ua{a_bits}, ub{b_bits};
#if !defined(__gfx1010__) && !defined(__gfx1011__) && !defined(__gfx1012__) && !defined(__gfx900__) && !defined(__gfx906__)
    return __builtin_amdgcn_fdot2(ua.v, ub.v, acc, false);
#else
    return fmaf((float)ua.v[0], (float)ub.v[0], fmaf((float)ua.v[1], (float)ub.v[1], acc));
#endif
}
static __device__ __forceinline__ float em_gacc_to_f32(em_gacc_t acc) { return acc; }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
typedef __half2 em_gacc_t;
static __device__ __forceinline__ em_gacc_t em_gacc_zero() { return __float2half2_rn(0.0f); }
static __device__ __forceinline__ em_gacc_t em_gacc_fdot2(uint32_t a_bits, uint32_t b_bits, em_gacc_t acc) {
    __half2 ha, hb;
    memcpy(&ha, &a_bits, sizeof(ha));
    memcpy(&hb, &b_bits, sizeof(hb));
    return __hfma2(ha, hb, acc);
}
static __device__ __forceinline__ float em_gacc_to_f32(em_gacc_t acc) {
    float2 f2 = __half22float2(acc);
    return f2.x + f2.y;
}
#else
typedef float em_gacc_t;
static __device__ __forceinline__ em_gacc_t em_gacc_zero() { return 0.0f; }
static __device__ __forceinline__ em_gacc_t em_gacc_fdot2(uint32_t a_bits, uint32_t b_bits, em_gacc_t acc) {
    __half2 ha, hb;
    memcpy(&ha, &a_bits, sizeof(ha));
    memcpy(&hb, &b_bits, sizeof(hb));
    float2 fa = __half22float2(ha);
    float2 fb = __half22float2(hb);
    return fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, acc));
}
static __device__ __forceinline__ float em_gacc_to_f32(em_gacc_t acc) { return acc; }
#endif

static __device__ __forceinline__ uint32_t em_pack_h16(uint32_t lo16, uint32_t hi16) {
    return (lo16 & 0xFFFFu) | (hi16 << 16);
}

static __device__ __forceinline__ float em_wave32_sum(float v) {
    v += EM_SHFL_XOR(v, 16);
    v += EM_SHFL_XOR(v, 8);
    v += EM_SHFL_XOR(v, 4);
    v += EM_SHFL_XOR(v, 2);
    v += EM_SHFL_XOR(v, 1);
    return v;
}

static __device__ __forceinline__ uint32_t em_f2_to_h2_u32(float a, float b) {
    __half2 h2 = __halves2half2(__float2half(a), __float2half(b));
    uint32_t u;
    memcpy(&u, &h2, sizeof(u));
    return u;
}

// --- Convert strided F32 rows [B, K] -> contiguous __half [B, K] ---
__global__ void em_f32_to_f16_rows_kernel(
    const float * __restrict__ src,
    __half * __restrict__ dst,
    int K, int B, int64_t nb1_f32)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (k < K && b < B) {
        dst[(size_t)b * K + k] = __float2half(src[(size_t)b * nb1_f32 + k]);
    }
}

// --- Convert strided MoE F32 [K, ne11, n_tokens] -> contiguous __half [n_tokens, ne11, K] ---
__global__ void em_f32_to_f16_moe_kernel(
    const float * __restrict__ src,
    __half * __restrict__ dst,
    int K, int ne11, int n_tokens, int64_t nb1_f32, int64_t nb2_f32)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    const int t = blockIdx.z;
    if (k < K && s < ne11 && t < n_tokens) {
        dst[((size_t)t * ne11 + s) * K + k] = __float2half(src[(size_t)t * nb2_f32 + (size_t)s * nb1_f32 + k]);
    }
}

// --- 1. EM1D_Q4_G64 GEMV (batched B >= 1, R_PER_WAVE=1, 346.1 GB/s @ B=1, 0 spills) ---
__global__ EM_LAUNCH_BOUNDS_GEMV
void em_gemv_q4_g64_kernel(
    const uint32_t * __restrict__ lo4,
    const __half   * __restrict__ scales,
    const uint16_t * __restrict__ lut_u16,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int64_t y_nb1_f32)
{
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.y;
    const int row = blockIdx.x * 8 + wave_id;

    const uint32_t my_lut = (lane < 16) ? (uint32_t)lut_u16[lane] : 0u;

    const int stride_u4 = K >> 5;
    const int stride_sc = K >> 6;
    const int n_iters   = (stride_u4 + 31) >> 5;
    const uint4 * lo4_u4 = reinterpret_cast<const uint4 *>(lo4);
    const uint4 * x_u4   = reinterpret_cast<const uint4 *>(x_h2_all + (size_t)b * (K >> 1));

    float acc = 0.0f;
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool valid = (c < stride_u4);
        const int g = c >> 1;
        const int xb = c << 2;

        uint4 w_row = make_uint4(0, 0, 0, 0);
        float sc_row = 0.0f;
        if (valid && row < N) {
            w_row  = lo4_u4[(size_t)row * stride_u4 + c];
            sc_row = __half2float(scales[(size_t)row * stride_sc + g]);
        }

        em_gacc_t gacc = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint4 xw = valid ? x_u4[xb + w] : make_uint4(0, 0, 0, 0);
            const uint32_t xv[4] = {xw.x, xw.y, xw.z, xw.w};
            const uint32_t v = (w == 0) ? w_row.x : (w == 1) ? w_row.y : (w == 2) ? w_row.z : w_row.w;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t l0 = EM_SHFL(my_lut, (v >> (p * 8))     & 0xFu, 32);
                const uint32_t l1 = EM_SHFL(my_lut, (v >> (p * 8 + 4)) & 0xFu, 32);
                gacc = em_gacc_fdot2(em_pack_h16(l0, l1), xv[p], gacc);
            }
        }
        acc = fmaf(em_gacc_to_f32(gacc), sc_row, acc);
    }

    if (row < N) {
        float s = em_wave32_sum(acc);
        if (lane == 0) y_all[(size_t)b * y_nb1_f32 + row] = s;
    }
}

// --- 1a. EM1D_Q4_G64 BT=4 GEMM (for B=4..16: 4x fewer dense weight reads, 0 extra VRAM) ---
__global__ EM_LAUNCH_BOUNDS
void em_gemm4_q4_g64_kernel(
    const uint32_t * __restrict__ lo4,
    const __half   * __restrict__ scales,
    const uint16_t * __restrict__ lut_u16,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int B, int64_t y_nb1_f32)
{
    const int tid     = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane    = tid & 31;
    const int b0      = blockIdx.y * 4;
    const int cnt     = (B - b0 < 4) ? (B - b0) : 4;
    const int row     = blockIdx.x * 8 + wave_id;

    const uint32_t my_lut = (lane < 16) ? (uint32_t)lut_u16[lane] : 0u;

    const int stride_u4 = K >> 5;
    const int stride_sc = K >> 6;
    const int n_iters   = (stride_u4 + 31) >> 5;
    const uint4 * lo4_u4 = reinterpret_cast<const uint4 *>(lo4);
    const size_t kh2 = (size_t)(K >> 1);
    const uint4 * x0 = reinterpret_cast<const uint4 *>(x_h2_all + (size_t)b0 * kh2);
    const uint4 * x1 = reinterpret_cast<const uint4 *>(x_h2_all + (size_t)(b0 + (cnt > 1 ? 1 : 0)) * kh2);
    const uint4 * x2 = reinterpret_cast<const uint4 *>(x_h2_all + (size_t)(b0 + (cnt > 2 ? 2 : 0)) * kh2);
    const uint4 * x3 = reinterpret_cast<const uint4 *>(x_h2_all + (size_t)(b0 + (cnt > 3 ? 3 : 0)) * kh2);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool vld = (c < stride_u4);
        const int g  = c >> 1;
        const int xb = c << 2;

        uint4 w_row = make_uint4(0, 0, 0, 0);
        float sc_row = 0.0f;
        if (vld && row < N) {
            w_row  = lo4_u4[(size_t)row * stride_u4 + c];
            sc_row = __half2float(scales[(size_t)row * stride_sc + g]);
        }

        em_gacc_t g0 = em_gacc_zero(), g1 = em_gacc_zero(), g2 = em_gacc_zero(), g3 = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint4 xw0 = vld ? x0[xb + w] : make_uint4(0, 0, 0, 0);
            const uint4 xw1 = (vld && cnt > 1) ? x1[xb + w] : make_uint4(0, 0, 0, 0);
            const uint4 xw2 = (vld && cnt > 2) ? x2[xb + w] : make_uint4(0, 0, 0, 0);
            const uint4 xw3 = (vld && cnt > 3) ? x3[xb + w] : make_uint4(0, 0, 0, 0);
            const uint32_t xv0[4] = {xw0.x, xw0.y, xw0.z, xw0.w};
            const uint32_t xv1[4] = {xw1.x, xw1.y, xw1.z, xw1.w};
            const uint32_t xv2[4] = {xw2.x, xw2.y, xw2.z, xw2.w};
            const uint32_t xv3[4] = {xw3.x, xw3.y, xw3.z, xw3.w};
            const uint32_t v = (w == 0) ? w_row.x : (w == 1) ? w_row.y : (w == 2) ? w_row.z : w_row.w;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t wh2 = em_pack_h16(EM_SHFL(my_lut, (v >> (p * 8)) & 0xFu, 32), EM_SHFL(my_lut, (v >> (p * 8 + 4)) & 0xFu, 32));
                g0 = em_gacc_fdot2(wh2, xv0[p], g0);
                if (cnt > 1) g1 = em_gacc_fdot2(wh2, xv1[p], g1);
                if (cnt > 2) g2 = em_gacc_fdot2(wh2, xv2[p], g2);
                if (cnt > 3) g3 = em_gacc_fdot2(wh2, xv3[p], g3);
            }
        }
        acc0 = fmaf(em_gacc_to_f32(g0), sc_row, acc0);
        if (cnt > 1) acc1 = fmaf(em_gacc_to_f32(g1), sc_row, acc1);
        if (cnt > 2) acc2 = fmaf(em_gacc_to_f32(g2), sc_row, acc2);
        if (cnt > 3) acc3 = fmaf(em_gacc_to_f32(g3), sc_row, acc3);
    }

    if (row < N) {
        float s0 = em_wave32_sum(acc0);
        if (lane == 0) y_all[(size_t)b0 * y_nb1_f32 + row] = s0;
        if (cnt > 1) { float s1 = em_wave32_sum(acc1); if (lane == 0) y_all[(size_t)(b0 + 1) * y_nb1_f32 + row] = s1; }
        if (cnt > 2) { float s2 = em_wave32_sum(acc2); if (lane == 0) y_all[(size_t)(b0 + 2) * y_nb1_f32 + row] = s2; }
        if (cnt > 3) { float s3 = em_wave32_sum(acc3); if (lane == 0) y_all[(size_t)(b0 + 3) * y_nb1_f32 + row] = s3; }
    }
}

// --- 1b. EM1D_Q4_G64 tile dequant to FP16 [tile_N, K] for fast rocBLAS prefill ---
__global__ void em_dequant_q4_g64_f16_kernel(
    const uint32_t * __restrict__ lo4,
    const __half   * __restrict__ scales,
    const uint16_t * __restrict__ lut_u16,
    __half         * __restrict__ dst_f16,
    int row_start, int tile_N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_u32 = tile_N * (K >> 3);
    if (idx >= total_u32) return;

    const int u32_per_row = K >> 3;
    const int local_r = idx / u32_per_row;
    const int c_u32   = idx - local_r * u32_per_row;
    const int row     = row_start + local_r;
    const int g       = c_u32 >> 3; // 8 u32s = 64 weights = 1 scale group

    const uint32_t v = lo4[(size_t)row * u32_per_row + c_u32];
    const float sc   = __half2float(scales[(size_t)row * (K >> 6) + g]);
    __half * out     = dst_f16 + (size_t)local_r * K + (c_u32 << 3);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint16_t bits = lut_u16[(v >> (i * 4)) & 0xFu];
        __half h;
        memcpy(&h, &bits, sizeof(__half));
        out[i] = __float2half(__half2float(h) * sc);
    }
}

// --- 1c. EM1D_Q4_G64 GET_ROWS (token_embd.weight -> F32 [K, n_tokens]) ---
__global__ void em_get_rows_q4_g64_kernel(
    const uint32_t * __restrict__ lo4,
    const __half   * __restrict__ scales,
    const uint16_t * __restrict__ lut_u16,
    const int32_t  * __restrict__ ids,
    float          * __restrict__ dst,
    int K, int n_tokens, int64_t dst_nb1_f32)
{
    const int c_u32 = blockIdx.x * blockDim.x + threadIdx.x;
    const int t     = blockIdx.y;
    const int u32_per_row = K >> 3;
    if (c_u32 >= u32_per_row || t >= n_tokens) return;

    const int row = ids[t];
    const int g   = c_u32 >> 3;
    const uint32_t v = lo4[(size_t)row * u32_per_row + c_u32];
    const float sc   = __half2float(scales[(size_t)row * (K >> 6) + g]);
    float * out      = dst + (size_t)t * dst_nb1_f32 + (c_u32 << 3);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint16_t bits = lut_u16[(v >> (i * 4)) & 0xFu];
        __half h;
        memcpy(&h, &bits, sizeof(__half));
        out[i] = __half2float(h) * sc;
    }
}

// --- 2. EM1D_Q3_G64 MoE GEMV (R_PER_WAVE=1, 246.4 GB/s @ B=1, 0 spills) ---
__global__ EM_LAUNCH_BOUNDS_GEMV
void em_gemv_q3_g64_moe_kernel(
    const uint32_t * __restrict__ lo2_all,
    const uint32_t * __restrict__ hi1_all,
    const __half   * __restrict__ scales_all,
    const uint16_t * __restrict__ lut_all,
    const int32_t  * __restrict__ ids,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t ids_nb1_i32, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane = tid & 31;
    const int slot = blockIdx.y;
    const int tok  = blockIdx.z;
    const int e    = ids[(size_t)tok * ids_nb1_i32 + slot];

    const uint32_t my_lut = (lane < 8) ? (uint32_t)lut_all[(size_t)e * 8 + lane] : 0u;

    const int row       = blockIdx.x * 8 + wave_id;
    const int stride_c  = K >> 5;
    const int stride_sc = K >> 6;
    const int n_iters   = (stride_c + 31) >> 5;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2    * lo2_u2 = reinterpret_cast<const uint2 *>(lo2_all) + exp_c_off;
    const uint32_t * hi1_u1 = hi1_all + exp_c_off;
    const __half   * sc_ptr = scales_all + exp_sc_off;

    const int x_slot = (ne11 == 1) ? 0 : slot;
    const uint4 * x_u4 = reinterpret_cast<const uint4 *>(x_h2_all + ((size_t)tok * ne11 + x_slot) * (K >> 1));

    float acc = 0.0f;
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool valid = (c < stride_c);
        const int g  = c >> 1;
        const int xb = c << 2;

        uint2    lo_row = make_uint2(0, 0);
        uint32_t hi_row = 0u;
        float    sc_row = 0.0f;
        if (valid && row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            lo_row = lo2_u2[idx_c];
            hi_row = hi1_u1[idx_c];
            sc_row = __half2float(sc_ptr[(size_t)row * stride_sc + g]);
        }

        em_gacc_t gacc = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint4 xw = valid ? x_u4[xb + w] : make_uint4(0, 0, 0, 0);
            const uint32_t xv[4] = {xw.x, xw.y, xw.z, xw.w};
            const int lo_shift_base = (w & 1) * 16;
            const int hi_shift_base = w * 8;
            const uint32_t lo_w = (w < 2) ? lo_row.x : lo_row.y;
            const uint32_t hi_w = hi_row;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t i0 = ((lo_w >> (lo_shift_base + p * 4))     & 0x3u) | (((hi_w >> (hi_shift_base + p * 2))     & 0x1u) << 2);
                const uint32_t i1 = ((lo_w >> (lo_shift_base + p * 4 + 2)) & 0x3u) | (((hi_w >> (hi_shift_base + p * 2 + 1)) & 0x1u) << 2);
                const uint32_t l0 = EM_SHFL(my_lut, i0, 32);
                const uint32_t l1 = EM_SHFL(my_lut, i1, 32);
                gacc = em_gacc_fdot2(em_pack_h16(l0, l1), xv[p], gacc);
            }
        }
        acc = fmaf(em_gacc_to_f32(gacc), sc_row, acc);
    }

    if (row < N) {
        float s = em_wave32_sum(acc);
        if (lane == 0) {
            const int out_row = is_w13 ? ((row & 1) ? ((N >> 1) + (row >> 1)) : (row >> 1)) : row;
            y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = s;
        }
    }
}

// --- 3. EM2D_K32_G128 MoE GEMV (R_PER_WAVE=1, 288.1 GB/s @ B=1, 0 spills) ---
__global__ EM_LAUNCH_BOUNDS_GEMV
void em_gemv_2d_k32_g128_moe_kernel(
    const uint32_t * __restrict__ lo4_all,
    const uint16_t * __restrict__ hi1_all,
    const __half   * __restrict__ scales_all,
    const uint32_t * __restrict__ lut_h2_all,
    const int32_t  * __restrict__ ids,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t ids_nb1_i32, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane = tid & 31;
    const int slot = blockIdx.y;
    const int tok  = blockIdx.z;
    const int e    = ids[(size_t)tok * ids_nb1_i32 + slot];

    const uint32_t my_lut = lut_h2_all[(size_t)e * 32 + lane];

    const int row       = blockIdx.x * 8 + wave_id;
    const int stride_c  = K >> 5;
    const int stride_sc = K >> 7;
    const int n_iters   = (stride_c + 31) >> 5;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2    * lo4_u2  = reinterpret_cast<const uint2 *>(lo4_all) + exp_c_off;
    const uint16_t * hi1_u16 = hi1_all + exp_c_off;
    const __half   * sc_ptr  = scales_all + exp_sc_off;

    const int x_slot = (ne11 == 1) ? 0 : slot;
    const uint4 * x_u4 = reinterpret_cast<const uint4 *>(x_h2_all + ((size_t)tok * ne11 + x_slot) * (K >> 1));

    float acc = 0.0f;
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool valid = (c < stride_c);
        const int g  = c >> 2;
        const int xb = c << 2;

        uint2    lo_row = make_uint2(0, 0);
        uint32_t hi_row = 0u;
        float    sc_row = 0.0f;
        if (valid && row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            lo_row = lo4_u2[idx_c];
            hi_row = (uint32_t)hi1_u16[idx_c];
            sc_row = __half2float(sc_ptr[(size_t)row * stride_sc + g]);
        }

        em_gacc_t gacc = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint4 xw = valid ? x_u4[xb + w] : make_uint4(0, 0, 0, 0);
            const uint32_t xv[4] = {xw.x, xw.y, xw.z, xw.w};
            const int lo_shift_base = (w & 1) * 16;
            const int hi_shift_base = w * 4;
            const uint32_t lo_w = (w < 2) ? lo_row.x : lo_row.y;
            const uint32_t hi_w = hi_row;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t idx = ((lo_w >> (lo_shift_base + p * 4)) & 0xFu) | (((hi_w >> (hi_shift_base + p)) & 0x1u) << 4);
                gacc = em_gacc_fdot2(EM_SHFL(my_lut, idx, 32), xv[p], gacc);
            }
        }
        acc = fmaf(em_gacc_to_f32(gacc), sc_row, acc);
    }

    if (row < N) {
        float s = em_wave32_sum(acc);
        if (lane == 0) {
            const int out_row = is_w13 ? ((row & 1) ? ((N >> 1) + (row >> 1)) : (row >> 1)) : row;
            y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = s;
        }
    }
}

// --- 3b. EM2D_K16_G128 MoE GEMV (2.125 bpw, NO hi1 plane, 8 GB tier) ---
__global__ EM_LAUNCH_BOUNDS_GEMV
void em_gemv_2d_k16_g128_moe_kernel(
    const uint32_t * __restrict__ lo4_all,
    const __half   * __restrict__ scales_all,
    const uint32_t * __restrict__ lut_h2_all,
    const int32_t  * __restrict__ ids,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t ids_nb1_i32, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane = tid & 31;
    const int slot = blockIdx.y;
    const int tok  = blockIdx.z;
    const int e    = ids[(size_t)tok * ids_nb1_i32 + slot];

    const uint32_t my_lut = lut_h2_all[(size_t)e * 16 + (lane & 15)];

    const int row       = blockIdx.x * 8 + wave_id;
    const int stride_c  = K >> 5;
    const int stride_sc = K >> 7;
    const int n_iters   = (stride_c + 31) >> 5;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2  * lo4_u2  = reinterpret_cast<const uint2 *>(lo4_all) + exp_c_off;
    const __half * sc_ptr  = scales_all + exp_sc_off;

    const int x_slot = (ne11 == 1) ? 0 : slot;
    const uint4 * x_u4 = reinterpret_cast<const uint4 *>(x_h2_all + ((size_t)tok * ne11 + x_slot) * (K >> 1));

    float acc = 0.0f;
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool valid = (c < stride_c);
        const int g  = c >> 2;
        const int xb = c << 2;

        uint2 lo_row = make_uint2(0, 0);
        float sc_row = 0.0f;
        if (valid && row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            lo_row = lo4_u2[idx_c];
            sc_row = __half2float(sc_ptr[(size_t)row * stride_sc + g]);
        }

        em_gacc_t gacc = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint4 xw = valid ? x_u4[xb + w] : make_uint4(0, 0, 0, 0);
            const uint32_t xv[4] = {xw.x, xw.y, xw.z, xw.w};
            const int lo_shift_base = (w & 1) * 16;
            const uint32_t lo_w = (w < 2) ? lo_row.x : lo_row.y;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t idx = (lo_w >> (lo_shift_base + p * 4)) & 0xFu;
                gacc = em_gacc_fdot2(EM_SHFL(my_lut, idx, 32), xv[p], gacc);
            }
        }
        acc = fmaf(em_gacc_to_f32(gacc), sc_row, acc);
    }

    if (row < N) {
        float s = em_wave32_sum(acc);
        if (lane == 0) {
            const int out_row = is_w13 ? ((row & 1) ? ((N >> 1) + (row >> 1)) : (row >> 1)) : row;
            y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = s;
        }
    }
}

// --- 4. Single-Launch Zero-Sync GPU-Grouped BT=8 MoE Prefill/Verify Kernels ---
__global__ void em_moe_build_tiles_kernel(
    const int32_t  * __restrict__ ids,
    uint32_t       * __restrict__ sorted_pairs,
    uint2          * __restrict__ tiles,
    int32_t        * __restrict__ d_num_tiles,
    int n_tokens, int n_used, int n_experts, int64_t ids_nb1_i32)
{
    __shared__ int s_counts[256];
    __shared__ int s_offsets[256];
    __shared__ int s_tile_off[256];

    const int tid = threadIdx.x;
    const int P   = n_tokens * n_used;

    s_counts[tid] = 0;
    __syncthreads();

    for (int p = tid; p < P; p += 256) {
        const int tok  = p / n_used;
        const int slot = p - tok * n_used;
        const int e    = ids[(size_t)tok * ids_nb1_i32 + slot];
        if (e >= 0 && e < n_experts) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int acc_p = 0;
        int acc_t = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int c   = s_counts[e];
            s_offsets[e]  = acc_p;
            s_tile_off[e] = acc_t;
            acc_p += c;
            acc_t += (c + 7) >> 3;
            s_counts[e]   = s_offsets[e];
        }
        *d_num_tiles = acc_t;
    }
    __syncthreads();

    for (int p = tid; p < P; p += 256) {
        const int tok  = p / n_used;
        const int slot = p - tok * n_used;
        const int e    = ids[(size_t)tok * ids_nb1_i32 + slot];
        if (e >= 0 && e < n_experts) {
            const int dst_i = atomicAdd(&s_counts[e], 1);
            sorted_pairs[dst_i] = ((uint32_t)tok << 8) | (uint32_t)slot;
        }
    }
    __syncthreads();

    if (tid < n_experts) {
        const int start = s_offsets[tid];
        const int cnt   = s_counts[tid] - start;
        const int tbase = s_tile_off[tid];
        for (int j = 0; j < cnt; j += 8) {
            const int c = (cnt - j < 8) ? (cnt - j) : 8;
            tiles[tbase + (j >> 3)] = make_uint2(((uint32_t)tid << 4) | (uint32_t)c, (uint32_t)(start + j));
        }
    }
}

__global__ EM_LAUNCH_BOUNDS
void em_gemm8_q3_g64_moe_kernel(
    const uint32_t * __restrict__ lo2_all,
    const uint32_t * __restrict__ hi1_all,
    const __half   * __restrict__ scales_all,
    const uint16_t * __restrict__ lut_all,
    const uint32_t * __restrict__ sorted_pairs,
    const uint2    * __restrict__ tiles,
    const int32_t  * __restrict__ d_num_tiles,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
    const int tile_id = blockIdx.y;
    if (tile_id >= d_num_tiles[0]) return;

    const int tid     = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane    = tid & 31;

    const uint2 ti    = tiles[tile_id];
    const int e       = (int)(ti.x >> 4);
    const int cnt     = (int)(ti.x & 0xFu);
    const int pstart  = (int)ti.y;

    __shared__ uint32_t s_pairs[8];
    __shared__ size_t   s_xbase[8];
    if (tid < cnt) {
        const uint32_t pr = sorted_pairs[pstart + tid];
        s_pairs[tid] = pr;
        const int tok  = (int)(pr >> 8);
        const int slot = (int)(pr & 0xFFu);
        s_xbase[tid] = ((size_t)tok * ne11 + ((ne11 == 1) ? 0 : slot)) * (size_t)(K >> 3);
    }
    __syncthreads();

    const uint32_t my_lut = (lane < 8) ? (uint32_t)lut_all[(size_t)e * 8 + lane] : 0u;

    const int row       = blockIdx.x * 8 + wave_id;
    const int stride_c  = K >> 5;
    const int stride_sc = K >> 6;
    const int n_iters   = (stride_c + 31) >> 5;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2    * lo2_u2 = reinterpret_cast<const uint2 *>(lo2_all) + exp_c_off;
    const uint32_t * hi1_u1 = hi1_all + exp_c_off;
    const __half   * sc_ptr = scales_all + exp_sc_off;
    const uint4    * x_u4   = reinterpret_cast<const uint4 *>(x_h2_all);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool valid = (c < stride_c);
        const int g  = c >> 1;
        const int xb = c << 2;

        uint2    lo_row = make_uint2(0, 0);
        uint32_t hi_row = 0u;
        float    sc_row = 0.0f;
        if (valid && row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            lo_row = lo2_u2[idx_c];
            hi_row = hi1_u1[idx_c];
            sc_row = __half2float(sc_ptr[(size_t)row * stride_sc + g]);
        }

        em_gacc_t gv[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) gv[b] = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int lo_shift_base = (w & 1) * 16;
            const int hi_shift_base = w * 8;
            const uint32_t lo_w = (w < 2) ? lo_row.x : lo_row.y;
            const uint32_t hi_w = hi_row;
            uint32_t wh2[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t i0 = ((lo_w >> (lo_shift_base + p * 4))     & 0x3u) | (((hi_w >> (hi_shift_base + p * 2))     & 0x1u) << 2);
                const uint32_t i1 = ((lo_w >> (lo_shift_base + p * 4 + 2)) & 0x3u) | (((hi_w >> (hi_shift_base + p * 2 + 1)) & 0x1u) << 2);
                wh2[p] = em_pack_h16(EM_SHFL(my_lut, i0, 32), EM_SHFL(my_lut, i1, 32));
            }
            if (valid) {
                for (int b = 0; b < cnt; ++b) {
                    const uint4 xw = x_u4[s_xbase[b] + xb + w];
                    em_gacc_t gb = gv[b];
                    gb = em_gacc_fdot2(wh2[0], xw.x, gb);
                    gb = em_gacc_fdot2(wh2[1], xw.y, gb);
                    gb = em_gacc_fdot2(wh2[2], xw.z, gb);
                    gb = em_gacc_fdot2(wh2[3], xw.w, gb);
                    gv[b] = gb;
                }
            }
        }
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b < cnt) acc[b] = fmaf(em_gacc_to_f32(gv[b]), sc_row, acc[b]);
        }
    }

    if (row < N) {
        const int out_row = is_w13 ? ((row & 1) ? ((N >> 1) + (row >> 1)) : (row >> 1)) : row;
        for (int b = 0; b < cnt; ++b) {
            const float sb = em_wave32_sum(acc[b]);
            if (lane == 0) {
                const uint32_t pr = s_pairs[b];
                const int tok  = (int)(pr >> 8);
                const int slot = (int)(pr & 0xFFu);
                y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = sb;
            }
        }
    }
}

__global__ EM_LAUNCH_BOUNDS
void em_gemm8_2d_k32_g128_moe_kernel(
    const uint32_t * __restrict__ lo4_all,
    const uint16_t * __restrict__ hi1_all,
    const __half   * __restrict__ scales_all,
    const uint32_t * __restrict__ lut_h2_all,
    const uint32_t * __restrict__ sorted_pairs,
    const uint2    * __restrict__ tiles,
    const int32_t  * __restrict__ d_num_tiles,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
    const int tile_id = blockIdx.y;
    if (tile_id >= d_num_tiles[0]) return;

    const int tid     = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane    = tid & 31;

    const uint2 ti    = tiles[tile_id];
    const int e       = (int)(ti.x >> 4);
    const int cnt     = (int)(ti.x & 0xFu);
    const int pstart  = (int)ti.y;

    __shared__ uint32_t s_pairs[8];
    __shared__ size_t   s_xbase[8];
    if (tid < cnt) {
        const uint32_t pr = sorted_pairs[pstart + tid];
        s_pairs[tid] = pr;
        const int tok  = (int)(pr >> 8);
        const int slot = (int)(pr & 0xFFu);
        s_xbase[tid] = ((size_t)tok * ne11 + ((ne11 == 1) ? 0 : slot)) * (size_t)(K >> 3);
    }
    __syncthreads();

    const uint32_t my_lut = lut_h2_all[(size_t)e * 32 + lane];

    const int row       = blockIdx.x * 8 + wave_id;
    const int stride_c  = K >> 5;
    const int stride_sc = K >> 7;
    const int n_iters   = (stride_c + 31) >> 5;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2    * lo4_u2  = reinterpret_cast<const uint2 *>(lo4_all) + exp_c_off;
    const uint16_t * hi1_u16 = hi1_all + exp_c_off;
    const __half   * sc_ptr  = scales_all + exp_sc_off;
    const uint4    * x_u4    = reinterpret_cast<const uint4 *>(x_h2_all);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool valid = (c < stride_c);
        const int g  = c >> 2;
        const int xb = c << 2;

        uint2    lo_row = make_uint2(0, 0);
        uint32_t hi_row = 0u;
        float    sc_row = 0.0f;
        if (valid && row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            lo_row = lo4_u2[idx_c];
            hi_row = (uint32_t)hi1_u16[idx_c];
            sc_row = __half2float(sc_ptr[(size_t)row * stride_sc + g]);
        }

        em_gacc_t gv[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) gv[b] = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int lo_shift_base = (w & 1) * 16;
            const int hi_shift_base = w * 4;
            const uint32_t lo_w = (w < 2) ? lo_row.x : lo_row.y;
            const uint32_t hi_w = hi_row;
            uint32_t wh2[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t idx = ((lo_w >> (lo_shift_base + p * 4)) & 0xFu) | (((hi_w >> (hi_shift_base + p)) & 0x1u) << 4);
                wh2[p] = EM_SHFL(my_lut, idx, 32);
            }
            if (valid) {
                for (int b = 0; b < cnt; ++b) {
                    const uint4 xw = x_u4[s_xbase[b] + xb + w];
                    em_gacc_t gb = gv[b];
                    gb = em_gacc_fdot2(wh2[0], xw.x, gb);
                    gb = em_gacc_fdot2(wh2[1], xw.y, gb);
                    gb = em_gacc_fdot2(wh2[2], xw.z, gb);
                    gb = em_gacc_fdot2(wh2[3], xw.w, gb);
                    gv[b] = gb;
                }
            }
        }
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b < cnt) acc[b] = fmaf(em_gacc_to_f32(gv[b]), sc_row, acc[b]);
        }
    }

    if (row < N) {
        const int out_row = is_w13 ? ((row & 1) ? ((N >> 1) + (row >> 1)) : (row >> 1)) : row;
        for (int b = 0; b < cnt; ++b) {
            const float sb = em_wave32_sum(acc[b]);
            if (lane == 0) {
                const uint32_t pr = s_pairs[b];
                const int tok  = (int)(pr >> 8);
                const int slot = (int)(pr & 0xFFu);
                y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = sb;
            }
        }
    }
}

__global__ EM_LAUNCH_BOUNDS_GEMV
void em_gemm8_2d_k16_g128_moe_kernel(
    const uint32_t * __restrict__ lo4_all,
    const __half   * __restrict__ scales_all,
    const uint32_t * __restrict__ lut_h2_all,
    const uint32_t * __restrict__ sorted_pairs,
    const uint2    * __restrict__ tiles,
    const int32_t  * __restrict__ d_num_tiles,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
    const int tile_id = blockIdx.y;
    if (tile_id >= d_num_tiles[0]) return;

    const int tid     = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane    = tid & 31;

    const uint2 ti    = tiles[tile_id];
    const int e       = (int)(ti.x >> 4);
    const int cnt     = (int)(ti.x & 0xFu);
    const int pstart  = (int)ti.y;

    __shared__ uint32_t s_pairs[8];
    __shared__ size_t   s_xbase[8];
    if (tid < cnt) {
        const uint32_t pr = sorted_pairs[pstart + tid];
        s_pairs[tid] = pr;
        const int tok  = (int)(pr >> 8);
        const int slot = (int)(pr & 0xFFu);
        s_xbase[tid] = ((size_t)tok * ne11 + ((ne11 == 1) ? 0 : slot)) * (size_t)(K >> 3);
    }
    __syncthreads();

    const uint32_t my_lut = lut_h2_all[(size_t)e * 16 + (lane & 15)];

    const int row       = blockIdx.x * 8 + wave_id;
    const int stride_c  = K >> 5;
    const int stride_sc = K >> 7;
    const int n_iters   = (stride_c + 31) >> 5;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2  * lo4_u2  = reinterpret_cast<const uint2 *>(lo4_all) + exp_c_off;
    const __half * sc_ptr  = scales_all + exp_sc_off;
    const uint4  * x_u4    = reinterpret_cast<const uint4 *>(x_h2_all);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int it = 0; it < n_iters; ++it) {
        const int c = (it << 5) + lane;
        const bool valid = (c < stride_c);
        const int g  = c >> 2;
        const int xb = c << 2;

        uint2 lo_row = make_uint2(0, 0);
        float sc_row = 0.0f;
        if (valid && row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            lo_row = lo4_u2[idx_c];
            sc_row = __half2float(sc_ptr[(size_t)row * stride_sc + g]);
        }

        em_gacc_t gv[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) gv[b] = em_gacc_zero();
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int lo_shift_base = (w & 1) * 16;
            const uint32_t lo_w = (w < 2) ? lo_row.x : lo_row.y;
            uint32_t wh2[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t idx = (lo_w >> (lo_shift_base + p * 4)) & 0xFu;
                wh2[p] = EM_SHFL(my_lut, idx, 32);
            }
            if (valid) {
                for (int b = 0; b < cnt; ++b) {
                    const uint4 xw = x_u4[s_xbase[b] + xb + w];
                    em_gacc_t gb = gv[b];
                    gb = em_gacc_fdot2(wh2[0], xw.x, gb);
                    gb = em_gacc_fdot2(wh2[1], xw.y, gb);
                    gb = em_gacc_fdot2(wh2[2], xw.z, gb);
                    gb = em_gacc_fdot2(wh2[3], xw.w, gb);
                    gv[b] = gb;
                }
            }
        }
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b < cnt) acc[b] = fmaf(em_gacc_to_f32(gv[b]), sc_row, acc[b]);
        }
    }

    if (row < N) {
        const int out_row = is_w13 ? ((row & 1) ? ((N >> 1) + (row >> 1)) : (row >> 1)) : row;
        for (int b = 0; b < cnt; ++b) {
            const float sb = em_wave32_sum(acc[b]);
            if (lane == 0) {
                const uint32_t pr = s_pairs[b];
                const int tok  = (int)(pr >> 8);
                const int slot = (int)(pr & 0xFFu);
                y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = sb;
            }
        }
    }
}

#ifndef GGML_USE_HIP
// --- 5. CUDA Tensor Core (WMMA 16x16x16 FP16->FP32) BT=16 MoE Prefill Kernels (sm_75+) ---
__global__ void em_moe_build_tiles16_kernel(
    const int32_t  * __restrict__ ids,
    uint32_t       * __restrict__ sorted_pairs,
    uint2          * __restrict__ tiles,
    int32_t        * __restrict__ d_num_tiles,
    int n_tokens, int n_used, int n_experts, int64_t ids_nb1_i32)
{
    __shared__ int s_counts[256];
    __shared__ int s_offsets[256];
    __shared__ int s_tile_off[256];

    const int tid = threadIdx.x;
    const int P   = n_tokens * n_used;

    s_counts[tid] = 0;
    __syncthreads();

    for (int p = tid; p < P; p += 256) {
        const int tok  = p / n_used;
        const int slot = p - tok * n_used;
        const int e    = ids[(size_t)tok * ids_nb1_i32 + slot];
        if (e >= 0 && e < n_experts) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int acc_p = 0;
        int acc_t = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int c   = s_counts[e];
            s_offsets[e]  = acc_p;
            s_tile_off[e] = acc_t;
            acc_p += c;
            acc_t += (c + 15) >> 4;
            s_counts[e]   = s_offsets[e];
        }
        *d_num_tiles = acc_t;
    }
    __syncthreads();

    for (int p = tid; p < P; p += 256) {
        const int tok  = p / n_used;
        const int slot = p - tok * n_used;
        const int e    = ids[(size_t)tok * ids_nb1_i32 + slot];
        if (e >= 0 && e < n_experts) {
            const int dst_i = atomicAdd(&s_counts[e], 1);
            sorted_pairs[dst_i] = ((uint32_t)tok << 8) | (uint32_t)slot;
        }
    }
    __syncthreads();

    if (tid < n_experts) {
        const int start = s_offsets[tid];
        const int cnt   = s_counts[tid] - start;
        const int tbase = s_tile_off[tid];
        for (int j = 0; j < cnt; j += 16) {
            const int c = (cnt - j < 16) ? (cnt - j) : 16;
            tiles[tbase + (j >> 4)] = make_uint2(((uint32_t)tid << 5) | (uint32_t)c, (uint32_t)(start + j));
        }
    }
}

__global__ __launch_bounds__(256)
void em_gemm16_wmma_q3_g64_moe_kernel(
    const uint32_t * __restrict__ lo2_all,
    const uint32_t * __restrict__ hi1_all,
    const __half   * __restrict__ scales_all,
    const uint16_t * __restrict__ lut_all,
    const uint32_t * __restrict__ sorted_pairs,
    const uint2    * __restrict__ tiles,
    const int32_t  * __restrict__ d_num_tiles,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    const int tile_id = blockIdx.y;
    if (tile_id >= d_num_tiles[0]) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const uint2 ti    = tiles[tile_id];
    const int e       = (int)(ti.x >> 5);
    const int cnt     = (int)(ti.x & 0x1Fu);
    const int pstart  = (int)ti.y;

    __shared__ uint32_t s_pairs[16];
    __shared__ size_t   s_xbase[16];
    alignas(16) __shared__ union {
        uint16_t w[128][40];
        float    c[8][256];
    } s_mem;
    alignas(16) __shared__ uint16_t s_X[16][40];

    if (tid < cnt) {
        const uint32_t pr = sorted_pairs[pstart + tid];
        s_pairs[tid] = pr;
        const int tok  = (int)(pr >> 8);
        const int slot = (int)(pr & 0xFFu);
        s_xbase[tid] = ((size_t)tok * ne11 + ((ne11 == 1) ? 0 : slot)) * (size_t)(K >> 3);
    }
    __syncthreads();

    const uint32_t my_lut = (lane < 8) ? (uint32_t)lut_all[(size_t)e * 8 + lane] : 0u;

    const int warp_row_base = blockIdx.x * 128 + warp_id * 16;
    const int r_in_warp     = lane >> 1;
    const int half_sel      = lane & 1;
    const int row           = warp_row_base + r_in_warp;

    const int stride_c  = K >> 5;
    const int stride_sc = K >> 6;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2    * lo2_u2 = reinterpret_cast<const uint2 *>(lo2_all) + exp_c_off;
    const uint32_t * hi1_u1 = hi1_all + exp_c_off;
    const __half   * sc_ptr = scales_all + exp_sc_off;
    const uint4    * x_u4   = reinterpret_cast<const uint4 *>(x_h2_all);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const uint4 zero_u4 = make_uint4(0, 0, 0, 0);

    for (int c = 0; c < stride_c; ++c) {
        if (tid < 64) {
            const int b = tid >> 2;
            const int v = tid & 3;
            const uint4 xw = (b < cnt) ? x_u4[s_xbase[b] + (c << 2) + v] : zero_u4;
            *reinterpret_cast<uint4 *>(&s_X[b][v << 3]) = xw;
        }

        uint32_t lo_w = 0u;
        uint32_t hi_w = 0u;
        __half2 sc_h2 = __float2half2_rn(0.0f);
        if (row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            const uint2 lo_row = lo2_u2[idx_c];
            lo_w = half_sel ? lo_row.y : lo_row.x;
            hi_w = hi1_u1[idx_c];
            sc_h2 = __half2half2(sc_ptr[(size_t)row * stride_sc + (c >> 1)]);
        }

        __half2 * w_dst = reinterpret_cast<__half2 *>(&s_mem.w[warp_id * 16 + r_in_warp][half_sel * 16]);
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            const int lo_shift_base = w * 16;
            const int hi_shift_base = (half_sel * 2 + w) * 8;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t i0 = ((lo_w >> (lo_shift_base + p * 4))     & 0x3u) | (((hi_w >> (hi_shift_base + p * 2))     & 0x1u) << 2);
                const uint32_t i1 = ((lo_w >> (lo_shift_base + p * 4 + 2)) & 0x3u) | (((hi_w >> (hi_shift_base + p * 2 + 1)) & 0x1u) << 2);
                const uint32_t wh2_u = em_pack_h16(EM_SHFL(my_lut, i0, 32), EM_SHFL(my_lut, i1, 32));
                __half2 wh2;
                memcpy(&wh2, &wh2_u, sizeof(wh2));
                w_dst[w * 4 + p] = __hmul2(wh2, sc_h2);
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

        const __half * w_ptr = reinterpret_cast<const __half *>(&s_mem.w[warp_id * 16][0]);
        const __half * x_ptr = reinterpret_cast<const __half *>(&s_X[0][0]);

        wmma::load_matrix_sync(a_frag, w_ptr, 40);
        wmma::load_matrix_sync(b_frag, x_ptr, 40);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        wmma::load_matrix_sync(a_frag, w_ptr + 16, 40);
        wmma::load_matrix_sync(b_frag, x_ptr + 16, 40);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(s_mem.c[warp_id], c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int idx = lane; idx < 256; idx += 32) {
        const int r = idx >> 4;
        const int b = idx & 15;
        const int cur_row = warp_row_base + r;
        if (b < cnt && cur_row < N) {
            const int out_row = is_w13 ? ((cur_row & 1) ? ((N >> 1) + (cur_row >> 1)) : (cur_row >> 1)) : cur_row;
            const uint32_t pr = s_pairs[b];
            const int tok  = (int)(pr >> 8);
            const int slot = (int)(pr & 0xFFu);
            y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = s_mem.c[warp_id][idx];
        }
    }
#endif
}

__global__ __launch_bounds__(256)
void em_gemm16_wmma_2d_k32_g128_moe_kernel(
    const uint32_t * __restrict__ lo4_all,
    const uint16_t * __restrict__ hi1_all,
    const __half   * __restrict__ scales_all,
    const uint32_t * __restrict__ lut_h2_all,
    const uint32_t * __restrict__ sorted_pairs,
    const uint2    * __restrict__ tiles,
    const int32_t  * __restrict__ d_num_tiles,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    const int tile_id = blockIdx.y;
    if (tile_id >= d_num_tiles[0]) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const uint2 ti    = tiles[tile_id];
    const int e       = (int)(ti.x >> 5);
    const int cnt     = (int)(ti.x & 0x1Fu);
    const int pstart  = (int)ti.y;

    __shared__ uint32_t s_pairs[16];
    __shared__ size_t   s_xbase[16];
    alignas(16) __shared__ union {
        uint16_t w[128][40];
        float    c[8][256];
    } s_mem;
    alignas(16) __shared__ uint16_t s_X[16][40];

    if (tid < cnt) {
        const uint32_t pr = sorted_pairs[pstart + tid];
        s_pairs[tid] = pr;
        const int tok  = (int)(pr >> 8);
        const int slot = (int)(pr & 0xFFu);
        s_xbase[tid] = ((size_t)tok * ne11 + ((ne11 == 1) ? 0 : slot)) * (size_t)(K >> 3);
    }
    __syncthreads();

    const uint32_t my_lut = lut_h2_all[(size_t)e * 32 + lane];

    const int warp_row_base = blockIdx.x * 128 + warp_id * 16;
    const int r_in_warp     = lane >> 1;
    const int half_sel      = lane & 1;
    const int row           = warp_row_base + r_in_warp;

    const int stride_c  = K >> 5;
    const int stride_sc = K >> 7;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2    * lo4_u2  = reinterpret_cast<const uint2 *>(lo4_all) + exp_c_off;
    const uint16_t * hi1_u16 = hi1_all + exp_c_off;
    const __half   * sc_ptr  = scales_all + exp_sc_off;
    const uint4    * x_u4    = reinterpret_cast<const uint4 *>(x_h2_all);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const uint4 zero_u4 = make_uint4(0, 0, 0, 0);

    for (int c = 0; c < stride_c; ++c) {
        if (tid < 64) {
            const int b = tid >> 2;
            const int v = tid & 3;
            const uint4 xw = (b < cnt) ? x_u4[s_xbase[b] + (c << 2) + v] : zero_u4;
            *reinterpret_cast<uint4 *>(&s_X[b][v << 3]) = xw;
        }

        uint32_t lo_w = 0u;
        uint32_t hi_w = 0u;
        __half2 sc_h2 = __float2half2_rn(0.0f);
        if (row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            const uint2 lo_row = lo4_u2[idx_c];
            lo_w = half_sel ? lo_row.y : lo_row.x;
            hi_w = (uint32_t)hi1_u16[idx_c];
            sc_h2 = __half2half2(sc_ptr[(size_t)row * stride_sc + (c >> 2)]);
        }

        __half2 * w_dst = reinterpret_cast<__half2 *>(&s_mem.w[warp_id * 16 + r_in_warp][half_sel * 16]);
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            const int lo_shift_base = w * 16;
            const int hi_shift_base = (half_sel * 2 + w) * 4;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t idx = ((lo_w >> (lo_shift_base + p * 4)) & 0xFu) | (((hi_w >> (hi_shift_base + p)) & 0x1u) << 4);
                const uint32_t wh2_u = EM_SHFL(my_lut, idx, 32);
                __half2 wh2;
                memcpy(&wh2, &wh2_u, sizeof(wh2));
                w_dst[w * 4 + p] = __hmul2(wh2, sc_h2);
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

        const __half * w_ptr = reinterpret_cast<const __half *>(&s_mem.w[warp_id * 16][0]);
        const __half * x_ptr = reinterpret_cast<const __half *>(&s_X[0][0]);

        wmma::load_matrix_sync(a_frag, w_ptr, 40);
        wmma::load_matrix_sync(b_frag, x_ptr, 40);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        wmma::load_matrix_sync(a_frag, w_ptr + 16, 40);
        wmma::load_matrix_sync(b_frag, x_ptr + 16, 40);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(s_mem.c[warp_id], c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int idx = lane; idx < 256; idx += 32) {
        const int r = idx >> 4;
        const int b = idx & 15;
        const int cur_row = warp_row_base + r;
        if (b < cnt && cur_row < N) {
            const int out_row = is_w13 ? ((cur_row & 1) ? ((N >> 1) + (cur_row >> 1)) : (cur_row >> 1)) : cur_row;
            const uint32_t pr = s_pairs[b];
            const int tok  = (int)(pr >> 8);
            const int slot = (int)(pr & 0xFFu);
            y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = s_mem.c[warp_id][idx];
        }
    }
#endif
}

__global__ __launch_bounds__(256, 2)
void em_gemm16_wmma_2d_k16_g128_moe_kernel(
    const uint32_t * __restrict__ lo4_all,
    const __half   * __restrict__ scales_all,
    const uint32_t * __restrict__ lut_h2_all,
    const uint32_t * __restrict__ sorted_pairs,
    const uint2    * __restrict__ tiles,
    const int32_t  * __restrict__ d_num_tiles,
    const uint32_t * __restrict__ x_h2_all,
    float          * __restrict__ y_all,
    int N, int K, int ne11, int is_w13, int64_t y_nb1_f32, int64_t y_nb2_f32)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    const int tile_id = blockIdx.y;
    if (tile_id >= d_num_tiles[0]) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const uint2 ti    = tiles[tile_id];
    const int e       = (int)(ti.x >> 5);
    const int cnt     = (int)(ti.x & 0x1Fu);
    const int pstart  = (int)ti.y;

    __shared__ uint32_t s_pairs[16];
    __shared__ size_t   s_xbase[16];
    alignas(16) __shared__ union {
        uint16_t w[128][40];
        float    c[8][256];
    } s_mem;
    alignas(16) __shared__ uint16_t s_X[16][40];

    if (tid < cnt) {
        const uint32_t pr = sorted_pairs[pstart + tid];
        s_pairs[tid] = pr;
        const int tok  = (int)(pr >> 8);
        const int slot = (int)(pr & 0xFFu);
        s_xbase[tid] = ((size_t)tok * ne11 + ((ne11 == 1) ? 0 : slot)) * (size_t)(K >> 3);
    }
    __syncthreads();

    const uint32_t my_lut = lut_h2_all[(size_t)e * 16 + (lane & 15)];

    const int warp_row_base = blockIdx.x * 128 + warp_id * 16;
    const int r_in_warp     = lane >> 1;
    const int half_sel      = lane & 1;
    const int row           = warp_row_base + r_in_warp;

    const int stride_c  = K >> 5;
    const int stride_sc = K >> 7;

    const size_t exp_c_off  = (size_t)e * N * stride_c;
    const size_t exp_sc_off = (size_t)e * N * stride_sc;
    const uint2  * lo4_u2  = reinterpret_cast<const uint2 *>(lo4_all) + exp_c_off;
    const __half * sc_ptr  = scales_all + exp_sc_off;
    const uint4  * x_u4    = reinterpret_cast<const uint4 *>(x_h2_all);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const uint4 zero_u4 = make_uint4(0, 0, 0, 0);

    for (int c = 0; c < stride_c; ++c) {
        if (tid < 64) {
            const int b = tid >> 2;
            const int v = tid & 3;
            const uint4 xw = (b < cnt) ? x_u4[s_xbase[b] + (c << 2) + v] : zero_u4;
            *reinterpret_cast<uint4 *>(&s_X[b][v << 3]) = xw;
        }

        uint32_t lo_w = 0u;
        __half2 sc_h2 = __float2half2_rn(0.0f);
        if (row < N) {
            const size_t idx_c = (size_t)row * stride_c + c;
            const uint2 lo_row = lo4_u2[idx_c];
            lo_w = half_sel ? lo_row.y : lo_row.x;
            sc_h2 = __half2half2(sc_ptr[(size_t)row * stride_sc + (c >> 2)]);
        }

        __half2 * w_dst = reinterpret_cast<__half2 *>(&s_mem.w[warp_id * 16 + r_in_warp][half_sel * 16]);
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            const int lo_shift_base = w * 16;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t idx = (lo_w >> (lo_shift_base + p * 4)) & 0xFu;
                const uint32_t wh2_u = EM_SHFL(my_lut, idx, 32);
                __half2 wh2;
                memcpy(&wh2, &wh2_u, sizeof(wh2));
                w_dst[w * 4 + p] = __hmul2(wh2, sc_h2);
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

        const __half * w_ptr = reinterpret_cast<const __half *>(&s_mem.w[warp_id * 16][0]);
        const __half * x_ptr = reinterpret_cast<const __half *>(&s_X[0][0]);

        wmma::load_matrix_sync(a_frag, w_ptr, 40);
        wmma::load_matrix_sync(b_frag, x_ptr, 40);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        wmma::load_matrix_sync(a_frag, w_ptr + 16, 40);
        wmma::load_matrix_sync(b_frag, x_ptr + 16, 40);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(s_mem.c[warp_id], c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int idx = lane; idx < 256; idx += 32) {
        const int r = idx >> 4;
        const int b = idx & 15;
        const int cur_row = warp_row_base + r;
        if (b < cnt && cur_row < N) {
            const int out_row = is_w13 ? ((cur_row & 1) ? ((N >> 1) + (cur_row >> 1)) : (cur_row >> 1)) : cur_row;
            const uint32_t pr = s_pairs[b];
            const int tok  = (int)(pr >> 8);
            const int slot = (int)(pr & 0xFFu);
            y_all[(size_t)tok * y_nb2_f32 + (size_t)slot * y_nb1_f32 + out_row] = s_mem.c[warp_id][idx];
        }
    }
#endif
}
#endif // !GGML_USE_HIP

// ============================================================================
// ggml-cuda Dispatch Hooks
// ============================================================================
bool ggml_cuda_emeans_has(const ggml_tensor * src0) {
    return emns_lookup(src0) != nullptr;
}

bool ggml_cuda_emeans_try_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const EmTensor * em = emns_lookup(src0);
    if (!em) return false;

    GGML_ASSERT(em->fmt == EM_FMT_EM1D_Q4_G64);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    const int K = (int)src0->ne[0];
    const int N = (int)src0->ne[1];
    const int B = (int)(src1->ne[1] * src1->ne[2] * src1->ne[3]);
    const int64_t x_nb1_f32 = src1->nb[1] / sizeof(float);
    const int64_t y_nb1_f32 = dst->nb[1]  / sizeof(float);
    hipStream_t stream = ctx.stream();

    // Cast src1 (F32) -> contiguous FP16 scratch
    ggml_cuda_pool_alloc<__half> x_f16(ctx.pool(), (size_t)B * K);
    {
        dim3 block(256);
        dim3 grid((K + 255) / 256, B);
        em_f32_to_f16_rows_kernel<<<grid, block, 0, stream>>>(
            (const float *)src1->data, x_f16.get(), K, B, x_nb1_f32);
    }

    const uint32_t * lo4 = em->ptr<uint32_t>("lo4");
    const __half   * sc  = em->ptr<__half>("scales");
    const uint16_t * lut = em->ptr<uint16_t>("lut");

    if (B <= 3) {
        // Direct Wave32 packed-LUT GEMV (346.1 GB/s @ B=1, 0 spills)
        dim3 block(256);
        dim3 grid((N + 7) / 8, B);
        em_gemv_q4_g64_kernel<<<grid, block, 0, stream>>>(
            lo4, sc, lut,
            reinterpret_cast<const uint32_t *>(x_f16.get()),
            (float *)dst->data,
            N, K, y_nb1_f32);
    } else if (B <= 16) {
        // Batched BT=4 Wave32 GEMM (4x fewer dense weight reads for speculative verification, 0 extra VRAM)
        dim3 block(256);
        dim3 grid((N + 7) / 8, (B + 3) / 4);
        em_gemm4_q4_g64_kernel<<<grid, block, 0, stream>>>(
            lo4, sc, lut,
            reinterpret_cast<const uint32_t *>(x_f16.get()),
            (float *)dst->data,
            N, K, B, y_nb1_f32);
    } else {
        // Fast prefill: chunked tile dequant to FP16 + rocBLAS GEMM (<= 18 MiB scratch)
        const int TILE_N = (N < 4096) ? N : 4096;
        ggml_cuda_pool_alloc<__half> w_f16(ctx.pool(), (size_t)TILE_N * K);
        cublasHandle_t handle = ctx.cublas_handle();
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        for (int r0 = 0; r0 < N; r0 += TILE_N) {
            const int tn = (r0 + TILE_N <= N) ? TILE_N : (N - r0);
            const int total_u32 = tn * (K >> 3);
            em_dequant_q4_g64_f16_kernel<<<(total_u32 + 255) / 256, 256, 0, stream>>>(
                lo4, sc, lut, w_f16.get(), r0, tn, K);

            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                tn, B, K,
                &alpha,
                w_f16.get(), CUDA_R_16F, K,
                x_f16.get(), CUDA_R_16F, K,
                &beta,
                (float *)dst->data + r0, CUDA_R_32F, (int)y_nb1_f32,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    }
    return true;
}

bool ggml_cuda_emeans_try_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const EmTensor * em = emns_lookup(src0);
    if (!em) return false;

    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);

    const int K         = (int)src0->ne[0];
    const int N         = (int)src0->ne[1];
    const int n_experts = (int)src0->ne[2];
    const int ne11      = (int)src1->ne[1];
    const int n_used    = (int)ids->ne[0];
    const int n_tokens  = (int)ids->ne[1];
    const int64_t x_nb1_f32   = src1->nb[1] / sizeof(float);
    const int64_t x_nb2_f32   = src1->nb[2] / sizeof(float);
    const int64_t ids_nb1_i32 = ids->nb[1]  / sizeof(int32_t);
    const int64_t y_nb1_f32   = dst->nb[1]  / sizeof(float);
    const int64_t y_nb2_f32   = dst->nb[2]  / sizeof(float);
    hipStream_t stream = ctx.stream();

    // Cast src1 (F32) -> contiguous FP16 once
    ggml_cuda_pool_alloc<__half> x_f16(ctx.pool(), (size_t)n_tokens * ne11 * K);
    {
        dim3 block(256);
        dim3 grid((K + 255) / 256, ne11, n_tokens);
        em_f32_to_f16_moe_kernel<<<grid, block, 0, stream>>>(
            (const float *)src1->data, x_f16.get(), K, ne11, n_tokens, x_nb1_f32, x_nb2_f32);
    }

    if (n_tokens < 4 || n_experts > 256) {
        // Direct Wave32 0-spill MoE GEMV for decode / tiny batches
        dim3 block(256);
        dim3 grid((N + 7) / 8, n_used, n_tokens);

        if (em->fmt == EM_FMT_EM1D_Q3_G64) {
            em_gemv_q3_g64_moe_kernel<<<grid, block, 0, stream>>>(
                em->ptr<uint32_t>("lo2"),
                em->ptr<uint32_t>("hi1"),
                em->ptr<__half>("scales"),
                em->ptr<uint16_t>("lut"),
                (const int32_t *)ids->data,
                reinterpret_cast<const uint32_t *>(x_f16.get()),
                (float *)dst->data,
                N, K, ne11, em->is_w13 ? 1 : 0, ids_nb1_i32, y_nb1_f32, y_nb2_f32);
        } else if (em->fmt == EM_FMT_EM2D_K32_G128) {
            em_gemv_2d_k32_g128_moe_kernel<<<grid, block, 0, stream>>>(
                em->ptr<uint32_t>("lo4"),
                em->ptr<uint16_t>("hi1"),
                em->ptr<__half>("scales"),
                em->ptr<uint32_t>("lut"),
                (const int32_t *)ids->data,
                reinterpret_cast<const uint32_t *>(x_f16.get()),
                (float *)dst->data,
                N, K, ne11, em->is_w13 ? 1 : 0, ids_nb1_i32, y_nb1_f32, y_nb2_f32);
        } else if (em->fmt == EM_FMT_EM2D_K16_G128) {
            em_gemv_2d_k16_g128_moe_kernel<<<grid, block, 0, stream>>>(
                em->ptr<uint32_t>("lo4"),
                em->ptr<__half>("scales"),
                em->ptr<uint32_t>("lut"),
                (const int32_t *)ids->data,
                reinterpret_cast<const uint32_t *>(x_f16.get()),
                (float *)dst->data,
                N, K, ne11, em->is_w13 ? 1 : 0, ids_nb1_i32, y_nb1_f32, y_nb2_f32);
        } else {
            GGML_ABORT("[emeans] unsupported MoE format %u", em->fmt);
        }
        return true;
    }

    // Single-launch zero-sync GPU-grouped MoE kernel (prefill & speculative verification)
#ifndef GGML_USE_HIP
    if (n_tokens >= 16 && n_experts <= 256 && ggml_cuda_info().devices[ctx.device].cc >= GGML_CUDA_CC_TURING) {
        const int P         = n_tokens * n_used;
        const int max_tiles = (P + 15) / 16 + n_experts;
        ggml_cuda_pool_alloc<uint32_t> d_sorted_pairs(ctx.pool(), (size_t)P);
        ggml_cuda_pool_alloc<uint2>    d_tiles(ctx.pool(), (size_t)max_tiles);
        ggml_cuda_pool_alloc<int32_t>  d_num_tiles(ctx.pool(), 1);

        em_moe_build_tiles16_kernel<<<1, 256, 0, stream>>>(
            (const int32_t *)ids->data,
            d_sorted_pairs.get(),
            d_tiles.get(),
            d_num_tiles.get(),
            n_tokens, n_used, n_experts, ids_nb1_i32);

        dim3 block(256);
        dim3 grid((N + 127) / 128, max_tiles);

        if (em->fmt == EM_FMT_EM1D_Q3_G64) {
            em_gemm16_wmma_q3_g64_moe_kernel<<<grid, block, 0, stream>>>(
                em->ptr<uint32_t>("lo2"),
                em->ptr<uint32_t>("hi1"),
                em->ptr<__half>("scales"),
                em->ptr<uint16_t>("lut"),
                d_sorted_pairs.get(),
                d_tiles.get(),
                d_num_tiles.get(),
                reinterpret_cast<const uint32_t *>(x_f16.get()),
                (float *)dst->data,
                N, K, ne11, em->is_w13 ? 1 : 0, y_nb1_f32, y_nb2_f32);
        } else if (em->fmt == EM_FMT_EM2D_K32_G128) {
            em_gemm16_wmma_2d_k32_g128_moe_kernel<<<grid, block, 0, stream>>>(
                em->ptr<uint32_t>("lo4"),
                em->ptr<uint16_t>("hi1"),
                em->ptr<__half>("scales"),
                em->ptr<uint32_t>("lut"),
                d_sorted_pairs.get(),
                d_tiles.get(),
                d_num_tiles.get(),
                reinterpret_cast<const uint32_t *>(x_f16.get()),
                (float *)dst->data,
                N, K, ne11, em->is_w13 ? 1 : 0, y_nb1_f32, y_nb2_f32);
        } else if (em->fmt == EM_FMT_EM2D_K16_G128) {
            em_gemm16_wmma_2d_k16_g128_moe_kernel<<<grid, block, 0, stream>>>(
                em->ptr<uint32_t>("lo4"),
                em->ptr<__half>("scales"),
                em->ptr<uint32_t>("lut"),
                d_sorted_pairs.get(),
                d_tiles.get(),
                d_num_tiles.get(),
                reinterpret_cast<const uint32_t *>(x_f16.get()),
                (float *)dst->data,
                N, K, ne11, em->is_w13 ? 1 : 0, y_nb1_f32, y_nb2_f32);
        } else {
            GGML_ABORT("[emeans] unsupported MoE format %u", em->fmt);
        }
        return true;
    }
#endif

    // Single-launch zero-sync GPU-grouped BT=8 MoE kernel (prefill & speculative verification)
    const int P         = n_tokens * n_used;
    const int max_tiles = (P + 7) / 8 + n_experts;
    ggml_cuda_pool_alloc<uint32_t> d_sorted_pairs(ctx.pool(), (size_t)P);
    ggml_cuda_pool_alloc<uint2>    d_tiles(ctx.pool(), (size_t)max_tiles);
    ggml_cuda_pool_alloc<int32_t>  d_num_tiles(ctx.pool(), 1);

    em_moe_build_tiles_kernel<<<1, 256, 0, stream>>>(
        (const int32_t *)ids->data,
        d_sorted_pairs.get(),
        d_tiles.get(),
        d_num_tiles.get(),
        n_tokens, n_used, n_experts, ids_nb1_i32);

    dim3 block(256);
    dim3 grid((N + 7) / 8, max_tiles);

    if (em->fmt == EM_FMT_EM1D_Q3_G64) {
        em_gemm8_q3_g64_moe_kernel<<<grid, block, 0, stream>>>(
            em->ptr<uint32_t>("lo2"),
            em->ptr<uint32_t>("hi1"),
            em->ptr<__half>("scales"),
            em->ptr<uint16_t>("lut"),
            d_sorted_pairs.get(),
            d_tiles.get(),
            d_num_tiles.get(),
            reinterpret_cast<const uint32_t *>(x_f16.get()),
            (float *)dst->data,
            N, K, ne11, em->is_w13 ? 1 : 0, y_nb1_f32, y_nb2_f32);
    } else if (em->fmt == EM_FMT_EM2D_K32_G128) {
        em_gemm8_2d_k32_g128_moe_kernel<<<grid, block, 0, stream>>>(
            em->ptr<uint32_t>("lo4"),
            em->ptr<uint16_t>("hi1"),
            em->ptr<__half>("scales"),
            em->ptr<uint32_t>("lut"),
            d_sorted_pairs.get(),
            d_tiles.get(),
            d_num_tiles.get(),
            reinterpret_cast<const uint32_t *>(x_f16.get()),
            (float *)dst->data,
            N, K, ne11, em->is_w13 ? 1 : 0, y_nb1_f32, y_nb2_f32);
    } else if (em->fmt == EM_FMT_EM2D_K16_G128) {
        em_gemm8_2d_k16_g128_moe_kernel<<<grid, block, 0, stream>>>(
            em->ptr<uint32_t>("lo4"),
            em->ptr<__half>("scales"),
            em->ptr<uint32_t>("lut"),
            d_sorted_pairs.get(),
            d_tiles.get(),
            d_num_tiles.get(),
            reinterpret_cast<const uint32_t *>(x_f16.get()),
            (float *)dst->data,
            N, K, ne11, em->is_w13 ? 1 : 0, y_nb1_f32, y_nb2_f32);
    } else {
        GGML_ABORT("[emeans] unsupported MoE format %u", em->fmt);
    }
    return true;
}

bool ggml_cuda_emeans_try_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const EmTensor * em = emns_lookup(src0);
    if (!em) return false;

    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(em->fmt == EM_FMT_EM1D_Q4_G64);
    GGML_ASSERT(src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);

    const int K        = (int)src0->ne[0];
    const int n_tokens = (int)ggml_nelements(src1);
    const int64_t dst_nb1_f32 = dst->nb[1] / sizeof(float);
    hipStream_t stream = ctx.stream();

    dim3 block(256);
    dim3 grid(((K >> 3) + 255) / 256, n_tokens);
    em_get_rows_q4_g64_kernel<<<grid, block, 0, stream>>>(
        em->ptr<uint32_t>("lo4"),
        em->ptr<__half>("scales"),
        em->ptr<uint16_t>("lut"),
        (const int32_t *)src1->data,
        (float *)dst->data,
        K, n_tokens, dst_nb1_f32);
    return true;
}

#else

extern "C" {
bool ggml_cuda_emeans_init(const char *) { return false; }
bool ggml_cuda_emeans_init_extra(const char *) { return false; }
ggml_backend_buffer_t ggml_cuda_emeans_get_stub_buffer(ggml_backend_buffer_type_t) { return nullptr; }
void * ggml_cuda_emeans_bind_tensor(const char *, int64_t, int64_t, int64_t) { return nullptr; }
bool ggml_cuda_emeans_is_stub(const void *) { return false; }
}
bool ggml_cuda_emeans_has(const ggml_tensor *) { return false; }
bool ggml_cuda_emeans_try_mul_mat(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_emeans_try_mul_mat_id(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
bool ggml_cuda_emeans_try_get_rows(ggml_backend_cuda_context &, ggml_tensor *) { return false; }

#endif
