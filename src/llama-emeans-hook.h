#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

struct llama_emeans_api {
    using init_fn      = bool (*)(const char *);
    using get_buf_fn   = ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t);
    using bind_fn      = void * (*)(const char *, int64_t, int64_t, int64_t);
    using is_stub_fn   = bool (*)(const void *);

    init_fn    init    = nullptr;
    get_buf_fn get_buf = nullptr;
    bind_fn    bind    = nullptr;
    is_stub_fn is_stub = nullptr;
    bool resolved = false;

    static llama_emeans_api & get() {
        static llama_emeans_api api;
        if (!api.resolved) {
            api.resolved = true;
            HMODULE mod = GetModuleHandleA("ggml-hip.dll");
            if (!mod) mod = GetModuleHandleA("ggml-cuda.dll");
            if (mod) {
                api.init    = reinterpret_cast<init_fn>(GetProcAddress(mod, "ggml_cuda_emeans_init"));
                api.get_buf = reinterpret_cast<get_buf_fn>(GetProcAddress(mod, "ggml_cuda_emeans_get_stub_buffer"));
                api.bind    = reinterpret_cast<bind_fn>(GetProcAddress(mod, "ggml_cuda_emeans_bind_tensor"));
                api.is_stub = reinterpret_cast<is_stub_fn>(GetProcAddress(mod, "ggml_cuda_emeans_is_stub"));
            }
        }
        return api;
    }

    bool ok() const { return init && get_buf && bind && is_stub; }
};
#else
#include <dlfcn.h>

struct llama_emeans_api {
    using init_fn      = bool (*)(const char *);
    using get_buf_fn   = ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t);
    using bind_fn      = void * (*)(const char *, int64_t, int64_t, int64_t);
    using is_stub_fn   = bool (*)(const void *);

    init_fn    init    = nullptr;
    get_buf_fn get_buf = nullptr;
    bind_fn    bind    = nullptr;
    is_stub_fn is_stub = nullptr;
    bool resolved = false;

    static llama_emeans_api & get() {
        static llama_emeans_api api;
        if (!api.resolved) {
            api.resolved = true;
            void * mod = dlopen("libggml-cuda.so", RTLD_LAZY | RTLD_NOLOAD);
            if (!mod) mod = dlopen("libggml-hip.so", RTLD_LAZY | RTLD_NOLOAD);
            void * handle = mod ? mod : RTLD_DEFAULT;
            api.init    = reinterpret_cast<init_fn>(dlsym(handle, "ggml_cuda_emeans_init"));
            api.get_buf = reinterpret_cast<get_buf_fn>(dlsym(handle, "ggml_cuda_emeans_get_stub_buffer"));
            api.bind    = reinterpret_cast<bind_fn>(dlsym(handle, "ggml_cuda_emeans_bind_tensor"));
            api.is_stub = reinterpret_cast<is_stub_fn>(dlsym(handle, "ggml_cuda_emeans_is_stub"));
        }
        return api;
    }

    bool ok() const { return init && get_buf && bind && is_stub; }
};
#endif
