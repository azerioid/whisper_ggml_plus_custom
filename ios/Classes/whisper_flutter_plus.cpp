#include "whisper/whisper.h"

#define DR_WAV_IMPLEMENTATION
#include "whisper/examples/dr_wav.h"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <iostream>
#include <chrono>
#include "json/json.hpp"
#include <stdio.h>

#if defined(__GNUC__)
#define FUNCTION_ATTRIBUTE __attribute__((visibility("default"))) __attribute__((used))
#elif defined(_MSC_VER)
#define FUNCTION_ATTRIBUTE __declspec(dllexport)
#else
#define FUNCTION_ATTRIBUTE
#endif

extern "C" const char* get_vad_model_path();

using json = nlohmann::json;

enum class whisper_vad_mode {
    auto_mode,
    disabled,
    enabled,
};

struct whisper_params
{
    int32_t seed = -1;
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 5;
    int32_t beam_size = -1;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;

    bool verbose = false;
    bool print_special_tokens = false;
    bool speed_up = false;
    bool translate = false;
    bool diarize = false;
    bool no_fallback = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool split_on_word = false;
    whisper_vad_mode vad_mode = whisper_vad_mode::auto_mode;

    std::string language = "id";
    std::string prompt;
    std::string model = "";
    std::string audio = "";
    std::string vad_model_path = "";
    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_outp = {};
};

static whisper_vad_mode parse_vad_mode(const json & json_body) {
    const std::string vad_mode = json_body.value("vad_mode", std::string("auto"));
    if (vad_mode == "disabled") {
        return whisper_vad_mode::disabled;
    }
    if (vad_mode == "enabled") {
        return whisper_vad_mode::enabled;
    }
    return whisper_vad_mode::auto_mode;
}

static std::string default_vad_model_path() {
    const char * vad_path = get_vad_model_path();
    return vad_path == nullptr ? std::string() : std::string(vad_path);
}

static struct whisper_context * g_ctx = nullptr;
static std::string g_model_path = "";
static std::mutex g_mutex;
static std::atomic<bool> g_should_abort(false);

static int clamp_thread_count(int requested, float audio_seconds, bool speed_up, bool is_turbo) {
    int threads = std::max(1, std::min(requested, 8));

    // Latency optimization for short clips: fewer threads reduce scheduling overhead.
    if (audio_seconds > 0.0f && audio_seconds <= 8.0f) {
        threads = std::min(threads, speed_up ? 2 : 3);
    } else if (audio_seconds > 0.0f && audio_seconds <= 20.0f) {
        threads = std::min(threads, 4);
    }

    // Turbo models remain memory-heavy; keep conservative upper bound.
    if (is_turbo) {
        threads = std::min(threads, 4);
    }

    return threads;
}

static void dispose_context_locked() {
    if (g_ctx != nullptr) {
        whisper_free(g_ctx);
        g_ctx = nullptr;
    }
    g_model_path.clear();
}

// Abort callback for whisper transcription
static bool abort_callback(void* user_data) {
    return g_should_abort.load();
}

char *jsonToChar(json jsonData)
{
    try {
        // Ensure ASCII encoding to avoid UTF-8 issues across FFI boundary
        // Non-ASCII characters (Korean, etc.) will be escaped as \uXXXX
        // Use 'replace' instead of 'strict' to handle malformed UTF-8 from Whisper output
        // (e.g., truncated multibyte sequences like 0xEC without following bytes)
        std::string result = jsonData.dump(-1, ' ', true, nlohmann::json::error_handler_t::replace);
        char *ch = new char[result.size() + 1];
        if (ch) {
            strcpy(ch, result.c_str());
        }
        return ch;
    } catch (const std::exception& e) {
        // Fallback for absolute safety
        std::string errorJson = "{\"@type\":\"error\",\"message\":\"JSON serialization failed\"}";
        char *ch = new char[errorJson.size() + 1];
        strcpy(ch, errorJson.c_str());
        return ch;
    }
}

json transcribe(json jsonBody)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    
    g_should_abort.store(false);

    whisper_params params;
    params.n_threads = std::max(1, jsonBody.value("threads", params.n_threads));
    params.verbose = jsonBody.value("is_verbose", false);
    params.translate = jsonBody.value("is_translate", false);
    params.language = jsonBody.value("language", std::string("auto"));
    params.print_special_tokens = jsonBody.value("is_special_tokens", false);
    params.no_timestamps = jsonBody.value("is_no_timestamps", false);
    params.model = jsonBody.value("model", std::string(""));
    params.audio = jsonBody.value("audio", std::string(""));
    params.split_on_word = jsonBody.value("split_on_word", false);
    params.diarize = jsonBody.value("diarize", false);
    params.speed_up = jsonBody.value("speed_up", false);
    params.vad_mode = parse_vad_mode(jsonBody);
    params.vad_model_path = jsonBody.value("vad_model_path", std::string(""));

    json jsonResult;
    jsonResult["@type"] = "transcribe";

    if (g_ctx == nullptr || g_model_path != params.model) {
        dispose_context_locked();
        
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = false; // Diagnostic mode: Disable Metal GPU to prevent native crashes
        cparams.flash_attn = false; // Disabled: known to cause iOS Native Crashes

        g_ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
        if (g_ctx != nullptr) {
            g_model_path = params.model;
        }
    }

    if (g_ctx == nullptr)
    {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "failed to initialize whisper context (possibly OOM)";
        return jsonResult;
    }

    std::vector<float> pcmf32;
    float audio_seconds = 0.0f;
    {
        drwav wav;
        if (!drwav_init_file(&wav, params.audio.c_str(), NULL))
        {
            jsonResult["@type"] = "error";
            jsonResult["message"] = " failed to open WAV file ";
            return jsonResult;
        }

        const int n = wav.totalPCMFrameCount;
        const uint32_t sample_rate = wav.sampleRate;
        std::vector<int16_t> pcm16(n * wav.channels);
        drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
        drwav_uninit(&wav);

        if (sample_rate > 0) {
            audio_seconds = static_cast<float>(n) / static_cast<float>(sample_rate);
        }

        pcmf32.resize(n);
        if (wav.channels == 1) {
            for (int i = 0; i < n; i++) pcmf32[i] = float(pcm16[i]) / 32768.0f;
        } else {
            for (int i = 0; i < n; i++) pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
    }

    // Model inspection for turbo-specific strategy and safe thread cap.
    const int model_n_text_layer = whisper_model_n_text_layer(g_ctx);
    const int model_n_vocab = whisper_model_n_vocab(g_ctx);
    const bool is_turbo = (model_n_text_layer == 4 && model_n_vocab == 51866);

    if (params.verbose) {
        fprintf(stderr, "[DEBUG] Model info - n_text_layer: %d, n_vocab: %d, is_turbo: %d\n",
                model_n_text_layer, model_n_vocab, is_turbo);
    }

    const bool prefer_speed_strategy = params.speed_up || params.no_timestamps;
    whisper_sampling_strategy strategy =
        (is_turbo && !prefer_speed_strategy)
            ? WHISPER_SAMPLING_BEAM_SEARCH
            : WHISPER_SAMPLING_GREEDY;
    whisper_full_params wparams = whisper_full_default_params(strategy);
    
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.no_timestamps = params.no_timestamps;
    wparams.translate = params.translate;
    wparams.language = params.language.c_str();
    wparams.n_threads = clamp_thread_count(
        params.n_threads,
        audio_seconds,
        params.speed_up,
        is_turbo
    );
    wparams.split_on_word = params.split_on_word;
    wparams.audio_ctx = params.speed_up ? 768 : 0; // Use smaller audio context for speedUp
    wparams.single_segment = false;

    if (params.split_on_word) {
        if (params.verbose) {
            fprintf(stderr, "[DEBUG] Disabling VAD because split_on_word requires stable timestamps\n");
        }
        wparams.vad = false;
    } else if (params.no_timestamps) {
        // For text-only output, VAD adds overhead and provides limited value.
        wparams.vad = false;
    } else if (params.vad_mode == whisper_vad_mode::disabled) {
        wparams.vad = false;
    } else {
        if (params.vad_model_path.empty()) {
            params.vad_model_path = default_vad_model_path();
        }
        if (!params.vad_model_path.empty()) {
            wparams.vad = true;
            wparams.vad_model_path = params.vad_model_path.c_str();
        } else if (params.vad_mode == whisper_vad_mode::enabled) {
            jsonResult["@type"] = "error";
            jsonResult["message"] =
                "VAD was explicitly enabled but no vad_model_path was available";
            return jsonResult;
        } else {
            wparams.vad = false;
        }
    }

    if (is_turbo) {
        wparams.beam_search.beam_size = 3;
        if (params.verbose && strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
            fprintf(stderr, "[DEBUG] Turbo model detected - using beam search (beam_size=3)\n");
        }
    }

    if (params.split_on_word) {
        wparams.max_len = 1;
        wparams.token_timestamps = true;
    }
    
    // Set abort callback
    wparams.abort_callback = abort_callback;
    wparams.abort_callback_user_data = nullptr;

    if (params.verbose) {
        fprintf(stderr, "[DEBUG] Transcription params - threads: %d, speed_up: %d, no_timestamps: %d, single_segment: %d, split_on_word: %d, max_len: %d, audio_seconds: %.2f\n",
                wparams.n_threads, params.speed_up, wparams.no_timestamps, wparams.single_segment, wparams.split_on_word, wparams.max_len, audio_seconds);
        fflush(stderr);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (whisper_full(g_ctx, wparams, pcmf32.data(), pcmf32.size()) != 0)
    {
        if (g_should_abort.load()) {
            jsonResult["@type"] = "aborted";
            jsonResult["message"] = "transcription aborted by user";
            g_should_abort.store(false);
            return jsonResult;
        }
        jsonResult["@type"] = "error";
        jsonResult["message"] = "failed to process audio";
        return jsonResult;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (params.verbose) {
        fprintf(stderr, "[DEBUG] Transcription completed in %lldms\n", (long long)duration);
        fflush(stderr);
    }

    const int n_segments = whisper_full_n_segments(g_ctx);
    std::vector<json> segmentsJson = {};
    std::string text_result = "";

    for (int i = 0; i < n_segments; ++i)
    {
        const char *text = whisper_full_get_segment_text(g_ctx, i);
        text_result += std::string(text);
        
        if (!params.no_timestamps) {
            json jsonSegment;
            jsonSegment["from_ts"] = whisper_full_get_segment_t0(g_ctx, i);
            jsonSegment["to_ts"] = whisper_full_get_segment_t1(g_ctx, i);
            jsonSegment["text"] = text;
            segmentsJson.push_back(jsonSegment);
        }
    }

    if (!params.no_timestamps) {
        jsonResult["segments"] = segmentsJson;
    }
    
    jsonResult["text"] = text_result;
    return jsonResult;
}

extern "C"
{
    FUNCTION_ATTRIBUTE char *vpnai_whisper_request(char *body)
    {
        try {
            json jsonBody = json::parse(body);
            if (jsonBody["@type"] == "abort") {
                g_should_abort.store(true);
                return jsonToChar({{"@type", "abort"}, {"message", "abort signal sent"}});
            }
            if (jsonBody["@type"] == "dispose") {
                std::lock_guard<std::mutex> lock(g_mutex);
                dispose_context_locked();
                return jsonToChar({{"@type", "dispose"}, {"message", "whisper context disposed"}});
            }
            if (jsonBody["@type"] == "getTextFromWavFile") {
                return jsonToChar(transcribe(jsonBody));
            }
            if (jsonBody["@type"] == "getVersion") {
                return jsonToChar({{"@type", "version"}, {"message", "lib v1.8.3-accel"}});
            }
            return jsonToChar({{"@type", "error"}, {"message", "method not found"}});
        } catch (const std::exception &e) {
            return jsonToChar({{"@type", "error"}, {"message", e.what()}});
        }
    }

    FUNCTION_ATTRIBUTE void vpnai_whisper_free_string(char *ptr)
    {
        delete[] ptr;
    }
}
