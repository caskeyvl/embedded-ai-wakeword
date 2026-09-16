/*
 * host_test_main.cpp — Linux host inference runner (NOT the student
 * deliverable; see main.cpp for that).
 *
 * Loads a 16 kHz mono 16-bit WAV file and a .pte model from disk, runs the
 * exact same spectrogram_compute() used on-device, executes the model via
 * ExecuTorch, and prints a prediction line in the same format the STM32
 * firmware uses over UART. This lets the whole pipeline (feature extraction
 * + model) be validated without any hardware, before ever touching main.cpp
 * or flashing a board.
 *
 * Usage:
 *   ./inference_host <audio.wav> [model.pte]
 *   (model.pte defaults to submission/model.pte, relative to cwd)
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/method.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/portable_type/tensor_impl.h"
#include "executorch/runtime/core/hierarchical_allocator.h"
#include "executorch/runtime/platform/runtime.h"
#include "executorch/extension/data_loader/file_data_loader.h"

#include "spectrogram.h"
#include "wakeword_labels.h"

using namespace executorch::runtime;
using namespace executorch::extension;
using namespace executorch::runtime::etensor;

/* Generous host-side pools -- large enough to also run the (host-only)
 * ExactMatchCNN, whose ~11.2M-parameter Dense layer needs a big activation
 * buffer (see README-training-parameters.md Appendix A). On-device sizes
 * are tuned separately in main.cpp, much smaller, once ImprovedKWSNet's
 * real memory-plan requirements are measured (see README.md "Memory
 * budget") -- the exact model is never embedded, so this generosity has no
 * on-device cost. */
static uint8_t method_pool[4 * 1024 * 1024];
static uint8_t planned_pool[192 * 1024 * 1024];

static float spectrogram_out[SPEC_OUT_SIZE];

static TensorImpl::SizesType    input_sizes[4]    = {1, 1, SPEC_OUT_FRAMES, SPEC_OUT_FREQ};
static TensorImpl::DimOrderType input_dim_order[4] = {0, 1, 2, 3};
static TensorImpl::StridesType  input_strides[4]   = {
    SPEC_OUT_FRAMES * SPEC_OUT_FREQ, SPEC_OUT_FRAMES * SPEC_OUT_FREQ, SPEC_OUT_FREQ, 1
};

/* Minimal RIFF/WAVE reader: mono, 16-bit PCM, 16 kHz only. Reads exactly
 * SPEC_NUM_SAMPLES samples, zero-padding a short clip or truncating a long
 * one to exactly 1.0 s -- matching the fixed-size window main.cpp reads out
 * of the mic ring buffer on-device. */
static bool load_wav(const char *path, int16_t *out_pcm) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return false;
    }

    uint8_t riff_hdr[12];
    if (fread(riff_hdr, 1, 12, f) != 12 ||
        memcmp(riff_hdr, "RIFF", 4) != 0 || memcmp(riff_hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "error: %s is not a RIFF/WAVE file\n", path);
        fclose(f);
        return false;
    }

    uint16_t num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    long data_offset = 0;
    uint32_t data_size = 0;

    for (;;) {
        uint8_t chunk_hdr[8];
        if (fread(chunk_hdr, 1, 8, f) != 8) break;
        uint32_t chunk_size = (uint32_t)chunk_hdr[4] | ((uint32_t)chunk_hdr[5] << 8) |
                               ((uint32_t)chunk_hdr[6] << 16) | ((uint32_t)chunk_hdr[7] << 24);

        if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (fread(fmt, 1, 16, f) != 16) { fclose(f); return false; }
            num_channels = (uint16_t)(fmt[2] | (fmt[3] << 8));
            sample_rate = (uint32_t)fmt[4] | ((uint32_t)fmt[5] << 8) |
                          ((uint32_t)fmt[6] << 16) | ((uint32_t)fmt[7] << 24);
            bits_per_sample = (uint16_t)(fmt[14] | (fmt[15] << 8));
            if (chunk_size > 16) fseek(f, (long)(chunk_size - 16), SEEK_CUR);
        } else if (memcmp(chunk_hdr, "data", 4) == 0) {
            data_offset = ftell(f);
            data_size = chunk_size;
            break;
        } else {
            fseek(f, (long)chunk_size + (long)(chunk_size & 1), SEEK_CUR);
        }
    }

    if (data_offset == 0) {
        fprintf(stderr, "error: %s has no data chunk\n", path);
        fclose(f);
        return false;
    }
    if (num_channels != 1 || bits_per_sample != 16 || sample_rate != SPEC_SAMPLE_RATE) {
        fprintf(stderr, "error: %s must be mono 16-bit %d Hz (got %u ch, %u bit, %u Hz)\n",
                path, SPEC_SAMPLE_RATE, num_channels, bits_per_sample, sample_rate);
        fclose(f);
        return false;
    }

    uint32_t n_available = data_size / 2;
    uint32_t n_read = (n_available < SPEC_NUM_SAMPLES) ? n_available : SPEC_NUM_SAMPLES;

    memset(out_pcm, 0, SPEC_NUM_SAMPLES * sizeof(int16_t));
    fseek(f, data_offset, SEEK_SET);
    size_t got = fread(out_pcm, sizeof(int16_t), n_read, f);
    fclose(f);

    printf("[host] Loaded %zu of %u requested samples (%.3f s) from %s\n",
           got, SPEC_NUM_SAMPLES, (double)got / SPEC_SAMPLE_RATE, path);
    return true;
}

static float softmax_confidence(const float *logits, int n, int pred) {
    float max_logit = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - max_logit);
    return expf(logits[pred] - max_logit) / sum;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <audio.wav> [model.pte]\n", argv[0]);
        return 1;
    }
    const char *wav_path = argv[1];
    const char *pte_path = (argc > 2) ? argv[2] : "submission/model.pte";

    static int16_t pcm[SPEC_NUM_SAMPLES];
    if (!load_wav(wav_path, pcm)) return 1;

    spectrogram_init();
    spectrogram_compute(pcm, spectrogram_out);

    runtime_init();

    Result<FileDataLoader> loader_result = FileDataLoader::from(pte_path);
    if (!loader_result.ok()) {
        fprintf(stderr, "error: could not open model: %s\n", pte_path);
        return 1;
    }
    FileDataLoader loader = std::move(loader_result.get());

    Result<Program> program_result = Program::load(&loader);
    if (!program_result.ok()) {
        fprintf(stderr, "error: Program::load failed (code %d)\n", (int)program_result.error());
        return 1;
    }
    Program program = std::move(program_result.get());

    MemoryAllocator method_allocator(sizeof(method_pool), method_pool);
    Span<uint8_t> planned_span(planned_pool, sizeof(planned_pool));
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator);

    Result<Method> method_result = program.load_method("forward", &memory_manager);
    if (!method_result.ok()) {
        fprintf(stderr, "error: load_method failed (code %d)\n", (int)method_result.error());
        return 1;
    }
    Method method = std::move(method_result.get());

    TensorImpl input_impl(ScalarType::Float, 4, input_sizes, spectrogram_out,
                           input_dim_order, input_strides);
    Tensor input_tensor(&input_impl);

    if (method.set_input(EValue(input_tensor), 0) != Error::Ok) {
        fprintf(stderr, "error: set_input failed\n");
        return 1;
    }
    if (method.execute() != Error::Ok) {
        fprintf(stderr, "error: execute failed\n");
        return 1;
    }

    EValue output = method.get_output(0);
    const float *logits = output.toTensor().const_data_ptr<float>();

    int pred = 0;
    for (int i = 1; i < WAKEWORD_NUM_CLASSES; i++) {
        if (logits[i] > logits[pred]) pred = i;
    }
    float conf = softmax_confidence(logits, WAKEWORD_NUM_CLASSES, pred);

    printf("PREDICTION: %s conf=%.4f\n", WAKEWORD_CLASS_NAMES[pred], conf);
    return 0;
}
