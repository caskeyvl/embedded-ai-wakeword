/*
 * main.cpp — Wake word / keyword-spotting firmware for STM32F407 Discovery.
 *
 * PDM microphone -> STFT spectrogram -> ExecuTorch CNN inference -> UART
 * prediction. See ../README.md for the full architecture, build/flash
 * instructions, and the meaning of every tunable constant below.
 *
 * This is the provided skeleton: the `STUDENT SECTION` below and a few
 * other `TODO(student)` spots (memory pool sizes, the ExecuTorch bring-up)
 * are your assignment; everything else here ships complete.
 */

#include <cstdint>
#include <cstring>
#include <cmath>

/* ── ExecuTorch includes ─────────────────────────────────────────────────── */
#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/method.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/portable_type/tensor_impl.h"
#include "executorch/runtime/core/hierarchical_allocator.h"
#include "executorch/runtime/platform/runtime.h"
#include "executorch/extension/data_loader/buffer_data_loader.h"

/* ── Project includes ────────────────────────────────────────────────────── */
#include "hal.h"           /* board bring-up: clocks, LEDs, UART (referenced from ../../stm32f4_audio_examples) */
#include "pdm_mic.h"        /* mic driver: pdm_mic_init/get_ring/samples_written (same repo) */
#include "spectrogram.h"    /* this project's feature-extraction module */
#include "wakeword_labels.h"
#include "spectrogram_norm_constants.h"  /* SPECTROGRAM_MEAN/STD -- pulled in transitively by spectrogram.c, listed here too since main.cpp's comments reference it */
#include "wakeword_pte.h"   /* generated at configure time from submission/model.pte (see ../CMakeLists.txt) */

using namespace executorch::runtime;
using namespace executorch::extension;
using namespace executorch::runtime::etensor;

/* ═══════════════════════════════════════════════════════════════════════
 * Runtime-tunable parameters — see ../README.md "Runtime-tunable
 * parameters" for the full explanation of each. Spectrogram-shape
 * constants (SPEC_*) live in spectrogram.h since they're also needed by
 * the training-side Python; only the inference-loop-specific constants
 * are defined here.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Inference window size, in samples -- fixed by the model's expected input
 * duration (1.0s).
 *
 * A naive design would run inference on a fixed periodic clock (a window
 * every WINDOW_SAMPLES samples, back-to-back). Measure how long
 * spectrogram_compute() + method.execute() actually take on this hardware
 * (see the timing-instrumentation TODO below) before assuming that's fast
 * enough -- ExecuTorch's portable/unoptimized kernels are not fast, and a
 * fixed periodic design that falls behind the mic's real-time capture will
 * silently drop audio every cycle. This firmware instead gates the
 * expensive pipeline behind a cheap voice-activity check (see the STUDENT
 * SECTION below) that only triggers a capture+inference when audio
 * activity is actually detected. */
static const uint32_t WINDOW_SAMPLES = SPEC_NUM_SAMPLES;  /* 16000 = 1.0 s */

/* ExecuTorch memory pools, both in main SRAM (128 KB @ 0x20000000, DMA-
 * reachable).
 *
 * TODO(student): these are currently sized 0/0 -- grow them until
 * load_method() (below) stops failing with MemoryAllocationFailed (0x21).
 * See ../README.md "Memory budget" for how to get a starting estimate
 * (planned_pool's minimum can be read directly off your exported .pte via
 * non_const_buffer_sizes; method_pool has to be found by trial). These are
 * NOT safe defaults for an arbitrary model -- re-measure if you change the
 * architecture. */
/* planned_pool minimum read off submission/model.pte via non_const_buffer_sizes
 * (37376 bytes) -- sized here with headroom. method_pool found by trial
 * (32KB loads fine for this model; see README-testing.md "Memory budget"). */
static uint8_t method_pool[32 * 1024];
static uint8_t planned_pool[40 * 1024];

/* Linear copy of one inference window, populated by the STUDENT SECTION's
 * read step before any spectrogram/inference work touches it. This copy is
 * NOT optional -- see the STUDENT SECTION comment on why reading directly
 * out of the mic ring while inference runs is unsafe. */
static int16_t window_scratch[WINDOW_SAMPLES];

/* Model input tensor storage. Placed in CCM RAM -- main SRAM has very
 * little headroom once dma_buf + window_scratch + method_pool + planned_pool
 * are accounted for (see ../README.md "Memory budget"). */
#ifdef TARGET_STM32
__attribute__((section(".ccmram")))
#endif
static float spectrogram_out[SPEC_OUT_SIZE];

static TensorImpl::SizesType    input_sizes[4]    = {1, 1, SPEC_OUT_FRAMES, SPEC_OUT_FREQ};
static TensorImpl::DimOrderType input_dim_order[4] = {0, 1, 2, 3};
static TensorImpl::StridesType  input_strides[4]   = {
    SPEC_OUT_FRAMES * SPEC_OUT_FREQ, SPEC_OUT_FRAMES * SPEC_OUT_FREQ, SPEC_OUT_FREQ, 1
};

static float softmax_confidence(const float *logits, int n, int pred) {
    float max_logit = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - max_logit);
    return expf(logits[pred] - max_logit) / sum;
}

int main(void) {
    /* ── Board + mic bring-up ────────────────────────────────────────────── */
    hal_init();
    pdm_mic_init();
    spectrogram_init();
    uart_puts("hal_init/pdm_mic_init/spectrogram_init OK\r\n");

    runtime_init();

    BufferDataLoader loader(wakeword_pte, wakeword_pte_len);
    Result<Program> program_result = Program::load(&loader, Program::Verification::Minimal);
    if (!program_result.ok()) {
        uart_printf("ERROR: Program::load failed (0x%x)\r\n", (unsigned)program_result.error());
        while (1);
    }
    Program program = std::move(program_result.get());

    MemoryAllocator method_allocator(sizeof(method_pool), method_pool);
    Span<uint8_t> planned_span(planned_pool, sizeof(planned_pool));
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator);

    Result<Method> method_result = program.load_method("forward", &memory_manager);
    if (!method_result.ok()) {
        uart_printf("ERROR: load_method failed (0x%x)\r\n", (unsigned)method_result.error());
        while (1);
    }
    Method method = std::move(method_result.get());
    uart_puts("ExecuTorch method loaded OK\r\n");

    /* Tensor/TensorImpl metadata (shape, strides, dtype) never changes across
     * iterations, so it's constructed once here -- but method.set_input()
     * itself must be called again every iteration, right before execute()
     * (see the STUDENT SECTION loop below): ExecuTorch's Method::execute()
     * does not automatically notice in-place mutations to a previously-bound
     * input tensor's underlying buffer, so re-binding is required even
     * though the pointer (spectrogram_out) never changes. */
    TensorImpl input_impl(ScalarType::Float, 4, input_sizes, spectrogram_out,
                           input_dim_order, input_strides);
    Tensor input_tensor(&input_impl);

    const int16_t *ring;
    uint32_t ring_capacity;
    pdm_mic_get_ring(&ring, &ring_capacity);

    uart_puts("READY\r\n");  /* sync marker for host tooling / Renode tests */

    /* ═══════════════════════════════════════════════════════════════════
     * ===== STUDENT SECTION =====
     *
     * Implement a voice-activity-gated inference loop. Everything you need
     * is already set up above:
     *
     *   pdm_mic_samples_written()        -- monotonic sample counter
     *   ring[i % ring_capacity]          -- circular PCM sample buffer
     *   window_scratch[WINDOW_SAMPLES]   -- capture destination
     *   spectrogram_compute(pcm, out)    -- pcm: WINDOW_SAMPLES contiguous
     *                                        int16 samples; out: writes
     *                                        SPEC_OUT_SIZE floats
     *   spectrogram_out                  -- overwrite its contents in place,
     *                                        then re-bind with
     *                                        method.set_input() before every
     *                                        execute() -- see the note above
     *                                        input_impl/input_tensor
     *   method.set_input(EValue(input_tensor), 0) / method.execute() /
     *   method.get_output(0)             -- run + read [1,4] logits
     *   softmax_confidence(logits, n, pred) -- provided above
     *   WAKEWORD_CLASS_NAMES[i]          -- class index -> name
     *   hal_ticks_ms()                   -- for both VAD timing and the
     *                                        inference-timing instrumentation
     *                                        below
     *   uart_printf(...)                 -- formatted UART output
     *
     * Suggested structure (a simple state machine -- pseudocode, not a
     * literal step-by-step you must follow exactly):
     *
     *   idle:
     *     - Periodically (don't spin on every loop iteration -- throttle
     *       against hal_ticks_ms()) look at a short, recent slice of ring[]
     *       directly (no copy needed here -- a torn read only risks a
     *       slightly noisy trigger decision, not a corrupted model input)
     *       and decide whether it looks like speech. Your choice of
     *       heuristic and thresholds (e.g. short-term RMS, zero-crossing
     *       rate, or something else) -- tune against your own recordings.
     *     - Debounce: require the "active" signal to hold for some minimum
     *       duration before triggering, so brief clicks/pops don't cause a
     *       false detection.
     *     - On trigger: record the current sample position as your
     *       detection onset (this is what submission/debug_values.json
     *       asks you to read back later), and switch to capturing.
     *
     *   capturing:
     *     - Copy newly-arrived samples out of ring[] into window_scratch
     *       incrementally (small, frequent copies keyed off
     *       pdm_mic_samples_written(), not one big wait-then-copy) until
     *       window_scratch holds WINDOW_SAMPLES samples. Consider starting
     *       the capture with some look-back margin before the detected
     *       onset, so the word isn't cut off at its very start.
     *     - IMPORTANT: always fill window_scratch via this incremental
     *       copy -- never run spectrogram_compute() directly against
     *       ring[]. The mic driver keeps overwriting ring[] in the
     *       background for as long as pdm_mic_init() ran; if
     *       spectrogram_compute()+method.execute() (which can take a
     *       while -- see the timing note below) are still running when
     *       the DMA wraps back around, you'll get a corrupted window. This
     *       is a subtle bug to reintroduce if you try to "optimize away"
     *       the copy -- a telltale symptom is implausibly saturated ~1.0
     *       confidence on every prediction.
     *     - Once the window is full:
     *         1. (measure) spectrogram_compute(window_scratch, spectrogram_out);
     *         2. method.set_input(EValue(input_tensor), 0); check for
     *            Error::Ok.
     *         3. (measure) method.execute(); check for Error::Ok.
     *         4. Read logits via
     *            method.get_output(0).toTensor().const_data_ptr<float>(),
     *            argmax over WAKEWORD_NUM_CLASSES, and
     *            softmax_confidence() for a confidence value. Below
     *            whatever confidence threshold you choose, report
     *            "unknown" instead of the raw label.
     *         5. uart_printf("PREDICTION: %s conf=%f\r\n", label, conf) --
     *            this exact format, every time, regardless of confidence
     *            (grading parses this line).
     *         6. Record both measured durations (in ms) -- these feed
     *            submission/timing.json's board_spectrogram_ms/
     *            board_inference_ms (or renode_* under Renode) fields; see
     *            README-submission-instructions.md.
     *         7. Return to idle, with some cooldown before re-triggering
     *            (avoid re-detecting the same word's tail).
     *
     * Optional: a periodic heartbeat print (UART line + LED toggle) so you
     * can tell the board is alive between detections.
     * ═══════════════════════════════════════════════════════════════════ */

    static const int32_t  VAD_WINDOW_SAMPLES = 320;   /* 20 ms RMS window */
    static const uint32_t VAD_CHECK_PERIOD_MS = 20;
    static const float    VAD_RMS_THRESHOLD = 900.0f;
    static const uint32_t VAD_DEBOUNCE_MS = 60;        /* sustained activity to trigger */
    static const uint32_t COOLDOWN_MS = 150;           /* after a prediction, before re-arming --
                                                         * kept short because spectrogram+inference
                                                         * (under Renode's simulated CPU, ~1s) already
                                                         * eats most of the gap between back-to-back
                                                         * words in the grader's multi-word clips */
    static const float    CONF_THRESHOLD = 0.5f;

    /* Debug values for submission/debug_values.json -- static so GDB can
     * print them by name after a breakpoint right after the DETECTED print. */
    static volatile uint32_t g_onset_sample = 0;
    static volatile uint32_t g_written_at_detection = 0;

    enum VadState { STATE_IDLE, STATE_CAPTURING };
    VadState state = STATE_IDLE;

    uint32_t next_vad_check_ms = 0;
    uint32_t active_run_ms = 0;
    uint32_t capture_start = 0;
    uint32_t cooldown_until_ms = 0;

    while (1) {
        uint32_t now = hal_ticks_ms();
        uint32_t written = pdm_mic_samples_written();

        if (state == STATE_IDLE) {
            if (now >= cooldown_until_ms && now >= next_vad_check_ms
                && written >= (uint32_t)VAD_WINDOW_SAMPLES) {
                next_vad_check_ms = now + VAD_CHECK_PERIOD_MS;

                uint32_t base = written - VAD_WINDOW_SAMPLES;
                int64_t sumsq = 0;
                for (int32_t i = 0; i < VAD_WINDOW_SAMPLES; i++) {
                    int32_t s = ring[(base + (uint32_t)i) % ring_capacity];
                    sumsq += (int64_t)s * s;
                }
                float rms = sqrtf((float)sumsq / (float)VAD_WINDOW_SAMPLES);

                if (rms > VAD_RMS_THRESHOLD) {
                    if (active_run_ms == 0) {
                        g_onset_sample = base;  /* first check that crossed threshold */
                    }
                    active_run_ms += VAD_CHECK_PERIOD_MS;

                    if (active_run_ms >= VAD_DEBOUNCE_MS) {
                        g_written_at_detection = pdm_mic_samples_written();
                        uart_puts("DETECTED\r\n");
                        /* Capture forward from the detected onset, not the most
                         * recent window ending now -- the latter is almost
                         * entirely pre-speech, since detection fires right as
                         * the word starts. */
                        capture_start = g_onset_sample;
                        state = STATE_CAPTURING;
                        active_run_ms = 0;
                    }
                } else {
                    active_run_ms = 0;
                }
            }
        } else { /* STATE_CAPTURING */
            uint32_t target = capture_start + WINDOW_SAMPLES;
            if ((int32_t)(written - target) >= 0) {
                for (uint32_t i = 0; i < WINDOW_SAMPLES; i++) {
                    window_scratch[i] = ring[(capture_start + i) % ring_capacity];
                }

                uint32_t t0 = hal_ticks_ms();
                spectrogram_compute(window_scratch, spectrogram_out);
                uint32_t t1 = hal_ticks_ms();

                Error set_err = method.set_input(EValue(input_tensor), 0);
                if (set_err != Error::Ok) {
                    uart_printf("ERROR: set_input failed (0x%x)\r\n", (unsigned)set_err);
                } else {
                    Error exec_err = method.execute();
                    uint32_t t2 = hal_ticks_ms();
                    if (exec_err != Error::Ok) {
                        uart_printf("ERROR: execute failed (0x%x)\r\n", (unsigned)exec_err);
                    } else {
                        const float *logits =
                            method.get_output(0).toTensor().const_data_ptr<float>();

                        int pred = 0;
                        for (int i = 1; i < WAKEWORD_NUM_CLASSES; i++) {
                            if (logits[i] > logits[pred]) pred = i;
                        }
                        float conf = softmax_confidence(logits, WAKEWORD_NUM_CLASSES, pred);
                        const char *label =
                            (conf >= CONF_THRESHOLD) ? WAKEWORD_CLASS_NAMES[pred] : "unknown";

                        uart_printf("DEBUG: onset=%u written=%u argmax=%s raw_conf=%f\r\n",
                                    (unsigned)g_onset_sample, (unsigned)g_written_at_detection,
                                    WAKEWORD_CLASS_NAMES[pred], (double)conf);
                        uart_printf("PREDICTION: %s conf=%f\r\n", label, (double)conf);
                        uart_printf("TIMING: spectrogram_ms=%u inference_ms=%u\r\n",
                                    (unsigned)(t1 - t0), (unsigned)(t2 - t1));
                    }
                }

                cooldown_until_ms = hal_ticks_ms() + COOLDOWN_MS;
                state = STATE_IDLE;
                active_run_ms = 0;
            }
        }
    }
    /* ===== END STUDENT SECTION ===== */
}
