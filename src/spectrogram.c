#include "spectrogram.h"
#include "spectrogram_norm_constants.h"
#include "fft.h"
#include <math.h>

/* On STM32: place scratch arrays in CCM RAM so they don't eat main SRAM
 * (this code runs from the main loop, not a DMA ISR, so CCM's CPU-only
 * accessibility is not a problem). See ../README.md "Memory budget". */
#ifdef TARGET_STM32
#define CCMRAM __attribute__((section(".ccmram")))
#else
#define CCMRAM
#endif

static CCMRAM fft_complex_t fft_buf[SPEC_FFT_LENGTH];
static CCMRAM float hann[SPEC_FRAME_LENGTH];
static CCMRAM float row_a[SPEC_N_FREQ_RAW];
static CCMRAM float row_b[SPEC_N_FREQ_RAW];

void spectrogram_init(void) {
    /* Periodic Hann window: w[n] = 0.5*(1 - cos(2*pi*n/N)), matching
     * torch.hann_window(SPEC_FRAME_LENGTH, periodic=True) (divides by N,
     * not N-1 — the "periodic" form used for STFT/spectral analysis). */
    for (int n = 0; n < SPEC_FRAME_LENGTH; n++) {
        hann[n] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * (float)n
                                       / (float)SPEC_FRAME_LENGTH));
    }
}

/* Computes |FFT| for one SPEC_FRAME_LENGTH-sample frame starting at
 * frame_index * SPEC_FRAME_STEP, writing SPEC_N_FREQ_RAW magnitude bins
 * into mag_out. int16 PCM is scaled to [-1, 1) first, matching the
 * waveform range tf.audio.decode_wav produces (which is what the TF
 * tutorial's tf.signal.stft operates on). */
static void stft_frame_magnitude(const int16_t *pcm, int frame_index, float *mag_out) {
    int start = frame_index * SPEC_FRAME_STEP;

    for (int n = 0; n < SPEC_FRAME_LENGTH; n++) {
        float s = (float)pcm[start + n] * (1.0f / 32768.0f);
        fft_buf[n].re = s * hann[n];
        fft_buf[n].im = 0.0f;
    }
    for (int n = SPEC_FRAME_LENGTH; n < SPEC_FFT_LENGTH; n++) {
        fft_buf[n].re = 0.0f;
        fft_buf[n].im = 0.0f;
    }

    fft512_forward(fft_buf);

    for (int k = 0; k < SPEC_N_FREQ_RAW; k++) {
        float re = fft_buf[k].re, im = fft_buf[k].im;
        mag_out[k] = sqrtf(re * re + im * im);
    }
}

void spectrogram_compute(const int16_t *pcm, float *out) {
    const float freq_scale = (float)SPEC_N_FREQ_RAW / (float)SPEC_OUT_FREQ;

    for (int j = 0; j < SPEC_OUT_FRAMES; j++) {
        /* Step 1: raw STFT magnitude for the two raw frames this output
         * row blends (see spectrogram.h step 2 for why this is exact). */
        stft_frame_magnitude(pcm, 2 * j, row_a);
        stft_frame_magnitude(pcm, 2 * j + 1, row_b);

        /* Step 2: time-axis resize -- fixed 0.5/0.5 average, in place. */
        for (int k = 0; k < SPEC_N_FREQ_RAW; k++) {
            row_a[k] = 0.5f * (row_a[k] + row_b[k]);
        }

        /* Step 3 + 4: freq-axis bilinear resize, then normalize. */
        float *out_row = out + j * SPEC_OUT_FREQ;
        for (int k = 0; k < SPEC_OUT_FREQ; k++) {
            float src = ((float)k + 0.5f) * freq_scale - 0.5f;
            if (src < 0.0f) src = 0.0f;
            if (src > (float)(SPEC_N_FREQ_RAW - 1)) src = (float)(SPEC_N_FREQ_RAW - 1);

            int i0 = (int)src;
            int i1 = i0 + 1;
            if (i1 > SPEC_N_FREQ_RAW - 1) i1 = SPEC_N_FREQ_RAW - 1;
            float w = src - (float)i0;

            float resized = (1.0f - w) * row_a[i0] + w * row_a[i1];
            out_row[k] = (resized - SPECTROGRAM_MEAN) / SPECTROGRAM_STD;
        }
    }
}
