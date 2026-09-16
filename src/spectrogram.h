#ifndef SPECTROGRAM_H
#define SPECTROGRAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Audio -> spectrogram feature extraction.
 *
 * Feature representation: linear-magnitude Short-Time Fourier Transform
 * (STFT) — |STFT(waveform)| — resized and normalized. This is deliberately
 * NOT MFCC and NOT a mel filterbank: it mirrors the TensorFlow "simple_audio"
 * tutorial's tf.abs(tf.signal.stft(...)) as closely as possible, just with
 * different frame_length/frame_step/resize constants (see
 * ../README.md "Runtime-tunable parameters" and ../README-training-parameters.md).
 *
 * This header is the normative spec for the feature pipeline: the Python
 * training-side implementation (models/spectrogram_features.py) must
 * reproduce it bit-for-bit in intent, since a model is only useful on-device
 * if the features it sees at inference time match what it was trained on.
 * ═══════════════════════════════════════════════════════════════════════ */

#define SPEC_SAMPLE_RATE     16000  /* Hz */
#define SPEC_NUM_SAMPLES     16000  /* input window length: 1.0 s of audio */
#define SPEC_FRAME_LENGTH    480    /* STFT frame length, samples (30 ms) */
#define SPEC_FRAME_STEP      160    /* STFT hop, samples (10 ms) */
#define SPEC_FFT_LENGTH      512    /* next power of 2 >= SPEC_FRAME_LENGTH */

#define SPEC_N_FRAMES_RAW    98     /* 1 + (SPEC_NUM_SAMPLES-SPEC_FRAME_LENGTH)/SPEC_FRAME_STEP */
#define SPEC_N_FREQ_RAW      257    /* SPEC_FFT_LENGTH/2 + 1 */

#define SPEC_OUT_FRAMES      49     /* resized time dimension -> model input dim 0 */
#define SPEC_OUT_FREQ        128    /* resized freq dimension -> model input dim 1 */
#define SPEC_OUT_SIZE        (SPEC_OUT_FRAMES * SPEC_OUT_FREQ)  /* 6272 */

/* One-time setup: precomputes the Hann window table used by every frame.
 * Call once, before the first spectrogram_compute(). Not thread-safe/
 * reentrant (this project assumes a single-threaded bare-metal main loop). */
void spectrogram_init(void);

/* Computes a (SPEC_OUT_FRAMES x SPEC_OUT_FREQ) normalized STFT-magnitude
 * feature map from exactly SPEC_NUM_SAMPLES contiguous int16 PCM samples,
 * ready to feed directly as the model's input tensor.
 *
 * Pipeline (must match models/spectrogram_features.py bit-for-bit in intent):
 *
 *   1. Raw STFT magnitude: for each of SPEC_N_FRAMES_RAW=98 frames (Hann-
 *      windowed, SPEC_FRAME_LENGTH=480 samples, SPEC_FRAME_STEP=160-sample
 *      hop, zero-padded to SPEC_FFT_LENGTH=512 for the FFT), compute
 *      |FFT(frame)| and keep bins 0..SPEC_N_FREQ_RAW-1=256 (the
 *      non-redundant half-spectrum of a real-valued signal). No pre-
 *      emphasis, no log/dB compression — plain linear magnitude, matching
 *      tf.abs(tf.signal.stft(waveform, frame_length, frame_step)).
 *
 *   2. Time-axis resize, 98 -> 49: because 98 == 2*SPEC_OUT_FRAMES exactly,
 *      TF's half-pixel bilinear resize collapses to a fixed 0.5/0.5 average
 *      of adjacent raw-frame pairs (2*j, 2*j+1) for each output row j —
 *      it is NOT a general resize on this axis.
 *
 *   3. Freq-axis resize, 257 -> 128: SPEC_N_FREQ_RAW/SPEC_OUT_FREQ =
 *      257/128 is not an integer ratio, so this IS a general bilinear
 *      resize, using TensorFlow's half-pixel source-coordinate mapping
 *      for each output column k:
 *          src = (k + 0.5) * (SPEC_N_FREQ_RAW / SPEC_OUT_FREQ) - 0.5
 *          src = clamp(src, 0, SPEC_N_FREQ_RAW - 1)
 *      then linearly interpolate between floor(src) and floor(src)+1.
 *
 *   4. Normalize: out = (resized - SPECTROGRAM_MEAN) / SPECTROGRAM_STD,
 *      where MEAN/STD are fixed constants computed once over the training
 *      corpus by models/train_wakeword.py (analogous to Keras
 *      layers.Normalization.adapt()) and checked in to
 *      spectrogram_norm_constants.h.
 *
 * Implementation note: this function streams the computation two raw
 * frames at a time rather than materializing the full (98, 257) matrix
 * (~100 KB) — see spectrogram.c.
 *
 * pcm: pointer to exactly SPEC_NUM_SAMPLES contiguous int16 samples. This
 *      function does no ring-buffer indexing of its own — by design, so it
 *      stays trivially testable on a host with a flat WAV buffer. Callers
 *      reading from a circular mic buffer (see pdm_mic.h) must copy a
 *      linear window out first.
 * out: caller-owned buffer of SPEC_OUT_SIZE floats, row-major [time][freq],
 *      i.e. out[t * SPEC_OUT_FREQ + f] for t in [0,SPEC_OUT_FRAMES) and
 *      f in [0,SPEC_OUT_FREQ). This is exactly the model's expected input
 *      layout for a (1, 1, SPEC_OUT_FRAMES, SPEC_OUT_FREQ) tensor. */
void spectrogram_compute(const int16_t *pcm, float *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPECTROGRAM_H */
