#ifndef FFT_H
#define FFT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal in-place complex FFT, sized for this project's 512-point STFT.
 * Adapted from ../../wakeword_et_dev/src/mfcc.c's fft512() (same radix-2
 * Cooley-Tukey decimation-in-time algorithm), split into its own module
 * since this project uses a plain STFT magnitude spectrogram, not MFCC. */

#define FFT_N 512

typedef struct {
    float re;
    float im;
} fft_complex_t;

/* In-place radix-2 DIT FFT of length FFT_N. x[] holds FFT_N complex samples
 * on input (imaginary parts should be 0 for real-valued audio frames) and
 * the complex spectrum on output, in natural (non-bit-reversed) order. */
void fft512_forward(fft_complex_t *x);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FFT_H */
