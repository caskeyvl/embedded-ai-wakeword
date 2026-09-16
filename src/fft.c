#include "fft.h"
#include <math.h>

void fft512_forward(fft_complex_t *x) {
    /* Bit-reversal permutation */
    int j = 0;
    for (int i = 1; i < FFT_N; i++) {
        int bit = FFT_N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            fft_complex_t t = x[i];
            x[i] = x[j];
            x[j] = t;
        }
    }

    /* Cooley-Tukey DIT butterflies */
    for (int len = 2; len <= FFT_N; len <<= 1) {
        float ang = -6.28318530717958647692f / (float)len; /* -2*pi/len */
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < FFT_N; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                fft_complex_t u = x[i + k];
                fft_complex_t v = {
                    cr * x[i + k + len / 2].re - ci * x[i + k + len / 2].im,
                    cr * x[i + k + len / 2].im + ci * x[i + k + len / 2].re
                };
                x[i + k]           = (fft_complex_t){ u.re + v.re, u.im + v.im };
                x[i + k + len / 2] = (fft_complex_t){ u.re - v.re, u.im - v.im };
                float tmp = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = tmp;
            }
        }
    }
}
