/* spectrogram_dump.c — standalone host tool (not part of the CMake build)
 * that runs the real ../src/spectrogram.c on a WAV file and writes the
 * resulting (SPEC_OUT_FRAMES x SPEC_OUT_FREQ) float32 feature map as raw
 * little-endian bytes to stdout. Used by compare_spectrogram.py to
 * numerically cross-check the C feature pipeline against the Python twin
 * (models/spectrogram_features.py) on identical input.
 *
 * Build (also done automatically by compare_spectrogram.py):
 *   gcc -O2 -I../src -o spectrogram_dump spectrogram_dump.c \
 *       ../src/spectrogram.c ../src/fft.c -lm
 *
 * Usage:
 *   ./spectrogram_dump input.wav > out.f32
 *
 * input.wav must be mono, 16-bit PCM, 16000 Hz, and contain at least
 * SPEC_NUM_SAMPLES (16000) samples -- only the first second is used. */

#include "spectrogram.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t read_u32le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t read_u16le(const unsigned char *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s input.wav > out.f32\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", argv[1]);
        return 1;
    }

    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "error: not a RIFF/WAVE file\n");
        return 1;
    }

    uint16_t num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    long data_offset = 0;
    uint32_t data_size = 0;

    for (;;) {
        unsigned char chunk_hdr[8];
        if (fread(chunk_hdr, 1, 8, f) != 8) break;
        uint32_t chunk_size = read_u32le(chunk_hdr + 4);

        if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
            unsigned char fmt[16];
            if (fread(fmt, 1, 16, f) != 16) { fprintf(stderr, "error: bad fmt chunk\n"); return 1; }
            num_channels = read_u16le(fmt + 2);
            sample_rate = read_u32le(fmt + 4);
            bits_per_sample = read_u16le(fmt + 14);
            if (chunk_size > 16) fseek(f, (long)(chunk_size - 16), SEEK_CUR);
        } else if (memcmp(chunk_hdr, "data", 4) == 0) {
            data_offset = ftell(f);
            data_size = chunk_size;
            break; /* stop at data; we don't need any later chunks */
        } else {
            fseek(f, (long)chunk_size + (long)(chunk_size & 1), SEEK_CUR);
        }
    }

    if (data_offset == 0) {
        fprintf(stderr, "error: no data chunk found\n");
        return 1;
    }
    if (num_channels != 1 || bits_per_sample != 16 || sample_rate != SPEC_SAMPLE_RATE) {
        fprintf(stderr, "error: expected mono 16-bit %d Hz, got %u ch, %u bit, %u Hz\n",
                SPEC_SAMPLE_RATE, num_channels, bits_per_sample, sample_rate);
        return 1;
    }

    uint32_t num_samples_available = data_size / 2;
    if (num_samples_available < SPEC_NUM_SAMPLES) {
        fprintf(stderr, "error: need >= %d samples, wav has %u\n", SPEC_NUM_SAMPLES, num_samples_available);
        return 1;
    }

    static int16_t pcm[SPEC_NUM_SAMPLES];
    fseek(f, data_offset, SEEK_SET);
    if (fread(pcm, sizeof(int16_t), SPEC_NUM_SAMPLES, f) != SPEC_NUM_SAMPLES) {
        fprintf(stderr, "error: short read on PCM data\n");
        return 1;
    }
    fclose(f);

    static float out[SPEC_OUT_SIZE];
    spectrogram_init();
    spectrogram_compute(pcm, out);

#ifdef _WIN32
    fprintf(stderr, "error: stdout binary mode not set up for _WIN32 in this tool\n");
    return 1;
#endif
    fwrite(out, sizeof(float), SPEC_OUT_SIZE, stdout);
    return 0;
}
