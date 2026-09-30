"""Python twin of ../src/spectrogram.c's feature extraction.

Turns one 1.0s window of raw int16 PCM into the same (OUT_FRAMES, OUT_FREQ)
linear-magnitude STFT feature map the firmware computes on-device. Plain STFT
magnitude -- NOT MFCC, NOT a mel filterbank, no log/dB compression -- mirrors
tf.abs(tf.signal.stft(...)) the same way src/spectrogram.c does. See
../README-training-parameters.md section 2 for the full spec this was
generated from, and ../src/spectrogram.h for the C implementation this must
match bit-for-bit in intent. Verified against the compiled C implementation
via ../tools/compare_spectrogram.py -- do not hand-edit the algorithm without
re-running that check.
"""
import numpy as np

# ── Parameters -- copied from src/spectrogram.h's SPEC_* macros -------------
SAMPLE_RATE = 16000    # SPEC_SAMPLE_RATE
NUM_SAMPLES = 16000    # SPEC_NUM_SAMPLES
FRAME_LENGTH = 480     # SPEC_FRAME_LENGTH
FRAME_STEP = 160       # SPEC_FRAME_STEP
FFT_LENGTH = 512       # SPEC_FFT_LENGTH
OUT_FRAMES = 49        # SPEC_OUT_FRAMES
OUT_FREQ = 128         # SPEC_OUT_FREQ

# ── Derived -- computed, not hardcoded (per README-training-parameters.md) --
N_FRAMES_RAW = 1 + (NUM_SAMPLES - FRAME_LENGTH) // FRAME_STEP   # 98
N_FREQ_RAW = FFT_LENGTH // 2 + 1                                # 257

assert N_FRAMES_RAW == 2 * OUT_FRAMES, \
    "time-axis resize assumes an exact 2:1 ratio (see step 3 below)"

# Periodic Hann window (divides by N, not N-1) -- matches
# torch.hann_window(FRAME_LENGTH, periodic=True) / src/spectrogram.c's hann[].
_HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME_LENGTH) / FRAME_LENGTH)

# Freq-axis half-pixel bilinear resize indices/weights (257 -> 128), fixed for
# every call -- precomputed once here rather than per-call.
_FREQ_SCALE = N_FREQ_RAW / OUT_FREQ
_k = np.arange(OUT_FREQ)
_src = (_k + 0.5) * _FREQ_SCALE - 0.5
_src = np.clip(_src, 0.0, N_FREQ_RAW - 1)
_I0 = np.floor(_src).astype(np.int64)
_I1 = np.minimum(_I0 + 1, N_FREQ_RAW - 1)
_W = _src - _I0


def compute_spectrogram_features(pcm, mean: float, std: float) -> np.ndarray:
    """pcm: int16, shape (NUM_SAMPLES,). Returns float32 NumPy array, shape
    (OUT_FRAMES, OUT_FREQ), row-major [time, freq]."""
    pcm = np.asarray(pcm)
    assert pcm.shape == (NUM_SAMPLES,), f"expected ({NUM_SAMPLES},), got {pcm.shape}"

    # Step 1: int16 -> float32 in [-1, 1), matching tf.audio.decode_wav.
    waveform = pcm.astype(np.float64) / 32768.0

    # Step 2: STFT magnitude. Frame manually (stride FRAME_STEP, length
    # FRAME_LENGTH) rather than torch.stft -- torch.stft centers the window
    # inside the FFT frame when win_length < n_fft, which does NOT match
    # tf.signal.stft / src/spectrogram.c (both window-then-pad-at-the-end).
    frames = np.lib.stride_tricks.sliding_window_view(waveform, FRAME_LENGTH)[::FRAME_STEP, :]
    assert frames.shape == (N_FRAMES_RAW, FRAME_LENGTH)

    windowed = frames * _HANN  # (N_FRAMES_RAW, FRAME_LENGTH)

    padded = np.zeros((N_FRAMES_RAW, FFT_LENGTH), dtype=np.float64)
    padded[:, :FRAME_LENGTH] = windowed  # zero-pad at the end, not centered

    spectrum = np.fft.rfft(padded, n=FFT_LENGTH, axis=1)  # (N_FRAMES_RAW, N_FREQ_RAW)
    raw = np.abs(spectrum)  # linear magnitude, no log/dB compression

    # Step 3: time-axis resize, 98 -> 49. N_FRAMES_RAW == 2*OUT_FRAMES exactly,
    # so TF's half-pixel bilinear resize collapses to a fixed 0.5/0.5 average
    # of adjacent raw-frame pairs (2*j, 2*j+1) -- not a generic downsample.
    resized_t = 0.5 * (raw[0::2, :] + raw[1::2, :])  # (OUT_FRAMES, N_FREQ_RAW)

    # Step 4: freq-axis resize, 257 -> 128. Not an integer ratio, so this is a
    # genuine bilinear resize using TF's half-pixel source-coordinate mapping
    # (precomputed _I0/_I1/_W above) -- NOT torch.nn.functional.interpolate /
    # PIL / cv2, whose default conventions don't match.
    resized = (1.0 - _W) * resized_t[:, _I0] + _W * resized_t[:, _I1]  # (OUT_FRAMES, OUT_FREQ)

    # Step 5: normalize with a single scalar mean/std shared across every
    # clip and every time/frequency position (see README-training-parameters.md
    # section 2.2 step 5 -- not per-clip, not per-bin).
    out = (resized - mean) / std
    return out.astype(np.float32)
