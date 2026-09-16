#!/usr/bin/env python3
"""Numeric cross-check: does the compiled C spectrogram.c produce the same
feature map as the Python twin (models/spectrogram_features.py) on the same
WAV file? This is the hardware-independent correctness gate for the feature
pipeline described in ../src/spectrogram.h -- run it before trusting either
implementation, and especially before training a model against the Python
side (a mismatch here means the model will see different features on-device
than it was trained on).

Usage:
    python3 tools/compare_spectrogram.py [path/to/test.wav]

If no WAV path is given, generates a synthetic 1s 16kHz mono test tone
(a couple of sine tones + a little noise -- content doesn't matter for this
numeric check, only that it's a valid 16kHz mono 16-bit 1s clip).

Reads the current SPECTROGRAM_MEAN/SPECTROGRAM_STD out of
../src/spectrogram_norm_constants.h so the comparison is correct whether
run before training (placeholder mean=0/std=1) or after (real values).
"""
import re
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT / "models"))
import spectrogram_features as sf  # noqa: E402


def read_norm_constants():
    text = (SRC_DIR / "spectrogram_norm_constants.h").read_text()
    mean = float(re.search(r"SPECTROGRAM_MEAN\s+([-\d.eE]+)f", text).group(1))
    std = float(re.search(r"SPECTROGRAM_STD\s+([-\d.eE]+)f", text).group(1))
    return mean, std


def make_synthetic_test_wav(path: Path):
    sr = sf.SAMPLE_RATE
    n = sf.NUM_SAMPLES
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)
    waveform = (
        0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2 * np.pi * 1200.0 * t)
        + 0.02 * rng.standard_normal(n)
    )
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return pcm


def build_dump_tool():
    exe = TOOLS_DIR / "spectrogram_dump"
    src = [str(TOOLS_DIR / "spectrogram_dump.c"), str(SRC_DIR / "spectrogram.c"), str(SRC_DIR / "fft.c")]
    cmd = ["gcc", "-O2", "-I", str(SRC_DIR), "-o", str(exe)] + src + ["-lm"]
    print("Building:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return exe


def run_dump_tool(exe: Path, wav_path: Path) -> np.ndarray:
    result = subprocess.run([str(exe), str(wav_path)], check=True, stdout=subprocess.PIPE)
    raw = result.stdout
    expected_bytes = sf.OUT_FRAMES * sf.OUT_FREQ * 4
    if len(raw) != expected_bytes:
        raise RuntimeError(f"expected {expected_bytes} bytes from spectrogram_dump, got {len(raw)}")
    arr = np.frombuffer(raw, dtype="<f4").reshape(sf.OUT_FRAMES, sf.OUT_FREQ)
    return arr.copy()


def main():
    if not shutil.which("gcc"):
        print("error: gcc not found on PATH", file=sys.stderr)
        return 1

    if len(sys.argv) > 1:
        wav_path = Path(sys.argv[1])
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == sf.SAMPLE_RATE, \
                "test wav must be mono 16-bit 16kHz"
            pcm = np.frombuffer(wf.readframes(sf.NUM_SAMPLES), dtype="<i2")
    else:
        wav_path = TOOLS_DIR / "_synthetic_test_tone.wav"
        pcm = make_synthetic_test_wav(wav_path)
        print(f"No WAV given, generated synthetic test tone at {wav_path}")

    mean, std = read_norm_constants()
    print(f"Using SPECTROGRAM_MEAN={mean} SPECTROGRAM_STD={std} (from src/spectrogram_norm_constants.h)")

    exe = build_dump_tool()
    c_features = run_dump_tool(exe, wav_path)
    py_features = sf.compute_spectrogram_features(pcm, mean=mean, std=std)

    diff = np.abs(c_features - py_features)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(f"max abs diff:  {max_abs:.6g}")
    print(f"mean abs diff: {mean_abs:.6g}")

    tolerance = 1e-3
    if max_abs > tolerance:
        print(f"FAIL: max abs diff {max_abs:.6g} exceeds tolerance {tolerance:.1e}", file=sys.stderr)
        return 1

    print(f"PASS: C and Python spectrogram pipelines agree within {tolerance:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
