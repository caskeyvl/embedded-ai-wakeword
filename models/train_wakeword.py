"""Training script for lab1_wakeword -- Task 1 in ../README.md.

Loads Google Speech Commands v0.02, builds a balanced 4-class
(yes/no/silence/unknown) dataset, computes spectrogram features via
spectrogram_features.py (verified against src/spectrogram.c -- see
../tools/compare_spectrogram.py), trains both ExactMatchCNN (host-only
accuracy reference) and ImprovedKWSNet (what actually ships), and writes:

  - ../src/spectrogram_norm_constants.h   (MEAN/STD from the training split)
  - ../submission/model.pte               (ImprovedKWSNet, ExecuTorch export)
  - ../submission/model_metrics.json      (accuracy/size for both models)

See ../README-training-parameters.md for the full spec this implements.

Usage:
    python3 models/train_wakeword.py                  # full run
    python3 models/train_wakeword.py --quick           # fast smoke test
"""
import argparse
import json
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectrogram_features as sf  # noqa: E402
from export_common import export_to_executorch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_WORDS = ("yes", "no")
N_CLASSES = 4
CLASS_NAMES = ("yes", "no", "silence", "unknown")  # index order -- must match src/wakeword_labels.h


# ═══════════════════════════════════════════════════════════════════════
# Model classes -- copied verbatim from README-training-parameters.md
# Appendix A. Nothing here is part of the assignment.
# ═══════════════════════════════════════════════════════════════════════

class ExactMatchCNN(nn.Module):
    """Literal port of the TF "simple_audio" tutorial's CNN head.
    HOST/ACCURACY-REFERENCE ONLY. Intentionally never embedded in the STM32
    firmware -- no quantization is introduced to force it to fit."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        flat_dim = 64 * 22 * 62  # 87,296
        self.fc1 = nn.Linear(flat_dim, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


class DSBlock(nn.Module):
    """Depthwise-separable conv block: DW 3x3 -> BN -> ReLU -> PW 1x1 -> BN -> ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = torch.relu(self.bn1(self.dw(x)))
        x = torch.relu(self.bn2(self.pw(x)))
        return x


class ImprovedKWSNet(nn.Module):
    """Depthwise-separable CNN with a global-average-pool head.
    THIS is the model embedded in firmware and actually flashed to hardware."""

    def __init__(self, n_classes: int = N_CLASSES, ch: int = 32, n_blocks: int = 4,
                 stem_kernel=(16, 8), stem_stride=(8, 8), stem_padding=(6, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=stem_kernel, stride=stem_stride, padding=stem_padding, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[DSBlock(ch) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════
# Dataset construction
# ═══════════════════════════════════════════════════════════════════════

def load_speechcommands(data_dir: str, subset: str):
    from torchaudio.datasets import SPEECHCOMMANDS
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return SPEECHCOMMANDS(root=data_dir, url="speech_commands_v0.02",
                           folder_in_archive="SpeechCommands", download=True, subset=subset)


def load_wav_int16(path: str) -> np.ndarray:
    """Reads a mono 16-bit PCM WAV via the stdlib `wave` module. Not
    torchaudio.load()/dataset[i]: this environment's torchaudio (2.11.0) has
    no working decode backend (routes through a separate `torchcodec`
    package that isn't installed) -- only the dataset object's file listing
    (._walker) is used, audio is decoded by hand instead."""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2, f"expected mono 16-bit PCM: {path}"
        assert wf.getframerate() == sf.SAMPLE_RATE, f"expected {sf.SAMPLE_RATE} Hz: {path}"
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2")


def pad_or_truncate(pcm: np.ndarray) -> np.ndarray:
    n = sf.NUM_SAMPLES
    if len(pcm) < n:
        out = np.zeros(n, dtype=np.int16)
        out[: len(pcm)] = pcm
        return out
    return pcm[:n].astype(np.int16)


def collect_word_examples(dataset, samples_per_class: int, rng: random.Random):
    """Returns dict: label -> list of int16 PCM arrays, for 'yes', 'no', and
    a pooled+shuffled 'unknown' bucket (every non-target word), each capped
    at samples_per_class. Pooling-then-shuffling for 'unknown' (rather than
    taking the first N words found) per README-training-parameters.md sec 1."""
    buckets = {"yes": [], "no": [], "unknown": []}
    for path in dataset._walker:
        label = os.path.basename(os.path.dirname(path))
        key = label if label in TARGET_WORDS else "unknown"
        buckets[key].append(path)

    out = {}
    for key in ("yes", "no", "unknown"):
        paths = buckets[key]
        rng.shuffle(paths)
        if len(paths) < samples_per_class:
            print(f"  warning: only {len(paths)} '{key}' examples available "
                  f"(wanted {samples_per_class}), using all of them")
        out[key] = [pad_or_truncate(load_wav_int16(p)) for p in paths[:samples_per_class]]
    return out


def collect_silence_examples(background_dir: Path, samples_per_class: int, rng: random.Random,
                              region: str):
    """Random 1.0s crops of _background_noise_/*.wav. `region` picks the first
    80% ("train") or last 20% ("val") of each file's samples, so train/val
    silence crops don't overlap."""
    wav_paths = sorted(background_dir.glob("*.wav"))
    assert wav_paths, f"no background noise wavs found under {background_dir}"

    clips = [load_wav_int16(str(p)) for p in wav_paths]

    n = sf.NUM_SAMPLES
    out = []
    for _ in range(samples_per_class):
        clip = clips[rng.randrange(len(clips))]
        total = len(clip)
        split = int(total * 0.8)
        if region == "train":
            lo, hi = 0, max(split - n, 1)
        else:
            lo, hi = split, max(total - n, split + 1)
        start = rng.randrange(lo, max(hi, lo + 1))
        out.append(pad_or_truncate(clip[start:start + n]))
    return out


def build_split(data_dir: str, subset: str, samples_per_class: int, rng: random.Random,
                 silence_region: str):
    print(f"Loading Speech Commands '{subset}' split (downloads on first run)...")
    dataset = load_speechcommands(data_dir, subset)
    background_dir = Path(dataset._path) / "_background_noise_"

    print(f"  collecting yes/no/unknown examples ({samples_per_class}/class)...")
    words = collect_word_examples(dataset, samples_per_class, rng)
    print(f"  collecting silence examples ({samples_per_class})...")
    silence = collect_silence_examples(background_dir, samples_per_class, rng, silence_region)

    pcm_list, label_list = [], []
    for label_idx, key in enumerate(CLASS_NAMES):
        items = silence if key == "silence" else words[key]
        pcm_list.extend(items)
        label_list.extend([label_idx] * len(items))
    return pcm_list, label_list


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction (the two-pass MEAN/STD procedure, section 2.2)
# ═══════════════════════════════════════════════════════════════════════

def compute_features_and_norm_stats(train_pcm):
    """Pass 1: unnormalized features for every training example, pooled into
    one MEAN/STD (section 2.2 step 5). Returns (raw_features, mean, std) --
    raw_features is cached and reused to build X_train (avoids recomputing
    the FFT a second time for the same examples)."""
    n = len(train_pcm)
    raw = np.empty((n, sf.OUT_FRAMES, sf.OUT_FREQ), dtype=np.float32)
    for i, pcm in enumerate(train_pcm):
        raw[i] = sf.compute_spectrogram_features(pcm, mean=0.0, std=1.0)
    mean = float(raw.mean())
    std = float(raw.std())
    return raw, mean, std


def make_tensors(pcm_list, label_list, mean, std, cached_raw=None):
    if cached_raw is not None:
        features = (cached_raw - mean) / std
    else:
        features = np.stack([
            sf.compute_spectrogram_features(pcm, mean=mean, std=std) for pcm in pcm_list
        ])
    X = torch.from_numpy(features).unsqueeze(1).float()  # (N, 1, 49, 128)
    y = torch.tensor(label_list, dtype=torch.long)
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, name,
                label_smoothing=0.1, weight_decay=0.0):
    device = torch.device("cpu")
    model.to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"[{name}] epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  gap={train_acc - val_acc:+.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--train-samples-per-class", type=int, default=1500)
    ap.add_argument("--val-samples-per-class", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model", choices=["improved", "exact", "both"], default="improved",
                     help="which model(s) to train (default: improved only -- fast iteration)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--export", action="store_true",
                     help="write norm constants, model.pte and model_metrics.json (needs --model both); "
                          "off by default so experiments never overwrite your submission")
    ap.add_argument("--quick", action="store_true",
                     help="fast smoke test: tiny sample counts + few epochs")
    args = ap.parse_args()
    if args.export and args.model != "both":
        ap.error("--export needs --model both (model_metrics.json requires both models' numbers)")
    sys.stdout.reconfigure(line_buffering=True)

    if args.quick:
        args.train_samples_per_class = 60
        args.val_samples_per_class = 20
        args.epochs = 3

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    submission_dir = PROJECT_ROOT / "submission"
    submission_dir.mkdir(exist_ok=True)

    # ── Build train/val splits ──────────────────────────────────────────
    train_pcm, train_labels = build_split(
        args.data_dir, "training", args.train_samples_per_class, rng, silence_region="train")
    val_pcm, val_labels = build_split(
        args.data_dir, "validation", args.val_samples_per_class, rng, silence_region="val")

    print(f"train: {len(train_pcm)} examples, val: {len(val_pcm)} examples")

    # ── Feature extraction + normalization stats (train split only) ────
    print("Computing spectrogram features + normalization stats (train split)...")
    train_raw, mean, std = compute_features_and_norm_stats(train_pcm)
    print(f"SPECTROGRAM_MEAN={mean:.6f}  SPECTROGRAM_STD={std:.6f}")

    X_train, y_train = make_tensors(train_pcm, train_labels, mean, std, cached_raw=train_raw)
    print("Computing spectrogram features (val split)...")
    X_val, y_val = make_tensors(val_pcm, val_labels, mean, std)

    if args.export:
        # ── Write normalization constants for the firmware build ───────────
        norm_path = PROJECT_ROOT / "src" / "spectrogram_norm_constants.h"
        norm_path.write_text(
            "#ifndef SPECTROGRAM_NORM_CONSTANTS_H\n"
            "#define SPECTROGRAM_NORM_CONSTANTS_H\n\n"
            "/* GENERATED FILE -- produced by models/train_wakeword.py. */\n\n"
            f"#define SPECTROGRAM_MEAN {mean}f\n"
            f"#define SPECTROGRAM_STD  {std}f\n\n"
            "#endif /* SPECTROGRAM_NORM_CONSTANTS_H */\n"
        )
        print(f"Wrote {norm_path}")


    example = (torch.randn(1, 1, sf.OUT_FRAMES, sf.OUT_FREQ),)
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              label_smoothing=args.label_smoothing, weight_decay=args.weight_decay)
    metrics = {}

    if args.model in ("exact", "both"):
        exact_model, exact_val_acc = train_model(
            ExactMatchCNN(), X_train, y_train, X_val, y_val, name="ExactMatchCNN", **kw)
        print(f"ExactMatchCNN best val_acc={exact_val_acc:.4f}")
        if args.export:
            _, exact_pte_bytes = export_to_executorch(
                exact_model, example, str(PROJECT_ROOT / "models" / "exact_match_cnn.pte"), print_ops=False)
            metrics.update({
                "exact_match_val_accuracy": exact_val_acc,
                "exact_match_param_count": sum(p.numel() for p in exact_model.parameters()),
                "exact_match_pte_bytes": exact_pte_bytes,
            })

    if args.model in ("improved", "both"):
        improved_model, improved_val_acc = train_model(
            ImprovedKWSNet(), X_train, y_train, X_val, y_val, name="ImprovedKWSNet", **kw)
        print(f"ImprovedKWSNet best val_acc={improved_val_acc:.4f}")
        if improved_val_acc < 0.60:
            print("WARNING: below the 0.60 sanity floor -- likely a pipeline bug, not under-tuning "
                  "(README-training-parameters.md section 4.3).")
        if args.export:
            _, improved_pte_bytes = export_to_executorch(
                improved_model, example, str(submission_dir / "model.pte"), print_ops=True)
            metrics.update({
                "improved_val_accuracy": improved_val_acc,
                "improved_param_count": sum(p.numel() for p in improved_model.parameters()),
                "improved_pte_bytes": improved_pte_bytes,
            })

    if args.export:
        metrics_path = submission_dir / "model_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote {metrics_path}")
    else:
        print("(experiment run: nothing written. Pass --model both --export for the final run.)")


if __name__ == "__main__":
    main()
