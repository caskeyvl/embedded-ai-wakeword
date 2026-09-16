# Training Parameters — Wake Word / Keyword-Spotting Model

**How to use this document**: this is the specification for Task 1 ("Train
both models") in `README.md` — not optional background reading, and not a
menu of things you could try if you're curious about the models or dataset.
Read it start to finish *before* writing any training code: it defines
exactly what the on-device firmware expects (the dataset, the feature
extraction/spectrogram pipeline, the model input/output contract, and the
ExecuTorch export procedure), and getting any of it wrong (especially §2,
the spectrogram) produces a model that trains fine but performs badly or
meaninglessly once embedded. Three pieces are given to you rather than
written from scratch, each for the same reason — they're numerically
fragile in ways that aren't the point of this assignment to rediscover: the
two model classes (Appendix A, copy them verbatim), the export helper
(Appendix A, same deal), and `compute_spectrogram_features` (§2 — generate
it with an AI coding assistant from the contract/algorithm spec given
there, then numerically verify it against the firmware's C implementation
and iterate until it matches; producing a *verified-correct* implementation
is the task, not hand-deriving the algorithm). Everything else — dataset
loading, the training loop, and calling the export helper — you write
yourself. Section 5 is the exception to all of this: it's genuinely
optional, for if you want to push past the baseline design for extra
accuracy/size tradeoffs.

Training is **not** wired into the CMake build — it's a standalone step you run
once, by hand, in a Python virtual environment. The build only needs the `.pte`
file it produces.

**Where your code goes**: this repo's `models/` directory ships empty. Put
your training script or notebook there under any name you like (e.g.
`models/train_wakeword.py` or `models/train_wakeword.ipynb`) — presence is
graded (see `README-submission-instructions.md`), content is not. Your
(AI-generated, then verified) spectrogram module must be named exactly
`models/spectrogram_features.py` and expose the constants/function
`tools/compare_spectrogram.py` imports (`SAMPLE_RATE`, `NUM_SAMPLES`,
`OUT_FRAMES`, `OUT_FREQ`, `compute_spectrogram_features(pcm, mean, std)`) —
that tool is how you verify it (required, not optional — see section 2
below); see section 2.2 for the exact contract.

## 1. Dataset

**Source**: [Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209),
loadable via `torchaudio.datasets.SPEECHCOMMANDS(url="speech_commands_v0.02",
folder_in_archive="SpeechCommands", download=True, subset=split)` with
`split` in `{"training", "validation", "testing"}`. This downloads/uses a
~2.4GB archive on first run — if you already have it (e.g. a classmate's copy,
or a prior project's `data/` directory), point `--data-dir` at that location
instead of re-downloading.

**Classes (4, in this exact index order — the model's output logit order)**:

| Index | Class     | Source |
|------:|-----------|--------|
| 0 | `yes`     | Speech Commands word "yes" |
| 1 | `no`      | Speech Commands word "no" |
| 2 | `silence` | synthesized (see below) |
| 3 | `unknown` | any other Speech Commands word |

This is the canonical binary keyword-spotting setup: `yes` and `no` are the
target commands (a subset of the 8 used by the
[TensorFlow "simple_audio" tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)),
plus two additional classes common practice for a deployable keyword spotter:

- **`silence`**: random 1.0-second crops of the `_background_noise_/*.wav`
  files that ship inside the Speech Commands archive (6 ambient recordings:
  dishes, cat meowing, exercise bike, pink noise, running tap, white noise).
  If a crop is shorter than 1.0s (near a clip's end), zero-pad it.
- **`unknown`**: a balanced random sample of recordings from all Speech
  Commands words that are *not* `yes` or `no` (e.g. `stop`, `up`, `left`,
  `right`, `go`, `down`, `bed`, `bird`, `cat`, `dog`, `eight`, `five`, ... —
  roughly 33 other words).

**Balancing**: cap every one of the 4 classes at the same `SAMPLES_PER_CLASS`
count (a script constant — the reference implementation uses 1500 for
training, 300 for validation), so the classifier isn't biased toward
over-represented classes. For `unknown`, first pool together indices from
*every* non-target word, shuffle, then take the first `SAMPLES_PER_CLASS` —
don't just take the first N words found.

## 2. Audio -> spectrogram feature extraction

**Your task here**: produce `models/spectrogram_features.py`, whose job is
to turn one window of raw PCM audio into the same `(49, 128)` feature map
the firmware computes on-device (`src/spectrogram.c`, provided) — the same
algorithm and the same numbers, just written in Python/PyTorch instead of C.
**Generate this with an AI coding assistant** rather than hand-deriving the
STFT/resize math yourself — give it the function contract (§2.2) and the
algorithm spec below (§2.1-§2.2) as its instructions. Your actual job is
what comes after generating a first draft: **verify it numerically** against
the firmware's compiled C implementation with `tools/compare_spectrogram.py`
(§7), and iterate — generate, run the check, look at what's wrong, fix it —
until it passes. Expect a first attempt to fail: this is the **most
important section to get exactly right**, and the pitfalls called out below
(§2.2 steps 2 and 4) are exactly the kind of subtle, plausible-looking
convention mismatch an AI assistant will readily produce unless specifically
told about them. A model is only useful on-device if the features it sees
at inference time match what it was trained on — a spectrogram
implementation that looks reasonable but doesn't actually match will train
and validate just fine while performing badly (or randomly) once flashed,
and nothing about training itself will tell you that's what happened.

**This project uses plain STFT magnitude — NOT MFCC, NOT a mel filterbank.**
No log/dB compression either. This deliberately mirrors the TF tutorial's
`spectrogram = tf.abs(tf.signal.stft(waveform, frame_length, frame_step))` as
closely as possible; only the frame/step/resize constants differ from the
tutorial's defaults.

### 2.1 Parameters — what each one controls

The actual numeric values live in exactly one place —
[`src/spectrogram.h`](src/spectrogram.h), `#define`d with a `SPEC_` prefix
(e.g. `SPEC_FRAME_LENGTH`), which is what the compiled C implementation is
built against. Copy each value from there into a same-named (unprefixed)
Python constant in your module — that's the exact name `tools/compare_spectrogram.py`
imports (§2.2) — rather than retyping the numbers here, so there's only one
place they can go stale:

| Python name | `spectrogram.h` macro | What it controls |
|---|---|---|
| `SAMPLE_RATE` | `SPEC_SAMPLE_RATE` | Mic sampling rate — the real-world time scale every sample-count parameter below is defined against. |
| `NUM_SAMPLES` | `SPEC_NUM_SAMPLES` | Length of one inference window; fixed at 1.0s of audio (matches the firmware's `WINDOW_SAMPLES`, see `README-testing.md`'s "Runtime-tunable parameters"). |
| `FRAME_LENGTH` | `SPEC_FRAME_LENGTH` | Samples per STFT frame — the core time/frequency-resolution tradeoff: a longer frame resolves frequency more finely but blurs *when* something happened. Also the length of the Hann window applied to each frame. |
| `FRAME_STEP` | `SPEC_FRAME_STEP` | Hop between consecutive frames (smaller than `FRAME_LENGTH`, so frames overlap). Together with `FRAME_LENGTH`/`NUM_SAMPLES`, determines `N_FRAMES_RAW` below. |
| `FFT_LENGTH` | `SPEC_FFT_LENGTH` | Size the windowed frame is zero-padded up to before the FFT — must be `>= FRAME_LENGTH` (a power of two, for an efficient FFT); padding interpolates the frequency axis more finely than a bare `FRAME_LENGTH`-point FFT would. |
| `N_FRAMES_RAW` | `SPEC_N_FRAMES_RAW` | **Derived, don't hardcode**: `1 + (NUM_SAMPLES - FRAME_LENGTH) // FRAME_STEP` — how many STFT frames one window produces. |
| `N_FREQ_RAW` | `SPEC_N_FREQ_RAW` | **Derived**: `FFT_LENGTH // 2 + 1` — the non-redundant half of a real-signal FFT's output. |
| `OUT_FRAMES` | `SPEC_OUT_FRAMES` | The model's input time dimension after resizing — fixed by the CNN architecture (Appendix A expects `(1, 1, OUT_FRAMES, OUT_FREQ)`), independent of the raw STFT shape. |
| `OUT_FREQ` | `SPEC_OUT_FREQ` | The model's input frequency dimension after resizing, same reasoning as `OUT_FRAMES`. |

Compute `N_FRAMES_RAW`/`N_FREQ_RAW` from the other constants in your Python
module rather than copying their numbers too — one fewer place a typo can
hide. (Concretely, with this project's parameters, `N_FRAMES_RAW=98` and
`N_FREQ_RAW=257` — mentioned here only so the numbers in the pipeline
description below make sense, not as values to type in anywhere.)

### 2.2 Pipeline (apply in this order)

**Function to generate**: `compute_spectrogram_features(pcm, mean, std)`.
Give an AI assistant the contract table and the five numbered steps below
as its spec — the more of this section you feed it verbatim (rather than
summarizing from memory), the less likely it reaches for a plausible but
wrong default (§2.2 steps 2 and 4 call out the two most likely ones).

| | Type / shape | Meaning |
|---|---|---|
| `pcm` (in) | `int16`, length `NUM_SAMPLES` | One linear window of raw PCM — the same int16 samples the firmware reads out of its mic ring buffer. **Not** a `torchaudio`-loaded float32 waveform — see "How this gets called" below if your dataset loader hands you one of those instead. |
| `mean`, `std` (in) | `float` | Normalization constants from step 5. During the one-time statistics pass (below) you call this with `mean=0.0, std=1.0` (a no-op) to get *unnormalized* features; every other call uses the real computed values. |
| return (out) | `float32`, shape `(OUT_FRAMES, OUT_FREQ)` = `(49, 128)` | The finished feature map, row-major `[time, freq]`. Return a plain NumPy array (`.numpy()` at the end if you compute internally in PyTorch) — `tools/compare_spectrogram.py` diffs this directly against the compiled C output with NumPy (§7). |

That exact name/signature is what `tools/compare_spectrogram.py` (provided)
imports to numerically cross-check your implementation against the
firmware's compiled C version.

**How this gets called**: `compute_spectrogram_features` isn't run
standalone — your own training script calls it (and so does
`tools/compare_spectrogram.py`, for verification), in two different ways,
both ahead of the actual training loop:

```python
# models/train_wakeword.py (sketch of the CALL SITES, not the whole script)
import numpy as np
import torch
import spectrogram_features as sf

# --- Pass 1: one-time normalization statistics (step 5). ----------------
# Every value from every clip's *unnormalized* feature map, pooled into one
# collection across the whole training split -- see step 5 for why this has
# to be a single pair of numbers, not per-clip or per-bin statistics.
raw_values = []
for pcm in training_split_pcm_clips():                    # int16, NUM_SAMPLES each
    raw_values.append(sf.compute_spectrogram_features(pcm, mean=0.0, std=1.0))
MEAN, STD = float(np.mean(raw_values)), float(np.std(raw_values))
# write MEAN/STD into src/spectrogram_norm_constants.h here -- see §2.4

# --- Pass 2: build the actual (now correctly normalized) training data. -
# make_example is called once per (pcm, label) pair, driven over the whole
# split, to build the tensors your training loop actually trains on:
def make_example(pcm, label):
    features = sf.compute_spectrogram_features(pcm, mean=MEAN, std=STD)   # (49, 128)
    x = torch.from_numpy(features).unsqueeze(0)                          # (1, 49, 128) -- add the model's channel dim
    return x, label

train_examples = [make_example(pcm, label) for pcm, label in training_split_pcm_and_labels()]
X_train = torch.stack([x for x, _ in train_examples])   # (N_train, 1, 49, 128)
y_train = torch.tensor([label for _, label in train_examples])
# ... same pattern for the validation split (with the same MEAN/STD -- never
# recompute statistics from the validation data). X_train/y_train (typically
# wrapped in a TensorDataset + DataLoader for batching) are what your
# training loop actually consumes from here -- ordinary PyTorch training,
# nothing spectrogram-specific left.
```

The two-pass structure (once with `mean=0/std=1` to gather statistics, once
for real) is the whole reason `compute_spectrogram_features` takes
`mean`/`std` as *arguments* rather than baking in constants directly — step
5 needs the resized-but-not-yet-normalized values to compute them, and this
is how you get those without writing two separate functions.

**One integration detail worth flagging**: `torchaudio.load()` with its
defaults returns an already-normalized `float32` waveform in `[-1, 1]`, not
raw `int16` PCM — `compute_spectrogram_features` expects the latter (step 1
below does that exact int16 -> float32 conversion itself, matching what the
on-device mic driver's samples look like). Either load PCM without that
normalization, or convert back (`(waveform * 32768).round().to(torch.int16)`)
before calling this function — don't double-normalize.

Internally, `compute_spectrogram_features` is the five steps below chained
together. You don't need to structure the generated code exactly like this,
but the shape at each stage is a useful thing to state in your prompt *and*
to sanity-check the result against — a shape mismatch at any stage means
something upstream is already wrong, before you even get to
`tools/compare_spectrogram.py`'s numeric diff:

```python
def compute_spectrogram_features(pcm, mean: float, std: float):
    """pcm: int16, shape (NUM_SAMPLES,). Returns float32 NumPy array, shape (OUT_FRAMES, OUT_FREQ)."""
    waveform  = ...   # step 1: int16 -> float32 in [-1, 1)
    raw       = ...   # step 2: (N_FRAMES_RAW, N_FREQ_RAW) STFT magnitude
    resized_t = ...   # step 3: (OUT_FRAMES, N_FREQ_RAW) -- time axis resized
    resized   = ...   # step 4: (OUT_FRAMES, OUT_FREQ) -- freq axis resized
    out = (resized - mean) / std   # step 5
    return out   # as a float32 NumPy array
```

1. **Waveform**: convert int16 PCM to float32 in `[-1, 1)` via `x / 32768.0`
   (matches what `tf.audio.decode_wav` produces for 16-bit PCM input — no
   other normalization).
2. **STFT magnitude**: for each of the `N_FRAMES_RAW` frames (stride
   `FRAME_STEP`, length `FRAME_LENGTH`), multiply by a periodic Hann window
   (`w[n] = 0.5 - 0.5*cos(2*pi*n/FRAME_LENGTH)`, i.e.
   `torch.hann_window(FRAME_LENGTH, periodic=True)` / `tf.signal.hann_window(FRAME_LENGTH,
   periodic=True)` — divides by `N`, not `N-1`), zero-pad the *windowed*
   frame to `FFT_LENGTH` samples **at the end** (not centered), take the
   FFT, and keep `|FFT|` for bins `0..N_FREQ_RAW-1` (the non-redundant half
   of a real-signal spectrum). Result: `(N_FRAMES_RAW, N_FREQ_RAW)` — i.e.
   `(98, 257)` with this project's parameters — matrix of non-negative
   floats.

   **Watch for `torch.stft` in the generated code — it will not match this
   spec.** `torch.stft` requires `win_length <= n_fft`, and when they differ
   (as here: `win_length=FRAME_LENGTH < n_fft=FFT_LENGTH`) it *centers* the
   window inside the FFT frame — splitting the zero-padding across both
   ends — rather than windowing the raw frame and padding the remainder at
   the end, which is what `tf.signal.stft` and this project's C
   implementation (`src/spectrogram.c`) both do. That's a real, silent
   numerical mismatch, not a style preference — the same class of
   convention problem as the resize step below (step 4) — and it's exactly
   what `tools/compare_spectrogram.py` would catch as a spectrogram diff.
   `torch.stft` is a very likely first guess for an AI assistant asked to
   compute an STFT in PyTorch, since it's the "obvious" library call — if
   your generated code uses it and the numeric check fails, this is the
   first thing to suspect. Either point your assistant at this exact
   constraint and ask it to frame/window/pad manually instead (via
   `Tensor.unfold`, `torch.hann_window`, `F.pad`, and `torch.fft.rfft`), or
   make that fix yourself once you see the mismatch.
3. **Time-axis resize, `N_FRAMES_RAW` → `OUT_FRAMES`**: because
   `N_FRAMES_RAW == 2 * OUT_FRAMES` **exactly** (`98 == 2*49`), a general
   bilinear resize collapses to a fixed 0.5/0.5 average of adjacent frame
   pairs: `out[j, :] = 0.5 * (raw[2*j, :] + raw[2*j+1, :])` for `j in
   range(OUT_FRAMES)`. This is *not* a coincidence to work around — it's a
   direct consequence of TF's half-pixel bilinear resize formula (§2.3)
   applied to an exact 2:1 downscale, and the reference C implementation
   relies on it (no generic resize library is used for this axis). Give
   your assistant this exact formula rather than "downsample 98 to 49" —
   a generic downsample (e.g. a strided slice, or a general resize call)
   won't reliably reproduce this averaging behavior.
4. **Freq-axis resize, `N_FREQ_RAW` → `OUT_FREQ`**: `N_FREQ_RAW / OUT_FREQ`
   (`257 / 128`) is *not* an integer ratio, so this is a genuine bilinear
   resize. For each output column `k in range(OUT_FREQ)`:
   ```
   scale = N_FREQ_RAW / OUT_FREQ
   src   = (k + 0.5) * scale - 0.5             # TF's half-pixel source mapping
   src   = clamp(src, 0, N_FREQ_RAW - 1)
   i0    = floor(src);  i1 = min(i0 + 1, N_FREQ_RAW - 1);  w = src - i0
   out[:, k] = (1 - w) * in[:, i0] + w * in[:, i1]
   ```
   **Watch for a generic resize call in the generated code — it's the
   likeliest thing to be wrong here.** TF's `tf.image.resize` / Keras
   `layers.Resizing` (default `method="bilinear"`, `antialias=False`) uses
   exactly this half-pixel convention. Framework resize functions
   (`torch.nn.functional.interpolate`, `PIL`, `cv2`) don't always default to
   the same convention (align-corners vs. half-pixel differ subtly), and an
   AI assistant asked to "resize 257 to 128" is likely to reach for one of
   those with whatever its default arguments are, not this exact formula.
   Give it the formula above explicitly rather than a verbal description of
   the resize, and treat any resize call that isn't visibly implementing
   this exact half-pixel math as suspect. `tools/compare_spectrogram.py` is
   the tool that catches a mismatch here: it numerically diffs the compiled
   C implementation against your `models/spectrogram_features.py`.
5. **Normalize**: `out = (resized - MEAN) / STD`. **`MEAN` and `STD` are
   each a single scalar number — not one per clip, not one per time frame
   or frequency bin — shared by every spectrogram this function will ever
   compute, for any clip, at both training and inference time.** They are
   *not* fixed constants you look up; you compute them yourself, exactly
   once, before you can normalize anything: run every clip in your
   **entire training split** through this function with `mean=0, std=1`
   (a no-op, so you get back *unnormalized* features), pool every value
   from every one of those `(OUT_FRAMES, OUT_FREQ)` feature maps together
   — across every clip *and* every time/frequency position, all of it, one
   giant collection of numbers — and take that whole collection's mean and
   standard deviation. That single pair of numbers is `MEAN`/`STD` (exactly
   what Keras `layers.Normalization.adapt()` does with a single-channel
   input, and exactly what "Pass 1" in "How this gets called" above
   computes). Do not compute per-batch, per-sample, or per-bin statistics,
   and don't include the validation split.

### 2.3 Why resize lives in feature extraction, not the model

The TF tutorial applies `layers.Resizing` as a Keras *model* layer. This
project instead performs resize (and normalization) as part of feature
extraction, both on the training side (`models/spectrogram_features.py`) and
the on-device side (`src/spectrogram.c`, provided) — so the exported model's
input is always a fixed `(1, 1, 49, 128)` tensor, and the ExecuTorch graph
contains no resize op at all. This sidesteps two embedded-specific concerns:
whether a resize operator is available/efficient in ExecuTorch's portable
kernel set, and keeping flash usage predictable. Functionally it's equivalent
to the tutorial's approach as long as the *exact same* resize math is applied
consistently at training and inference time (§2.2) — which is why this
document specifies the resize algorithm in such precise, from-scratch detail
rather than "just use your framework's resize function."

### 2.4 Required output for the firmware build

Your training script must write the computed `MEAN`/`STD` — the single
scalar pair from step 5 above, pooled across your entire training split,
not a per-clip or per-bin value — to `src/spectrogram_norm_constants.h` in
this format (regenerating the file each training run):

```c
#define SPECTROGRAM_MEAN <value>f
#define SPECTROGRAM_STD  <value>f
```

The provided placeholder in that file (`MEAN=0, STD=1`, a no-op
normalization) lets the project compile before you've trained anything, but
predictions will be meaningless until you regenerate it. The CMake build
recompiles `src/spectrogram.c` against whatever is currently in that file —
there's no separate registration step.

## 3. Model input/output contract

| | Shape | dtype |
|---|---|---|
| Input | `[1, 1, 49, 128]` | `float32` |
| Output | `[1, 4]` | `float32` (raw logits, not softmax — apply softmax/argmax yourself if you want a probability) |

Input layout is row-major `[batch, channel, time, freq]`, i.e. element
`(t, f)` of the `(49, 128)` feature map from §2 goes at flat index
`t * 128 + f` within the single input channel. Output class order is the
table in §1.

## 4. Two models: a baseline and a challenge

This assignment asks you to train **two** models — both class definitions are
given to you verbatim in Appendix A, so this section is about what to do with
them, not how to define them.

### 4.1 `ExactMatchCNN` — literal tutorial port (baseline, host-only)

A literal port of the TF tutorial's CNN head:
`Conv2D → Conv2D → MaxPool2D → Dropout → Flatten → Dense(128) → Dropout →
Dense(4)`, with `Resizing`/`Normalization` removed (already handled in
feature extraction per §2.3).

**This model does not fit on the STM32F407, and that's expected, not a bug
to fix.** The tutorial's own `Resizing(32, 32)` step keeps its Flatten layer
small (`64 * 13 * 13 = 10,816` elements); this assignment's larger
`(49, 128)` spectrogram, run through the same `Conv2D → Conv2D → MaxPool2D`
head with no resizing, flattens to `64 * 22 * 62 = 87,296` elements instead.
The very next layer, `Dense(87296 → 128)`, is alone **~11.2 million
parameters** (~44MB at float32) — more than 40x the STM32F407's entire 1MB
flash, even before ExecuTorch's runtime overhead. Do **not** introduce
quantization to force this to fit (that's a separable, later concern) — this
model exists purely as a **host-side accuracy reference**: train it, export
it, run it against WAV files on your workstation, and see what a faithful,
un-adapted tutorial port achieves on this larger-spectrogram problem. Do not
attempt to flash it.

### 4.2 `ImprovedKWSNet` — embedded-appropriate design (what actually ships)

A small depthwise-separable CNN with a global-average-pool head instead of a
large Flatten→Dense (see Appendix A for the exact code):

```
Conv2d(1->ch, kernel=(16,8), stride=(8,8), padding=(6,2)) -> BatchNorm -> ReLU   (stem)
[DepthwiseConv2d(3x3) -> BN -> ReLU -> PointwiseConv2d(1x1) -> BN -> ReLU] x n_blocks
AdaptiveAvgPool2d(1)
Linear(ch -> 4)
```

Default `ch=32`, `n_blocks=4` (10,052 parameters, `.pte` ≈ 55-60KB). The key
architectural idea: a depthwise-separable block's parameter count depends only
on the channel width, not on spatial resolution, and the global-average-pool
head has no Flatten step at all — so however large the input spectrogram is,
the parameter count (and thus flash footprint) stays small. This is the model
that gets embedded in firmware and actually runs on hardware — export it to
`submission/model.pte`, which is what `CMakeLists.txt`'s `WAKEWORD_PTE` cache
variable points at by default (see §6).

**Why the stem uses stride 8, not something milder**: ExecuTorch's memory
planner sizes the on-device activation buffer to the model's peak
concurrently-live tensor pair. A milder stem stride (e.g. matching the
tutorial's implicit ~2x downsampling) leaves the depthwise-separable blocks
operating on a large `(ch, 24, 64)` feature map, which needs **~290KB** of
activation-buffer RAM — more than the entire STM32F407's 128KB of main SRAM.
Stride 8 brings that down to `(ch, 6, 16)`, needing only ~30-40KB. If you
change the stem shape, **re-measure** before assuming your variant fits (see
`README-testing.md`'s "Memory budget" for the how-to — it's a five-line
Python snippet against the exported `.pte`, not a guess).

### 4.3 What to expect

A faithful tutorial port (`ExactMatchCNN`) generally scores a few points
higher on validation accuracy than the embeddable `ImprovedKWSNet` — more
capacity generally helps on the same distribution it was trained on. That's
the point of training both: raw capacity and on-device feasibility are
different axes, and a well-designed small model can give up a modest amount
of accuracy for an enormous size reduction. Report both models' accuracy and
size in `submission/model_metrics.json` (see
`README-submission-instructions.md`) — there's no fixed target beyond a
sanity floor (a correctly-trained `ImprovedKWSNet` should clear 60% val
accuracy by a wide margin; if you're near chance, something in your
feature-extraction or training pipeline is broken, not just under-tuned).

`no` and `unknown` tend to be the weakest classes for both models — `no` is
short and acoustically close to a few other words that now fall into
`unknown`, and `unknown` is inherently the hardest class (it has to reject
everything that isn't `yes`, `no`, or silence, drawn from ~33 different
words). Narrowing the target vocabulary to just `yes`/`no` (rather than a
larger multi-word command set) generally raises both models' overall
validation accuracy substantially, at the cost of a less interesting
classification problem — a real accuracy/scope tradeoff worth noticing.

## 5. Levers if you want to design your own model (challenge)

The `ExactMatchCNN` vs. `ImprovedKWSNet` comparison above is meant to make a
point: a literal tutorial port doesn't survive contact with a 1MB-flash
target, but a modest architectural adaptation (depthwise-separable
convolutions + global average pooling) gets you something that both fits
*and* achieves reasonable accuracy. Things worth trying, roughly in order of
likely impact:

- **Channel width (`ch`) and block count (`n_blocks`)**: the most direct
  accuracy/size dial. Re-measure the activation buffer size (§4.2) after any
  change — it scales with `ch`, not `n_blocks`.
- **Data augmentation**: time-shift the 1-second window randomly by a few
  hundred ms, mix in background noise at a few SNR levels, or apply
  SpecAugment-style random time/frequency masking directly on the resized
  `(49, 128)` feature map before normalization. The `unknown`/`no`
  classes are the most likely to benefit.
- **`SAMPLES_PER_CLASS`**: more training data per class, especially for
  `unknown` (currently sampling a fixed count from ~33 pooled words).
- **Label smoothing** on the `CrossEntropyLoss` — often helps with a
  many-way, imbalanced-difficulty classification problem like this one.
- **Stem kernel/stride shape**: the specific `(16,8)`/`(8,8)`/`(6,2)` choice
  in §4.2 was picked to hit a memory budget, not because it's provably
  optimal — a differently-shaped stem that still lands in a similar
  activation-buffer budget is fair game.
- **Swapping `AdaptiveAvgPool2d` for a small `Flatten` + `Linear` head**: this
  usually helps accuracy some, at a real parameter-count cost — a good way
  to *feel* the Flatten-vs-GlobalAvgPool tradeoff from §4.1 directly, at a
  much smaller scale than the exact-match model's blowup.

**Checking your flash budget** before assuming a variant fits: build the
`stm32` CMake preset (`cmake --build --preset stm32`) and read the
`arm-none-eabi-size` output it prints — flash usage is the `text` column,
must stay comfortably under 1MB alongside the ExecuTorch runtime and firmware
code (~150-220KB baseline, see `README-testing.md`). For the *RAM/CCM*
budget (the usual binding constraint here, not flash), see
`README-testing.md`'s "Memory budget".

## 6. Export procedure

Use the helper below (Appendix A) from your own training script:

```python
from export_common import export_to_executorch  # copy this file's function into your own script

model.eval()  # MANDATORY -- folds BatchNorm into conv weights
example_input = (torch.randn(1, 1, 49, 128),)
export_to_executorch(model, example_input, "submission/model.pte")
```

This prints the operator list your exported graph needs (best-effort,
`.out`-suffixed) — this is informational, for whoever (re)builds ExecuTorch
itself. See `README-testing.md`'s "Building/rebuilding ExecuTorch" for the
exact `cmake -DEXECUTORCH_SELECT_OPS_LIST=...` step, and rebuild if you see
`OperatorMissing` (error code `0x14`) when loading the model at runtime.

**Output filename the build/submission expects**: `submission/model.pte` for
the `ImprovedKWSNet` export (see `README-submission-instructions.md` for the
full submission layout; `CMakeLists.txt`'s `WAKEWORD_PTE` cache variable
points at it by default). `ExactMatchCNN`'s `.pte` isn't consumed by the
embedded build at all — export it somewhere under `models/` for your own
host-side (`./build/host/inference_host`) accuracy comparison; only its
*numbers* (accuracy, param count, byte size) need to make it into
`submission/model_metrics.json`, not the file itself.

## 7. Numerically verifying your spectrogram implementation

This is the verification loop §2 asks you to run against your (likely
AI-generated) `models/spectrogram_features.py` — not a one-time formality,
and not optional. Run it before trusting any training run:

```bash
python3 tools/compare_spectrogram.py                 # synthetic test tone
python3 tools/compare_spectrogram.py your_test.wav    # or a real clip
```

This builds and runs the provided C `spectrogram.c`/`fft.c` against the same
input and numerically diffs the two feature maps. A mismatch here means your
model will see different features on-device than it was trained on — a
correctness bug worth catching before you spend time training, not after.

If it fails: read `max abs diff` before touching anything. A diff on the
order of the feature values themselves (not a tiny floating-point rounding
difference) usually means an algorithm-level mismatch, not a numerical
precision issue — most often the `torch.stft` or generic-resize pitfalls
called out in §2.2. Feed the failure back to whatever generated the code
(the diff value, and the specific §2.2 step you suspect) and regenerate that
part, rather than starting over from scratch.

## Appendix A: provided model code and export helper

Copy these into your own file(s) under `models/` — they are the exact
`torch.nn.Module` definitions and export helper this document is written
around; nothing about them is part of the assignment.

### `ExactMatchCNN`, `DSBlock`, `ImprovedKWSNet`

```python
import torch
import torch.nn as nn

OUT_FRAMES = 49
OUT_FREQ = 128
N_CLASSES = 4


class ExactMatchCNN(nn.Module):
    """Literal port of the TF "simple_audio" tutorial's CNN head:
    Conv2D -> Conv2D -> MaxPool2D -> Dropout -> Flatten -> Dense(128) ->
    Dropout -> Dense(n_classes) (n_classes=4 here). The tutorial applies this after resizing
    its spectrogram down to (32, 32); this project's spectrogram resizes to
    the larger (49, 128) instead, so the Flatten output here is
    64*22*62 = 87,296 elements instead of the tutorial's 64*13*13 = 10,816
    -- and the first Dense layer alone becomes a ~11.2M-parameter (~44MB
    fp32) matrix.

    HOST/ACCURACY-REFERENCE ONLY. Intentionally never embedded in the STM32
    firmware -- no quantization is introduced to force it to fit.
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        # (49,128) -conv3x3-> (47,126) -conv3x3-> (45,124) -pool2-> (22,62)
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
    """Depthwise-separable conv block: DW 3x3 -> BN -> ReLU -> PW 1x1 -> BN -> ReLU.
    Parameter count depends only on `channels`, not on spatial size -- this
    is what keeps ImprovedKWSNet small in flash regardless of the (49,128)
    input's larger spatial footprint than a typical MFCC feature map."""

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
    """Depthwise-separable CNN with a global-average-pool head -- no large
    Flatten->Dense anywhere, so parameter count (and therefore .pte size)
    stays in the tens of thousands regardless of spatial input size. Fits
    STM32F407 flash comfortably in plain fp32, no quantization needed.

    THIS is the model embedded in firmware and actually flashed to hardware.

    Input:  [B, 1, 49, 128]
    Output: [B, n_classes] logits

    Stem kernel/stride note: this project's (49,128) spectrogram is much
    larger spatially than a typical MFCC feature map, so a mild stem stride
    leaves DS blocks operating on a large activation map -- e.g.
    stride=(2,2) here leaves a (ch,24,64) map, and ExecuTorch's memory
    planner needs ~290KB of activation-buffer RAM to run it, which does not
    fit in the STM32F407's 128KB main SRAM alongside the mic-window buffer
    and ExecuTorch's method_pool (see README-testing.md "Memory budget").
    stride=(8,8) brings the stem straight down to a (ch,6,16) map (~30-40KB
    activation buffer for ch in [16,32]) while still leaving the DS blocks a
    non-trivial 96-position spatial map to work with -- this is the binding
    constraint that sets the default stem shape below, not an accuracy
    consideration. Changing this? Re-measure non_const_buffer_sizes before
    assuming a variant fits on-device (see README-testing.md).

    Levers for your own variant: channel width `ch`, block count `n_blocks`,
    stem kernel/stride (mind the memory-budget note above), or swapping
    AdaptiveAvgPool2d for a small Flatten+Linear head (and watching the size
    cost return).
    """

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
```

### `export_to_executorch` (export helper)

```python
import os

import torch


def export_to_executorch(model: torch.nn.Module, example_input: tuple, pte_path: str, print_ops: bool = True):
    """model.eval() -> torch.export.export -> exir.to_edge -> to_executorch,
    writes the .pte to pte_path. Returns (exported_program, size_bytes).

    Prints the full operator list used by the exported graph -- this is
    informational only (for setting EXECUTORCH_SELECT_OPS_LIST when
    rebuilding ExecuTorch itself)."""
    import executorch.exir as exir

    model.eval()  # mandatory -- folds BatchNorm into conv weights
    ep = torch.export.export(model, example_input)
    edge = exir.to_edge(ep)

    if print_ops:
        # Op names must be read from the EDGE-lowered graph (after
        # exir.to_edge), not the raw torch.export graph -- the raw graph's
        # ATen "default"-variant names don't correspond 1:1 to what's
        # registered in the portable kernel library. node.target.name()
        # gives the clean base name (e.g. "aten::convolution"); the kernel
        # registry entry EXECUTORCH_SELECT_OPS_LIST needs is normally that
        # name + ".out" (e.g. "aten::convolution.out") -- but this is a
        # best-effort suggestion, not guaranteed: verify by attempting the
        # stm32 build and watching for OperatorMissing (error code 0x14 at
        # load_method), adjusting the list if any op doesn't match.
        ops = sorted({
            node.target.name() for node in edge.exported_program().graph.nodes
            if node.op == "call_function" and hasattr(node.target, "name")
        })
        suggested = [op + ".out" for op in ops]
        print("\n=== Operators used by this model ===")
        for op in ops:
            print(f"  {op}")
        print("=====================================")
        print("Best-effort EXECUTORCH_SELECT_OPS_LIST suggestion (verify against")
        print("OperatorMissing errors when building/running -- see README-testing.md):")
        print("  -DEXECUTORCH_SELECT_OPS_LIST=\"" + ",".join(suggested) + "\"\n")

    et_program = edge.to_executorch()

    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)

    size = os.path.getsize(pte_path)
    print(f"Written: {pte_path} ({size} bytes, {size / 1024:.1f} KB)")
    return ep, size
```
