# Building, running, and testing — lab1_wakeword

Environment setup, the build targets (`host`, `stm32`, `stm32-renode`),
running the firmware in each of those environments, and debugging it —
everything you need once you're ready to compile and exercise your code.
See `README.md` for what to build (the lab's tasks) and
`README-submission-instructions.md` for what to package once it works.

**How `tools/` fits in**: this repo ships two provided tools, both used
below — `tools/compare_spectrogram.py` (plus the `tools/spectrogram_dump.c`
harness it builds) numerically cross-checks your
`models/spectrogram_features.py` against the compiled C
`src/spectrogram.c`, so you catch a feature-extraction mismatch before
spending time training against it (see "Build / test on host" below). This
matters more than it might sound like: `spectrogram_features.py` is meant
to be produced with an AI coding assistant rather than hand-derived
(`README-training-parameters.md` §2), and this tool is the check that a
generated implementation actually matches the firmware bit-for-bit, not
just "looks like an STFT." There's no provided tool for generating
`ops.json`/`memory.json` — those are self-derived from your own build
artifacts; see `README-submission-instructions.md` for exactly how.

## Prerequisites

```bash
sudo apt install -y gcc-arm-none-eabi binutils-arm-none-eabi gdb-multiarch openocd ninja-build cmake xxd
```

Plus a working ExecuTorch checkout + build (host and cortex-m4 targets) and
Python venv, and "Building/rebuilding ExecuTorch" below for this project's
specific op-list requirement.

## Build / test on host (no hardware needed)

Validates the full pipeline (spectrogram + ExecuTorch) against WAV files
once you've trained a model and filled in `src/main.cpp`'s ExecuTorch
bring-up (note: `inference_host` links `src/host_test_main.cpp`, which is
provided and separate from `main.cpp`'s `STUDENT SECTION` — this build
mainly validates your model/spectrogram, not your inference-loop code):

```bash
source ~/executorch/et-env/bin/activate
cmake --preset host
cmake --build --preset host

./build/host/inference_host testdata/sample_yes.wav submission/model.pte
# [host] Loaded 16000 of 16000 requested samples (1.000 s) from testdata/sample_yes.wav
# PREDICTION: yes conf=0.9746
```

`testdata/` has a few real Speech Commands clips (`sample_stop.wav`,
`sample_yes.wav`, `sample_up.wav`) and one background-noise clip
(`sample_silence.wav`) for a quick sanity check — `sample_stop.wav` and
`sample_up.wav` are not target classes, so expect `unknown` for those. Try
your own WAV (mono, 16-bit, 16kHz; shorter clips are zero-padded, longer
ones truncated to the first second).

**Cross-checking the spectrogram** (do this before trusting any model
training against `models/spectrogram_features.py`):

```bash
python3 tools/compare_spectrogram.py                 # synthetic test tone
python3 tools/compare_spectrogram.py your_test.wav    # or a real clip
# max abs diff:  5.0e-05
# PASS: C and Python spectrogram pipelines agree within 1.0e-03
```

## Build / flash on hardware

```bash
source ~/executorch/et-env/bin/activate
cmake --preset stm32
cmake --build --preset stm32
cmake --build --preset stm32 --target flash   # or: openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "transport select hla_swd" -c "program build/stm32/firmware.bin 0x08000000 verify reset exit"
```

Connect a USB-serial adapter to USART2 (PA2=TX, PA3=RX) at 115200 8N1 and
watch for `PREDICTION: <label> conf=<float>` lines as you speak "yes" or
"no" near the board (anything else should predict `unknown`).

To point the build at a different `.pte` (e.g. while iterating before
copying your final export into `submission/`): `cmake --preset stm32
-DWAKEWORD_PTE=/path/to/your.pte` and reconfigure.

## Building/rebuilding ExecuTorch

ExecuTorch's op-kernel selection (`EXECUTORCH_SELECT_OPS_LIST`) is a
property of the prebuilt `libexecutorch_selected_kernels.a`, not something
this project's `CMakeLists.txt` controls. If you train a model whose op list
differs from the default build's, your export step (Appendix A's
`export_to_executorch`) prints the op list your model needs (best-effort,
`.out`-suffixed) — rebuild both ExecuTorch targets with it:

```bash
source ~/executorch/et-env/bin/activate

OPS="aten::_native_batch_norm_legit_no_training.out,aten::batch_norm.out,\
aten::convolution.out,aten::relu.out,aten::mean.out,aten::adaptive_avg_pool2d.out,\
aten::addmm.out,aten::view_copy.out,aten::permute_copy.out,\
aten::max_pool2d_with_indices.out,dim_order_ops::_clone_dim_order.out,\
dim_order_ops::_to_dim_order_copy.out"
# (a generous starting superset covering both models as described in
#  README-training-parameters.md -- trim or extend based on your own
#  model's printed op list)

cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/cortex-m4 \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/executorch/executorch/cmake/toolchains/arm-none-eabi.cmake \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DEXECUTORCH_BUILD_KERNELS_PORTABLE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_SELECT_OPS_LIST="$OPS"
cmake --build ~/executorch/executorch-build/cortex-m4 -j$(nproc)

cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/host \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_BUILD_KERNELS_PORTABLE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_SELECT_OPS_LIST="$OPS"
cmake --build ~/executorch/executorch-build/host -j$(nproc)
```

If `load_method` fails at runtime with error `0x14` (`OperatorMissing`), an
op is missing from this list — check the exact op name in the error/log and
add it.

## Runtime-tunable parameters

All defined as named constants, documented inline where declared:

| Constant | Value | File | Meaning |
|---|---:|---|---|
| `SPEC_SAMPLE_RATE` | 16000 Hz | `src/spectrogram.h` | mic sample rate |
| `SPEC_NUM_SAMPLES` | 16000 | `src/spectrogram.h` | inference window length (1.0s) |
| `SPEC_FRAME_LENGTH` | 480 | `src/spectrogram.h` | STFT frame length (30ms) |
| `SPEC_FRAME_STEP` | 160 | `src/spectrogram.h` | STFT hop (10ms) |
| `SPEC_OUT_FRAMES` × `SPEC_OUT_FREQ` | 49 × 128 | `src/spectrogram.h` | resized feature map shape |
| `WINDOW_SAMPLES` | 16000 | `src/main.cpp` | inference window length |
| `WAKEWORD_CLASS_NAMES[4]` | see below | `src/wakeword_labels.h` | class index → name, must match training |
| `method_pool` / `planned_pool` | **TODO(student)**, currently `0`/`0` | `src/main.cpp` | ExecuTorch memory pools, see "Memory budget" below |
| UART baud | 115200 8N1 | `hal.c` (shared repo) | prediction output |
| Prediction format | `PREDICTION: %s conf=%f\r\n` | `src/main.cpp` | fixed-prefix, Robot-Framework-parseable |

Class order (`src/wakeword_labels.h`, index = model output logit index):
`yes, no, silence, unknown`.

Spectrogram-computation constants (`SPEC_*`) are also the normative spec for
your training-side reimplementation — see `README-training-parameters.md`.

## Memory budget

The STM32F407 has three memory regions relevant here:

| Region | Capacity | Notes |
|---|---:|---|
| Flash (0x08000000) | 1024 KB | firmware code + embedded model |
| Main SRAM (0x20000000, DMA-reachable) | 128 KB | `dma_buf` (shared mic driver) + `window_scratch` + your `method_pool`/`planned_pool` + stack |
| CCM RAM (0x10000000, CPU-only) | 65,536 B | `pcm_ring` (shared mic driver) + `spectrogram.c` FFT/Hann/row scratch + `spectrogram_out` (model input tensor storage) |

`method_pool`/`planned_pool` are currently sized `0`/`0` in `src/main.cpp` —
sizing them is your task, not a fixed value to copy. **How to find a
starting point instead of guessing blind:**

```python
import executorch.exir as exir, torch
# ... build/load your model, export as usual through to_edge() ...
et_program = edge_program.to_executorch()
print(et_program.executorch_program.execution_plan[0].non_const_buffer_sizes)
# [0, <bytes needed>] -- planned_pool must be >= that second number
```

`method_pool`'s minimum isn't recorded in the `.pte` the same way — it's a
runtime bump allocator with no static size hint, so it has to be found by
trial: start generously (e.g. 64KB), get `load_method()` succeeding, then
shrink and re-test to find roughly where `MemoryAllocationFailed` (0x21)
starts, and leave some headroom above that.

**CCM RAM is tight** — `pcm_ring` and the spectrogram scratch buffers alone
use a large fraction of its 64KB, so `spectrogram_out` (the model input
tensor storage) competing for the remainder is a real constraint, not
theoretical. If your model's input size or the spectrogram scratch changes,
re-check that everything still fits. If you change the model architecture,
**re-measure `planned_pool`** rather than assuming any number you've seen
elsewhere still fits — a bigger model's activation buffer can easily need
much more RAM (see the cautionary story in `README-training-parameters.md`
§4.2: a milder stem stride needed ~290KB, more than all of main SRAM).

## Testing with Renode

Renode simulates the STM32F407 without real hardware, using a WAV-derived
synthetic audio source in place of the real PDM mic (Renode has no PDM
peripheral model) — see `../stm32f4_audio_examples/audio_in_to_wav`'s
`synthetic_mic.c` for the substitution mechanism this reuses.

```bash
cmake --preset stm32-renode
cmake --build --preset stm32-renode

renode renode/wakeword_demo.resc
```

In another terminal, connect to the bridged UART (predictions stream over
TCP port 3456):

```bash
python3 -c "
import socket
s = socket.create_connection(('127.0.0.1', 3456))
while True:
    print(s.recv(256).decode(errors='replace'), end='')
"
# or: nc localhost 3456   /  picocom  /  telnet localhost 3456
```

**Connect once and hold the connection open** — Renode's simple socket
terminal does not replay history to a client that connects late, and an
abrupt disconnect (unread buffered data still in flight) can leave the
server unable to accept new connections, requiring a fresh `renode
renode/wakeword_demo.resc` restart.

The default injected clip (`renode/default_synthetic_audio.bin`, from
`testdata/sample_yes_stop_2s.wav`) is 2 seconds: a real "yes" clip followed
by a real "stop" clip, each zero-padded/truncated to exactly 1 second — long
enough to observe two distinct `PREDICTION:` lines per loop of the
(looping) injected clip. Since `stop` is not a target class, expect
`PREDICTION: yes ...` followed by `PREDICTION: unknown ...` (rather than a
`stop` prediction) — still two distinct lines, which is the point of this
demo clip. Expect the *first* prediction or two after boot to be noisy (the
ring buffer starts zero-filled, and the very first window can catch a
partially-filled ring) — steady-state predictions on repeated loops should
be far more reliable. This is a property of streaming/live audio
classification in general, not specific to Renode.

**Testing your own WAV**:

```bash
python3 ../stm32f4_audio_examples/audio_in_to_wav/tools/wav_to_pcm_bin.py \
  your.wav --output /tmp/your.bin --max-samples 96000   # compiled buffer capacity

renode -e "\$bin=@build/stm32-renode/firmware.elf; \$pcm_file=@/tmp/your.bin; \$pcm_samples=<N>" \
  renode/wakeword_demo.resc
```

(`<N>` = number of samples in your converted clip.)

**Growing the synthetic-mic buffer**: the compiled-in array defaults to
96,000 samples (6s) — too short for longer clips (including what
`README-submission-instructions.md` asks you to regenerate before your
final Renode build). Regenerate it at any length:

```bash
python3 ../stm32f4_audio_examples/audio_in_to_wav/tools/wav_to_c_array.py \
  --output src/generated/synthetic_audio_data.h --seconds <N>
```

This only affects `synthetic_mic.c`'s compiled-in array capacity — it does
not change anything about the real-mic build or your actual deployed
firmware's behavior. Note that `synthetic_audio_pcm` is declared `const`,
so its *capacity* costs real flash regardless of how much of it Renode
later overwrites at runtime — don't set `--seconds` far larger than you
actually need.

**Reading debug values via GDB under Renode** (for `submission/debug_values.json`):

```bash
gdb-multiarch build/stm32-renode/firmware.elf
(gdb) target remote :3333
(gdb) monitor reset halt
(gdb) load
(gdb) break <the line in your STUDENT SECTION right after it prints DETECTED>
(gdb) continue
# once hit:
(gdb) print <your written/onset variables>
```

## GDB / fault debugging

```bash
gdb-multiarch build/stm32/firmware.elf
(gdb) target remote :3333
(gdb) monitor reset halt
(gdb) load
(gdb) continue
```

If stopped in `Default_Handler`: `info registers r1 r2 r3` (r1=PC at fault,
r2=LR, r3=CFSR). Common CFSR values: `0x00080000` = NOCP (FPU not enabled —
shouldn't happen, `startup_stm32f407.s` enables it first thing),
`0x00020000` = UNALIGNED, `0x00000001` = IACCVIOL.
`arm-none-eabi-addr2line -e build/stm32/firmware.elf 0x<PC>` maps a fault PC
back to source.

## Manually replicating the live tests (B1, T1-T4) without the full grader

`grade.py`'s live/behavioral checks (B1, T1, T2, T3, T4 — the ones that
actually drive predictions through your firmware, as opposed to the
format/plausibility checks on your JSON files, which you self-check just
by re-deriving them the same way you produced them) all use the same
injection mechanism as `renode/wakeword_demo.resc`: load a fixed
WAV-derived PCM clip into `synthetic_audio_pcm`, override its sample
count, boot, and capture `PREDICTION:` lines over the bridged UART. The
grader's exact fixture clips (built from Speech Commands + real
recordings) aren't shipped in this repo, but you can exercise the same
pass/fail logic with your own WAV clips using "Testing your own WAV" above
— useful for iterating on your VAD/model without waiting ~3.5-4 minutes for
a full `grade.py` run.

**B1 — boots fault-free.** The grader loads `firmware_stm32_renode.elf`,
runs it for 8 simulated seconds, then reads the CFSR register
(`0xE000ED28`) and requires it to be exactly `0`. Self-check: run your
Renode build for a while (the default demo clip is fine), then — without
waiting for a fault — read CFSR directly:

```bash
gdb-multiarch build/stm32-renode/firmware.elf
(gdb) target remote :3333
(gdb) print/x *(unsigned int*)0xE000ED28
```

A nonzero value means the same fault conditions as "GDB / fault debugging"
above.

**T1 / T3 — single-word clips.** For each fixed clip (T1: 6 Speech
Commands "yes"/"no" clips, training-distribution audio; T3: 10
real-recorded "yes"/"no" clips from an actual speaker/mic), the grader
injects it, collects the first **3** `PREDICTION:` lines, and passes if
the expected label appears **anywhere** among those 3 — not necessarily
first. T1 is worth 10 points/clip (60 total, all-or-nothing per clip); T3
is worth 5 points total across all 10 clips, tolerant scoring (1 wrong is
free, then -0.5/additional wrong). Self-check: convert your own "yes"/"no"
WAV with `wav_to_pcm_bin.py` (see "Testing your own WAV" above), run it
through `wakeword_demo.resc`, and watch the first 3 `PREDICTION:` lines on
the bridged UART — the expected label should appear at least once.

**T2 — 4-word sequence.** The grader injects a fixed "yes,no,yes,no"
sequence clip, collects the first **8** predictions, and passes only if
**both** "yes" and "no" appear somewhere among them (order doesn't matter,
and not every individual word needs to be recovered) — worth 25 points,
all-or-nothing. Self-check: concatenate a few of your own "yes"/"no" WAVs
into one clip, convert and inject it the same way, and confirm both labels
show up somewhere across the first 8 predictions.

**T4 — two 5-word sequences.** The grader injects two ~20s, 5-word
real-recorded sequences (e.g. `yes,no,yes,no,yes`), collects the first
**8** predictions per sequence, and scores by **multiset overlap**: how
many of the 5 expected labels are matched by a same-labeled prediction (a
`Counter` intersection, so extra repeated predictions don't inflate the
score, and position doesn't matter — a settled prediction cycle can be
phase-shifted relative to the actual word order even when every word was
correctly identified). "Wrong" = `5 - matched`, summed across both
sequences (max 10 total); worth 5 points, tolerant scoring (1 wrong out of
10 is free, then -0.5/additional). Self-check: record or assemble your own
5-word "yes"/"no" sequence, inject it, collect 8 predictions, and count by
hand how many of your 5 expected labels are covered — a prediction can
only "use up" one occurrence of its label.

**Everything else** (F1, M1, TIME1, DBG1, ACC1, SZ1, SHAPE): format or
plausibility checks on files you generate yourself — see
`README-submission-instructions.md`'s per-file sections. Self-grading them
is just re-running the same derivation you used to produce the file and
comparing against the stated format/threshold; there's no separate
"replicate the grader" step needed.

## Known caveats

- **CCM RAM is tight** (see "Memory budget" above) — any change to
  spectrogram scratch or model input size needs a corresponding check that
  things still fit.
- **`method_pool`/`planned_pool` must be re-measured per model** — there's
  no universal safe default; see "Memory budget" above.
- **Read the mic window before processing it, never in place** — copy each
  window out of the ring buffer (into a scratch array) before computing its
  spectrogram. ExecuTorch's portable (unoptimized) kernels can take long
  enough on this kind of model that reading `ring[]` directly while
  inference runs risks the DMA overwriting it mid-read — a real bug that's
  easy to reintroduce if you "optimize away" the copy. Symptom to watch for:
  implausibly saturated ~1.0 confidence on every prediction.
