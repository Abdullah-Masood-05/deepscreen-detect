# deepscreen-detect

[![Rust](https://img.shields.io/badge/Rust-1.97-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![ort](https://img.shields.io/badge/ort-2.0.0--rc.12-purple)](https://github.com/pykeio/ort)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D6?logo=windows&logoColor=white)](#requirements)

Camera frames in, proctoring signals out. A plain Rust library — no UI, no
window, no web view, and **no Tauri dependency**. The application that ships
this library is [deepscreen-viewer](https://github.com/Abdullah-Masood-05/deepscreen-viewer).

Written for an online-exam proctoring system that needs to answer, from a
webcam alone: is one person present, are they looking at the screen, are their
eyes moving independently of their head, and is there a phone or a book in
shot.

## Why it is a separate library

The module this replaces was JavaScript running MediaPipe in a web worker. It
could not be tested without a browser, a camera and a running React app, so no
single part of it could ever be measured in isolation — which is why it was
impossible to make fast.

This crate inverts that. It runs headless, in CI, against recorded video, and
every model can be benchmarked on its own:

```bash
detect-cli bench --all --iters 50
detect-cli live --source file:clip.mp4
```

## What it does today

| Signal | Model | Rate | Status |
|---|---|---|---|
| Face box + 5 keypoints | YuNet 2023mar | 15 Hz | working, validated |
| Head pose (yaw/pitch/roll) | MobileNetV3-Small | 15 Hz | working, validated |
| Gaze + **eye-in-head** | MobileGaze (MobileOne-S0) | 15 Hz | working |
| Prohibited objects | YOLOX-Nano | 1 Hz | working, validated |
| Identity | ArcFace `w600k_mbf` | 0.2 Hz | model wired, not yet used |

**Eye-in-head** (`gaze − head_pose`) is the signal worth having: a candidate
can hold their head square to the camera and flick their eyes down-left to a
phone on the desk. Combined gaze alone cannot distinguish that from simply
turning to face the screen.

Fusion — turning these per-frame signals into violations with hold timers and
hysteresis — is the next piece of work and is not built yet. This library
currently produces `Signals`, not decisions.

## Measured

All figures on an Intel i7-11850H, CPU execution provider, release build.

| Stage | p50 | p95 |
|---|---|---|
| Capture (1280×720 MJPEG) | 30.8 fps sustained | — |
| Face + pose + gaze worker | see note | see note |
| YOLOX-Nano (1 Hz worker) | 11.6 ms | 12.8 ms |

Per-model floors, synthetic input, 50 iterations after 5 warm-up:

| Model | p50 | p95 |
|---|---|---|
| YuNet (fp32) | 4.7 ms | 6.0 ms |
| Head pose | 1.5 ms | 2.1 ms |
| Gaze | 9.5 ms | 10.6 ms |
| YOLOX-Nano | 11.6 ms | 12.8 ms |

> **Note on the combined worker figure.** An earlier 5.9 ms p50 was recorded
> here for face + pose + gaze. It is wrong: those three models have floors
> summing to 15.8 ms, so a worker running all three cannot be faster than that.
> The gaze model's reliability gate was returning a held value with zeroed
> timings, so gated frames contributed no latency while still reporting a gaze
> value. A session with gaze demonstrably running measures ~23.5 ms p50.
>
> The gate is now explicit — a gated frame reports no gaze value and no timings
> — but how often it fires in normal use has not yet been measured, so no
> replacement number is quoted. Measuring it is the next piece of work.

Capture and detection run on separate threads with latest-frame semantics, so
detection running slower than capture drops frames rather than backing up the
camera. At 15 Hz against a 30 fps source roughly half the frames are skipped,
by design — a stale frame is worthless.

Two findings from that work, both counter-intuitive enough to be worth stating:

- **ONNX Runtime's thread pools spin-wait by default.** Adding a 1 Hz object
  worker tripled the 15 Hz face worker's latency (7.3 → 23.2 ms) because
  YOLOX's threads spun through the 988 ms they were idle. Turning spinning off
  brought it to 5.9 ms — better than before the object worker existed.
- **A downloaded INT8 model is not necessarily faster.** The official INT8
  YuNet runs **10.7× slower** than fp32 here. It is QOperator format, which
  ONNX Runtime documents as slow on x86-64. The CPU has AVX-512 VNNI, so the
  usual explanation does not apply.

## Architecture

```
capture thread ──► ArcSwap<Frame>   latest-frame slot, overwrite, never a queue
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   face worker @ 15 Hz            object worker @ 1 Hz
   YuNet → pose → gaze            YOLOX-Nano
          │                               │
          └───────────────┬───────────────┘
                          ▼
                    Signals  ──►  fusion (not built yet) ──► Violations
```

Rules that hold throughout:

- One ONNX Runtime `Session` per model, owned by exactly one thread. No
  `Mutex<Session>` anywhere in the inference path.
- Every tunable number lives in one `Config` struct and nowhere else.
- `Signals` is `Serialize + Deserialize`, so a session can be recorded once and
  replayed through fusion thousands of times with zero inference. Threshold
  tuning that requires re-running models does not get done.
- The object worker never depends on face presence. A phone held over the face
  is the case that matters most, and gating objects on a detected face throws
  exactly that away.

## Requirements

- Rust 1.97+
- **ffmpeg and ffprobe on `PATH`** for `camera:` and `file:` sources. Frames
  are decoded in a subprocess and piped in as raw RGB24 rather than linking a C
  decoder. `dir:` sources need nothing, which is what CI uses.
- Windows x64. The camera path uses DirectShow; other platforms can use
  `file:` and `dir:` sources.

## Models

Weights are **not committed**. Download them into `models/`:

```bash
mkdir -p models && cd models

# Face detection — YuNet, MIT
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Head pose — MIT
curl -LO https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx
mv mobilenetv3_small.onnx headpose_mobilenetv3_small.onnx

# Gaze — MobileGaze, MIT
curl -LO https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx

# Objects — YOLOX-Nano, Apache 2.0
curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
```

| Slot | Source | Licence |
|---|---|---|
| Face | [opencv/opencv_zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| Head pose | [yakhyo/head-pose-estimation](https://github.com/yakhyo/head-pose-estimation) | MIT |
| Gaze | [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) | MIT |
| Objects | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache 2.0 |
| Identity | [deepinsight/insightface — buffalo_sc](https://github.com/deepinsight/insightface) | see upstream |

YOLOX-Nano was chosen over the faster-sounding alternatives because it is
Apache 2.0. Ultralytics' YOLO26n is AGPL-3.0, which would require open-sourcing
anything that ships it or buying a commercial licence — and it measured 32.2 ms
against YOLOX-Nano's 12.0 ms on this hardware, so the permissive option was
also the quicker one.

## Using the CLI

```bash
cargo build --release        # always measure release; debug is ~15× slower

detect-cli devices --formats                        # cameras and their modes
detect-cli inspect models/*.onnx                    # real tensor interfaces
detect-cli bench --all --iters 50 --report bench.md # per-model p50/p95
detect-cli config --out dev.toml                    # every tunable number

detect-cli live   --source camera:0                 # live, with stats
detect-cli live   --source file:clip.mp4 --paced
detect-cli record --source file:clip.mp4 --out signals/clip.jsonl
detect-cli replay signals/clip.jsonl
```

`inspect` earns its keep: it reported that YuNet's released ONNX takes 640×640
and not the 320×320 its documentation implies, that the head-pose model returns
a 3×3 rotation matrix rather than Euler angles, and that MobileGaze emits two
90-bin classification heads rather than regressed angles. Guessing any of those
produces plausible-looking output that is quietly wrong.

`record` deliberately bypasses the threaded pipeline and processes **every**
frame, so the same clip always produces the same JSONL. A recording whose
contents depend on how fast the recording machine was is not a regression
fixture.

## Library use

```rust
use deepscreen_detect::{Config, Detector, SourceSpec};

let mut config = Config::default();
config.models.fill_missing_from_dir("models");

let source = "camera:0".parse::<SourceSpec>()?.open(&config.capture, false)?;

let mut detector = Detector::builder().config(config).build()?;
detector.start(source)?;

// Edge-triggered decisions, low rate.
let events = detector.events();

// Level-triggered continuous values, polled at whatever rate the UI wants.
let state = detector.snapshot();
```

There is deliberately no way to *push* continuous values at frame rate. The old
module re-rendered its UI on every frame because detection pushed state; here
that bug cannot be expressed.

## Tests

```bash
cargo test --release
```

Model-dependent tests skip cleanly when `models/` is empty, so a fresh clone
passes. Where they can, they check correctness rather than just latency:

- Head-pose sign is verified by a physical invariant — mirroring an image must
  negate yaw and roll and leave pitch alone.
- The YOLOX decode is verified against YOLOX's own canonical test image, whose
  expected output (dog, bicycle, truck) is documented by every YOLO demo.
- A clean clip containing no phone and no book must produce zero detections.
  False positives are what make a proctoring system unusable.

## Licence

MIT. Model weights carry their own licences — see the table above.
