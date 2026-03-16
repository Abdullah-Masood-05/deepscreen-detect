# models/

All five slots are present. Every number below is **measured on this machine**,
not estimated — `detect-cli inspect` for the interfaces, `detect-cli bench --all`
for the latencies. The `[verify]` markers from MODELS.md §5.0 are resolved.

Committed through git-lfs (see `../.gitattributes`). Each is loaded exactly
once, on a background thread at app start, and warmed up with 3–5 dummy
inferences before the first real frame.

## What is here

| Slot | File | Real input | Real output | MB | Licence | Cadence |
|---|---|---|---|---|---|---|
| Face + 5 keypoints | `face_detection_yunet_2023mar.onnx` | `1x3x640x640` | 12 tensors, strides 8/16/32 | 0.2 | MIT | 15 Hz |
| Face (INT8) | `face_detection_yunet_2023mar_int8.onnx` | `1x3x640x640` | same | 0.1 | MIT | — see below |
| Head pose | `headpose_mobilenetv3_small.onnx` | `1x3x224x224` | `rotation_matrix [1,3,3]` | 5.8 | MIT | 15 Hz |
| Gaze | `mobileone_s0_gaze.onnx` | `1x3x448x448` | `yaw [1,90]`, `pitch [1,90]` | 4.7 | MIT | 15 Hz |
| Objects | `yolo26n.onnx` | `1x3x640x640` | `output0 [1,300,6]` | 9.5 | **AGPL-3.0** | 1–2 Hz |
| Identity | `w600k_mbf.onnx` | `[?,3,112,112]` | `516 [1,512]` | 13.0 | — | 0.2 Hz |

~33 MB total, close to the spec's ~32 MB estimate.

## Measured latency — i7-11850H, CPU EP, 50 iters after 5 warm-up

Synthetic zero-tensor forward passes. No preprocessing, no decode: a floor, not
a budget. The numbers `detect-cli live` reports include preprocessing and are
the ones to design against.

| model | p50 ms | p95 ms |
|---|---|---|
| yunet fp32 | 4.49 | 5.58 |
| **yunet int8** | **47.96** | **51.03** |
| headpose mobilenetv3-small | 1.60 | 2.11 |
| mobilegaze mobileone-s0 | 9.46 | 11.23 |
| arcface w600k_mbf | 5.45 | 7.35 |
| yolo26n | 32.24 | 35.49 |

**The face worker fits.** YuNet + pose + gaze is 15.6 ms against a 66.7 ms
budget at 15 Hz — about 23%. YOLO26n at 32 ms once per second and ArcFace at
5.5 ms every five seconds are rounding errors on top. Roughly 27% of one core
for the whole stack before any GPU is involved.

## Three corrections to MODELS.md §5.0

1. **YuNet's input is 640×640, not 320×320**, and the head emits twelve
   tensors with no NMS. Anchor decode and suppression are in
   `src/models/face.rs`.
2. **The head-pose model outputs a 3×3 rotation matrix, not Euler angles**, and
   takes 224×224 rather than 60×60. This is `yakhyo/head-pose-estimation`
   (MIT, same author as MobileGaze, so the same export toolchain) rather than
   `head-pose-estimation-adas-0001`, which is only published as OpenVINO IR.
   Postprocessing has to convert the matrix to yaw/pitch/roll.
3. **MobileGaze is 4.97 MB, not ~8 MB**, takes 448×448, and emits two 90-bin
   *classification* heads rather than regressed angles. Decode is softmax then
   expectation over bin centres — the L2CS-Net scheme it inherits.

`yolo26n.onnx` came straight from `ultralytics/assets` release `v8.4.0`, so no
Python, torch or export step was needed. Its `(1, 300, 6)` NMS-free output is
exactly as §5.3 promised.

## The INT8 result, which contradicts the spec

MODELS.md §5.1 calls YuNet's official INT8 "the happy exception — use it
without hesitation", on the strength of OpenCV Zoo's accuracy eval. That eval
is about *accuracy*, and it holds. **Speed on this CPU is another matter: INT8
is 10.7× slower than fp32.**

The cause is not missing VNNI — an i7-11850H is Tiger Lake and has AVX-512
VNNI. Counting op types in the file gives the answer:

```
QLinearConv:      53      <- QOperator format
QuantizeLinear:   10      <- QDQ would be hundreds, paired with Dequantize
DequantizeLinear: 32
```

That is **QOperator**, which is trap #3 in §5.1's own table: *"S8S8 with
QOperator will be slow on x86-64 CPUs and should be avoided in general."* So
the spec predicted this failure mode and then exempted the one model that
exhibits it.

Practical consequences:

- **Ship fp32 YuNet.** The INT8 file stays only as evidence for the write-up.
- The "ship both and pick at startup via a micro-benchmark" policy in §5.1 is
  vindicated — here it would have silently saved 43 ms per frame.
- If INT8 is wanted later, quantize YuNet yourself with
  `quant_format=QuantFormat.QDQ` and `activation_type=QuantType.QUInt8`,
  calibrated on real webcam frames, then re-benchmark. Do not assume a
  downloaded INT8 model is fast anywhere.

## Still to decide before step 7

**YOLO26 is AGPL-3.0.** Having it on disk for an FYP is fine; shipping
DeepScreen closed-source is not, without open-sourcing derivatives or buying an
Ultralytics enterprise licence. Decide *before* building post-processing around
`(1, 300, 6)`, because swapping to RF-DETR Nano, YOLOX-Nano or
EfficientDet-Lite0 later means writing the NMS that shape lets you skip.

## Also on disk elsewhere

`../../DeepScreen-DesktopApp/src-tauri/resources/face-recognition/` holds the
original `w600k_mbf.onnx` (copied here) and `det_500m.onnx` (2.5 MB,
SCRFD-500M, unused). Now that YuNet works there is no reason to reach for
SCRFD — it is 10× larger for the same job.
