//! `deepscreen-detect` — camera frames in, proctoring violations out.
//!
//! This crate knows nothing about Tauri, React, the assessment flow, or the
//! backend, and it never will (MODELS.md §0). That single constraint is what
//! makes it possible to build and benchmark the whole detection module before
//! touching the app, test it headless in CI, and tune it against recorded
//! video rather than a live exam.
//!
//! # Build status
//!
//! Following the build order in MODELS.md §11. Each step is independently
//! measurable; fusing steps loses track of what helped.
//!
//! | Step | What | State |
//! |---|---|---|
//! | 1 | Crate skeleton, types, config, `detect-cli`, file replay | **done** |
//! | 2 | Camera capture behind `FrameSource` | not started |
//! | 3 | YuNet face detection + baseline bench | **done** (8.4 ms p50 CPU) |
//! | 4 | Threading skeleton, `ArcSwap` frame bus, `Detector` | **done** |
//! | 5 | DirectML | not started |
//! | 6 | Pose + gaze | not started |
//! | 7 | YOLO26n on its own worker | not started |
//! | 8 | Fusion, record/replay tuning | not started |
//! | 9 | ArcFace identity | not started |
//! | 10 | Quantization | not started |
//! | 11 | Tauri adapter | not started |
//!
//! Face detection runs through `detect-cli live` today. There is no `Detector` yet — it arrives with the threading skeleton at
//! step 4. Until then the CLI drives a `FrameSource` directly, which is
//! exactly what step 1 is for: proving the types, the config and the harness
//! before any model or thread exists to blame.

pub mod capture;
pub mod config;
pub mod error;
pub mod models;
pub mod pipeline;
pub mod report;
pub mod types;

pub use capture::{FrameSource, SourceSpec};
pub use config::Config;
pub use error::{DetectError, Result};
pub use pipeline::{Detected, Detector, DetectorBuilder};
pub use report::{FrameStats, Latencies, LatencySummary, SessionReport, SignalStatus};
pub use types::{
    BBox, Contribution, DegradeReason, DetectorState, Event, EyeAspect, FaceDetection,
    FaceKeypoints, Frame, Gaze, HeadPose, ObjectDetection, Severity, SignalCoverage, SignalSource,
    Signals, Violation, ViolationKind,
};

/// Version of the `Signals` JSONL format. Bump when a change would make an
/// old recording replay to different violations — recordings are the
/// regression corpus, and silently reinterpreting them would be worse than
/// refusing to read them.
pub const SIGNALS_FORMAT_VERSION: u32 = 1;
