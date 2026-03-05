# SOTA Object Tracking Landscape (2026)

This page summarizes recent state-of-the-art (SOTA) trends in multi-object tracking (MOT), including classic online trackers and transformer-based methods.

## Quick answer: which tracker is "best"?

There is no single best tracker across all datasets and constraints.

- **For real-time, detector-agnostic online MOT**, ByteTrack/BoT-SORT style pipelines remain very strong practical baselines.
- **For leaderboard-focused accuracy**, newer methods (including recent ByteTrack descendants such as **McByte**) can outperform earlier baselines on selected benchmarks.
- **For transformer-based MOT**, end-to-end methods (e.g., TrackFormer/MOTR/MeMOTR-style families) are competitive in ID consistency, but usually require heavier models and more task-specific training.

## Why rankings vary

Leaderboard position depends on:

1. **Benchmark** (MOT17, MOT20, DanceTrack, SportsMOT, BDD100K, etc.)
2. **Metric** (HOTA vs IDF1 vs MOTA)
3. **Runtime target** (offline vs online, FPS budget)
4. **Training setup** (private detections, ReID data, test-time tricks)

Because these differ, compare trackers inside a common protocol before selecting one for production.

## High-performing tracker families

### 1) Association-first online trackers

- SORT
- ByteTrack
- OC-SORT
- BoT-SORT
- McByte (newer ByteTrack-family method)

Typical characteristics:
- Fast, simple integration with existing detectors.
- Strong trade-off between speed and quality.
- Performance heavily depends on detection quality.

### 2) Transformer-based trackers (end-to-end MOT)

Representative families:
- **TrackFormer**
- **MOTR / MOTRv2**
- **MeMOTR** and related memory-based variants
- Query-based track-by-attention approaches in recent MOT literature

Typical characteristics:
- Better long-range temporal reasoning and ID consistency.
- Higher compute and training complexity.
- Strong when you can train/tune the whole stack jointly.

## Practical recommendation

If you need a robust default pipeline today:

1. Start with **ByteTrack** or **BoT-SORT-style** tracker.
2. Tune thresholds per dataset (activation, IoU, confidence split).
3. Add appearance features/ReID if ID switches are your bottleneck.
4. Evaluate transformer-based trackers if your priority is top accuracy over runtime simplicity.

## Notes for this repository

This repository focuses on clean, modular tracker implementations that are easy to plug into existing detection pipelines. For latest benchmark claims, always verify against current public leaderboards and paper settings.


## Recommended transformer pick

If you specifically want a transformer-family direction, **MeMOTR-style memory-query tracking** is a strong practical choice to prioritize for identity stability. In this repository, `TransformerTracker` is the lightweight online implementation aligned with that memory-token idea.
