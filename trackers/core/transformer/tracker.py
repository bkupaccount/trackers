# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.sort.kalman import SORTKalmanBoxTracker


@dataclass
class _TrackToken:
    bbox: np.ndarray
    embedding: np.ndarray
    hits: int
    age: int
    time_since_update: int
    tracker_id: int


class TransformerTracker(BaseTracker):
    """Transformer-inspired online multi-object tracker.

    This tracker keeps a compact memory token per track and associates incoming
    detections by fusing spatial overlap with embedding similarity. The design
    is inspired by query-based transformer MOT families (e.g., TrackFormer,
    MOTR, MeMOTR), while staying detector-agnostic and lightweight.

    Args:
        lost_track_buffer: Number of frames to keep unmatched tracks alive.
        frame_rate: Input stream frame rate used to scale lost track buffer.
        track_activation_threshold: Minimum confidence to initialize new tracks.
        minimum_consecutive_frames: Hits required before exposing stable IDs.
        minimum_similarity_threshold: Minimum fused similarity for matching.
        appearance_weight: Weight of appearance similarity in fused score.
        iou_weight: Weight of IoU similarity in fused score.
        embedding_momentum: EMA momentum when updating track embeddings.
    """

    tracker_id = "transformer"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.35,
        minimum_consecutive_frames: int = 2,
        minimum_similarity_threshold: float = 0.3,
        appearance_weight: float = 0.65,
        iou_weight: float = 0.35,
        embedding_momentum: float = 0.9,
    ) -> None:
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.track_activation_threshold = track_activation_threshold
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_similarity_threshold = minimum_similarity_threshold
        self.appearance_weight = appearance_weight
        self.iou_weight = iou_weight
        self.embedding_momentum = embedding_momentum
        self.tracks: list[_TrackToken] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker memory tokens and return detections with track IDs."""
        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        detection_boxes = detections.xyxy
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )
        embeddings = self._extract_embeddings(detections)

        tracker_ids = np.full(len(detections), -1, dtype=int)
        valid_det_indices = np.where(confidences >= self.track_activation_threshold)[0]

        if len(self.tracks) > 0 and len(valid_det_indices) > 0:
            sim_matrix = self._build_similarity_matrix(
                detection_boxes[valid_det_indices], embeddings[valid_det_indices]
            )
            row_indices, col_indices = linear_sum_assignment(sim_matrix, maximize=True)

            used_dets: set[int] = set()
            for row, col in zip(row_indices, col_indices):
                if sim_matrix[row, col] < self.minimum_similarity_threshold:
                    continue
                det_index = int(valid_det_indices[int(col)])
                track = self.tracks[int(row)]
                track.bbox = detection_boxes[det_index].copy()
                track.embedding = self._ema_embedding(
                    track.embedding, embeddings[det_index], self.embedding_momentum
                )
                track.hits += 1
                track.time_since_update = 0
                if (
                    track.hits >= self.minimum_consecutive_frames
                    and track.tracker_id == -1
                ):
                    track.tracker_id = SORTKalmanBoxTracker.get_next_tracker_id()
                tracker_ids[det_index] = track.tracker_id
                used_dets.add(det_index)

            for det_index in valid_det_indices:
                if int(det_index) in used_dets:
                    continue
                self.tracks.append(
                    _TrackToken(
                        bbox=detection_boxes[det_index].copy(),
                        embedding=embeddings[det_index].copy(),
                        hits=1,
                        age=1,
                        time_since_update=0,
                        tracker_id=-1,
                    )
                )
        else:
            for det_index in valid_det_indices:
                self.tracks.append(
                    _TrackToken(
                        bbox=detection_boxes[det_index].copy(),
                        embedding=embeddings[det_index].copy(),
                        hits=1,
                        age=1,
                        time_since_update=0,
                        tracker_id=-1,
                    )
                )

        self.tracks = [
            t
            for t in self.tracks
            if t.time_since_update < self.maximum_frames_without_update
        ]

        result = detections
        result.tracker_id = tracker_ids
        return result

    def reset(self) -> None:
        """Reset tracker state for a new video sequence."""
        self.tracks = []

    def _build_similarity_matrix(
        self, detection_boxes: np.ndarray, detection_embeddings: np.ndarray
    ) -> np.ndarray:
        track_boxes = np.array([track.bbox for track in self.tracks], dtype=np.float32)
        iou = sv.box_iou_batch(track_boxes, detection_boxes)

        track_embeddings = np.array(
            [track.embedding for track in self.tracks], dtype=np.float32
        )
        appearance = self._cosine_similarity(track_embeddings, detection_embeddings)
        appearance = (appearance + 1.0) / 2.0

        return self.iou_weight * iou + self.appearance_weight * appearance

    @staticmethod
    def _ema_embedding(
        previous: np.ndarray, current: np.ndarray, momentum: float
    ) -> np.ndarray:
        norm = np.linalg.norm(current)
        current = current / max(norm, 1e-6)
        blended = momentum * previous + (1.0 - momentum) * current
        return blended / max(np.linalg.norm(blended), 1e-6)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
        b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
        return a_norm @ b_norm.T

    @staticmethod
    def _extract_embeddings(detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return np.empty((0, 4), dtype=np.float32)

        if "embedding" in detections.data:
            embeddings = np.asarray(detections.data["embedding"], dtype=np.float32)
            if embeddings.ndim == 2 and embeddings.shape[0] == len(detections):
                return embeddings

        boxes = detections.xyxy.astype(np.float32)
        centers = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)
        )
        size = np.column_stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]))
        fallback = np.concatenate((centers, size), axis=1)
        norms = np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-6)
        return fallback / norms
