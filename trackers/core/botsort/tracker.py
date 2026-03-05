# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.bytetrack.kalman import ByteTrackKalmanBoxTracker
from trackers.core.sort.kalman import SORTKalmanBoxTracker
from trackers.core.sort.utils import get_alive_trackers, get_iou_matrix


class BOTSORTTracker(BaseTracker):
    """Lightweight BoT-SORT style online multi-object tracker.

    This tracker follows the high-level BoT-SORT strategy: a ByteTrack-like
    two-stage association with confidence-aware first-pass matching.

    Note:
        This is a compact implementation focused on motion + confidence cues.
        Full BoT-SORT variants in literature often add appearance ReID and
        camera-motion compensation modules for stronger identity consistency.

    Args:
        lost_track_buffer: Number of frames to keep unmatched tracks alive.
        frame_rate: Input stream frame rate used to scale lost track buffer.
        track_activation_threshold: Confidence threshold to initialize tracks.
        minimum_consecutive_frames: Number of updates required before track ID
            is exposed (otherwise `-1` is returned).
        minimum_iou_threshold: Minimum fused similarity for a valid match.
        high_conf_det_threshold: Threshold to split high/low confidence
            detections for two-stage association.
        low_conf_det_threshold: Detections below this threshold are ignored.
        association_lambda: Balance between IoU and confidence in first-stage
            matching. 1.0 uses pure IoU, lower values use more confidence.
    """

    tracker_id = "botsort"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.5,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
        low_conf_det_threshold: float = 0.1,
        association_lambda: float = 0.85,
    ) -> None:
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.track_activation_threshold = track_activation_threshold
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.low_conf_det_threshold = low_conf_det_threshold
        self.association_lambda = float(np.clip(association_lambda, 0.0, 1.0))
        self.tracks: list[ByteTrackKalmanBoxTracker] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state and return detections with `tracker_id` values."""
        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        for tracker in self.tracks:
            tracker.predict()

        detection_boxes = detections.xyxy
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.zeros(len(detections), dtype=np.float32)
        )

        valid_mask = confidences >= self.low_conf_det_threshold
        valid_indices = np.where(valid_mask)[0]

        high_mask = confidences[valid_indices] >= self.high_conf_det_threshold
        high_indices = valid_indices[high_mask]
        low_indices = valid_indices[~high_mask]

        matched_pairs, unmatched_track_ids, unmatched_high_ids = self._associate(
            detections=high_indices,
            boxes=detection_boxes,
            confidences=confidences,
            use_confidence=True,
        )

        low_matched_pairs, unmatched_track_ids, _ = self._associate(
            detections=low_indices,
            boxes=detection_boxes,
            confidences=confidences,
            track_subset=unmatched_track_ids,
            use_confidence=False,
        )

        for track_id, det_id in [*matched_pairs, *low_matched_pairs]:
            self.tracks[track_id].update(detection_boxes[det_id])

        for unmatched_det_id in unmatched_high_ids:
            if confidences[unmatched_det_id] >= self.track_activation_threshold:
                self.tracks.append(
                    ByteTrackKalmanBoxTracker(detection_boxes[unmatched_det_id])
                )

        self.tracks = get_alive_trackers(
            self.tracks,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_frames_without_update=self.maximum_frames_without_update,
        )

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []
        for det_id in valid_indices:
            best_track = None
            best_iou = 0.0
            det_box = detection_boxes[det_id : det_id + 1]
            for tracker in self.tracks:
                if tracker.time_since_update > 0:
                    continue
                iou = float(
                    sv.box_iou_batch(tracker.get_state_bbox()[None, :], det_box)[0, 0]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_track = tracker

            if best_track is None or best_iou < self.minimum_iou_threshold:
                continue

            if (
                best_track.number_of_successful_updates
                >= self.minimum_consecutive_frames
                and best_track.tracker_id == -1
            ):
                best_track.tracker_id = SORTKalmanBoxTracker.get_next_tracker_id()

            out_det_indices.append(int(det_id))
            out_tracker_ids.append(int(best_track.tracker_id))

        tracked_detections = (
            detections[out_det_indices] if out_det_indices else sv.Detections.empty()
        )
        tracked_detections.tracker_id = np.array(out_tracker_ids, dtype=int)
        return tracked_detections

    def _associate(
        self,
        detections: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
        track_subset: list[int] | None = None,
        use_confidence: bool = True,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if track_subset is None:
            track_subset = list(range(len(self.tracks)))

        if len(track_subset) == 0 or len(detections) == 0:
            return [], track_subset, [int(det_id) for det_id in detections]

        subset_tracks = [self.tracks[i] for i in track_subset]
        iou_matrix = get_iou_matrix(subset_tracks, boxes[detections])

        if use_confidence:
            conf_matrix = np.tile(confidences[detections], (len(track_subset), 1))
            similarity = (
                self.association_lambda * iou_matrix
                + (1.0 - self.association_lambda) * conf_matrix
            )
        else:
            similarity = iou_matrix

        cost_matrix = 1.0 - similarity
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_pairs: list[tuple[int, int]] = []
        matched_track_rows: set[int] = set()
        matched_det_cols: set[int] = set()

        for row, col in zip(row_indices, col_indices):
            if similarity[row, col] < self.minimum_iou_threshold:
                continue
            track_idx = track_subset[int(row)]
            det_idx = int(detections[int(col)])
            matched_pairs.append((track_idx, det_idx))
            matched_track_rows.add(int(row))
            matched_det_cols.add(int(col))

        unmatched_track_ids = [
            track_subset[i]
            for i in range(len(track_subset))
            if i not in matched_track_rows
        ]
        unmatched_detection_ids = [
            int(detections[i])
            for i in range(len(detections))
            if i not in matched_det_cols
        ]

        return matched_pairs, unmatched_track_ids, unmatched_detection_ids

    def reset(self) -> None:
        """Reset tracker state for a new video sequence."""
        self.tracks = []
