import cv2
import numpy as np
import pandas as pd
from copy import deepcopy

from utils import (
    read_video,
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt


# ======================================================
# CONFIG
# ======================================================
INPUT_VIDEO = "/Users/kellynie/Desktop/sciencetennis_project/tennis_analysis/input_videos/input_video.mp4"
OUTPUT_VIDEO = "/Users/kellynie/Desktop/sciencetennis_project/tennis_analysis/output_videos/output_video.mp4"

TABLE_WIDTH = 320
FPS = 24

RECOVERY_DIST_METERS = 2   # how close to baseline center = recovered


# ======================================================
# BOUNCE DETECTOR
# ======================================================
class BounceDetector:
    def __init__(self, history=8, min_rise=8):
        self.y_vals = []
        self.history = history
        self.min_rise = min_rise
        self.in_fall = False
        self.last_min_y = None
        self.bounce_fired = False

    def update(self, y):
        if y is None:
            return
        self.y_vals.append(y)
        if len(self.y_vals) > self.history:
            self.y_vals.pop(0)

    def get_bounce_status(self):
        if len(self.y_vals) < 4:
            return None

        y_prev = self.y_vals[-2]
        y_curr = self.y_vals[-1]

        if y_curr > y_prev:
            self.in_fall = True
            self.last_min_y = y_curr
            self.bounce_fired = False
            return None

        if self.in_fall:
            self.last_min_y = min(self.last_min_y, y_curr)
            if (y_prev - y_curr) > self.min_rise and not self.bounce_fired:
                self.bounce_fired = True
                self.in_fall = False
                return "In" if self.last_min_y > 200 else "Out"

        return None


# ======================================================
# RECOVERY TRACKER
# ======================================================
class RecoveryTracker:
    def __init__(self):
        self.active = {1: None, 2: None}
        self.last_result = {1: None, 2: None}

    def start(self, pid, frame_idx):
        self.active[pid] = frame_idx

    def update(self, pid, frame_idx, player_pos, baseline_center_px, px_to_meter):
        start_frame = self.active[pid]
        if start_frame is None:
            return

        dist_px = measure_distance(player_pos, baseline_center_px)
        dist_m = dist_px * px_to_meter

        if dist_m <= RECOVERY_DIST_METERS:
            time_sec = (frame_idx - start_frame) / FPS
            self.last_result[pid] = f"{time_sec:.2f}s"
            self.active[pid] = None

    def cancel_if_needed(self, pid):
        if self.active[pid] is not None:
            self.last_result[pid] = "No recovery"
            self.active[pid] = None


# ======================================================
# TABLE DRAWING
# ======================================================
def draw_table(height, ball_coords, player_coords, bounce_status, recovery_status):
    table = np.full((height, TABLE_WIDTH, 3), 245, dtype=np.uint8)

    y = 40
    dy = 28

    # ---- BALL INFO ----
    cv2.putText(table, "BALL INFO", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    y += dy

    if ball_coords:
        cv2.putText(table, f"Ball: {ball_coords}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        cv2.putText(table, "Ball: not detected", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)

    # ---- PLAYER INFO ----
    y += dy * 2
    cv2.putText(table, "PLAYER INFO", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    y += dy

    for pid in [1, 2]:
        text = f"P{pid}: {player_coords.get(pid, '---')}"
        cv2.putText(table, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y += dy

    # ---- RECOVERY ----
    y += dy
    cv2.putText(table, "RECOVERY", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    y += dy

    for pid in [1, 2]:
        val = recovery_status.get(pid, "---")
        cv2.putText(table, f"P{pid}: {val}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y += dy

    # ---- BOUNCE ----
    y += dy
    cv2.putText(table, "BOUNCE", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    y += dy

    if bounce_status:
        color = (0, 180, 0) if bounce_status == "In" else (0, 0, 255)
        cv2.putText(table, bounce_status, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(table, "...", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 140, 140), 2)

    return table


# ======================================================
# MAIN
# ======================================================
def main():
    frames = read_video(INPUT_VIDEO)

    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(
        model_path="/Users/kellynie/Desktop/sciencetennis_project/tennis_analysis/models/last.pt"
    )

    player_dets = player_tracker.detect_frames(frames)
    ball_dets = ball_tracker.interpolate_ball_positions(
        ball_tracker.detect_frames(frames)
    )

    court_detector = CourtLineDetector(
        "/Users/kellynie/Desktop/sciencetennis_project/tennis_analysis/models/keypoints_model.pth"
    )
    court_kps = court_detector.predict(frames[0])
    player_dets = player_tracker.choose_and_filter_players(court_kps, player_dets)

    mini_court = MiniCourt(frames[0])
    player_mc, ball_mc = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_dets, ball_dets, court_kps
    )

    # Baseline centers (mini-court space)
    baseline_centers = mini_court.get_baseline_centers()

    px_to_meter = constants.DOUBLE_LINE_WIDTH / mini_court.get_width_of_mini_court()

    bounce_detector = BounceDetector()
    recovery_tracker = RecoveryTracker()

    final_frames = []

    for i, frame in enumerate(frames):
        ball_coords = None
        if ball_dets[i]:
            x1, y1, x2, y2 = next(iter(ball_dets[i].values()))
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            ball_coords = (cx, cy)
            bounce_detector.update(cy)

        player_coords = {}
        if player_dets[i]:
            for pid, (x1, y1, x2, y2) in player_dets[i].items():
                px, py = int((x1 + x2) / 2), int((y1 + y2) / 2)
                player_coords[pid] = (px, py)

        # ---- Detect shot (ball closest to player) ----
        if ball_coords and player_coords:
            shooter = min(
                player_coords.keys(),
                key=lambda p: measure_distance(player_coords[p], ball_coords)
            )
            recovery_tracker.start(shooter, i)
            recovery_tracker.cancel_if_needed(1 if shooter == 2 else 2)

        # ---- Update recovery ----
        for pid in player_coords:
            recovery_tracker.update(
                pid,
                i,
                player_coords[pid],
                baseline_centers[pid],
                px_to_meter
            )

        table = draw_table(
            frame.shape[0],
            ball_coords,
            player_coords,
            bounce_detector.get_bounce_status(),
            recovery_tracker.last_result
        )

        final_frames.append(np.hstack((frame, table)))

    save_video(final_frames, OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
