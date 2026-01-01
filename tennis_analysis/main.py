import cv2
import numpy as np
import pandas as pd
from copy import deepcopy

from utils import (
    read_video,
    save_video,
    measure_distance,
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

BALL_MODEL = "/Users/kellynie/Desktop/sciencetennis_project/tennis_analysis/models/last.pt"
COURT_MODEL = "/Users/kellynie/Desktop/sciencetennis_project/tennis_analysis/models/keypoints_model.pth"

FPS = 24
TABLE_WIDTH = 320


# ======================================================
# TABLE DRAWING
# ======================================================
def draw_ball_table(height, ball_coords, last_ball_speed):
    table = np.full((height, TABLE_WIDTH, 3), 245, dtype=np.uint8)

    y = 40
    dy = 32

    cv2.putText(table, "BALL INFO", (20, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    y += dy * 2

    if ball_coords:
        cv2.putText(table, f"X: {ball_coords[0]}", (20, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += dy
        cv2.putText(table, f"Y: {ball_coords[1]}", (20, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += dy
    else:
        cv2.putText(table, "Ball not detected", (20, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
        y += dy * 2

    y += dy
    cv2.putText(table, "Last Shot Speed", (20, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y += dy

    if last_ball_speed > 0:
        cv2.putText(table, f"{last_ball_speed:.1f} km/h", (20, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 0), 2)
    else:
        cv2.putText(table, "...", (20, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 2)

    return table


def draw_speed_table(height, stats_row):
    table = np.full((height, TABLE_WIDTH, 3), 235, dtype=np.uint8)

    y = 40
    dy = 30

    cv2.putText(table, "SPEED STATS", (20, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    y += dy * 2

    for pid in [1, 2]:
        cv2.putText(table, f"PLAYER {pid}", (20, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        y += dy

        cv2.putText(
            table,
            f"Last Shot: {stats_row[f'player_{pid}_last_shot_speed']:.1f} km/h",
            (20, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        y += dy

        cv2.putText(
            table,
            f"Avg Shot: {stats_row[f'player_{pid}_average_shot_speed']:.1f} km/h",
            (20, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        y += dy

        cv2.putText(
            table,
            f"Avg Move: {stats_row[f'player_{pid}_average_player_speed']:.1f} km/h",
            (20, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        y += dy * 1.5

    return table


# The following is the latest change that merge original features and newly generated features: Jan 1 2026
# ======================================================
# MAIN
# ======================================================
def main():
    video_frames = read_video(INPUT_VIDEO)

    # -------------------------------
    # DETECTION
    # -------------------------------
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path=BALL_MODEL)

    player_dets = player_tracker.detect_frames(video_frames)
    ball_dets = ball_tracker.detect_frames(video_frames)
    ball_dets = ball_tracker.interpolate_ball_positions(ball_dets)

    # -------------------------------
    # COURT
    # -------------------------------
    court_detector = CourtLineDetector(COURT_MODEL)
    court_kps = court_detector.predict(video_frames[0])
    player_dets = player_tracker.choose_and_filter_players(court_kps, player_dets)

    mini_court = MiniCourt(video_frames[0])

    player_mc, ball_mc = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_dets, ball_dets, court_kps
    )

    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_dets)

    # -------------------------------
    # STATS CALCULATION
    # -------------------------------
    stats = [{
        "frame_num": 0,
        "player_1_number_of_shots": 0,
        "player_1_total_shot_speed": 0,
        "player_1_last_shot_speed": 0,
        "player_1_total_player_speed": 0,
        "player_1_last_player_speed": 0,
        "player_2_number_of_shots": 0,
        "player_2_total_shot_speed": 0,
        "player_2_last_shot_speed": 0,
        "player_2_total_player_speed": 0,
        "player_2_last_player_speed": 0,
    }]

    for i in range(len(ball_shot_frames) - 1):
        start = ball_shot_frames[i]
        end = ball_shot_frames[i + 1]
        dt = (end - start) / FPS
        if dt == 0:
            continue

        ball_dist_px = measure_distance(ball_mc[start][1], ball_mc[end][1])
        ball_dist_m = convert_pixel_distance_to_meters(
            ball_dist_px,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        ball_speed = (ball_dist_m / dt) * 3.6

        players = player_mc[start]
        hitter = min(players.keys(),
                     key=lambda p: measure_distance(players[p], ball_mc[start][1]))
        opponent = 1 if hitter == 2 else 2

        opp_dist_px = measure_distance(player_mc[start][opponent],
                                       player_mc[end][opponent])
        opp_dist_m = convert_pixel_distance_to_meters(
            opp_dist_px,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        opp_speed = (opp_dist_m / dt) * 3.6

        cur = deepcopy(stats[-1])
        cur["frame_num"] = start

        cur[f"player_{hitter}_number_of_shots"] += 1
        cur[f"player_{hitter}_total_shot_speed"] += ball_speed
        cur[f"player_{hitter}_last_shot_speed"] = ball_speed

        cur[f"player_{opponent}_total_player_speed"] += opp_speed
        cur[f"player_{opponent}_last_player_speed"] = opp_speed

        stats.append(cur)

    df = pd.DataFrame(stats)
    frames_df = pd.DataFrame({"frame_num": range(len(video_frames))})
    df = pd.merge(frames_df, df, on="frame_num", how="left").ffill().fillna(0)

    df["player_1_average_shot_speed"] = df["player_1_total_shot_speed"] / df["player_1_number_of_shots"].replace(0, 1)
    df["player_2_average_shot_speed"] = df["player_2_total_shot_speed"] / df["player_2_number_of_shots"].replace(0, 1)
    df["player_1_average_player_speed"] = df["player_1_total_player_speed"] / df["player_2_number_of_shots"].replace(0, 1)
    df["player_2_average_player_speed"] = df["player_2_total_player_speed"] / df["player_1_number_of_shots"].replace(0, 1)

    # -------------------------------
    # DRAW VIDEO
    # -------------------------------
    frames = player_tracker.draw_bboxes(video_frames, player_dets)
    frames = ball_tracker.draw_bboxes(frames, ball_dets)
    frames = court_detector.draw_keypoints_on_video(frames, court_kps)

    frames = mini_court.draw_mini_court(frames)
    frames = mini_court.draw_points_on_mini_court(frames, player_mc)
    frames = mini_court.draw_points_on_mini_court(frames, ball_mc, color=(0, 255, 255))

    final_frames = []

    for i, frame in enumerate(frames):
        stats_row = df.iloc[i]

        ball_coords = None
        if ball_dets[i]:
            x1, y1, x2, y2 = next(iter(ball_dets[i].values()))
            ball_coords = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        last_ball_speed = max(
            stats_row["player_1_last_shot_speed"],
            stats_row["player_2_last_shot_speed"]
        )

        table1 = draw_ball_table(frame.shape[0], ball_coords, last_ball_speed)
        table2 = draw_speed_table(frame.shape[0], stats_row)

        final_frames.append(np.hstack((frame, table1, table2)))

    save_video(final_frames, OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
