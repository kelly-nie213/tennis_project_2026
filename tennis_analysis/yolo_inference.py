import cv2
from ultralytics import YOLO

# Load YOLO model (update path if needed)
ball_model = YOLO("/Users/kellynie/Desktop/sciencetennis_project/tennis_analysis/models/last.pt")


def detect_ball(frame):
    """
    Detect tennis ball in a frame.
    Returns:
        annotated_frame
        ball_coords -> (x, y) or None
    """
    annotated = frame.copy()
    result = ball_model(frame, conf=0.3)[0]

    ball_coords = None

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            ball_coords = (cx, cy)

            # draw ball
            cv2.circle(annotated, ball_coords, 5, (0, 255, 0), -1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break  # only track best ball

    return annotated, ball_coords
