import argparse
import cv2
import base64
import json
import re
from google import genai
from google.genai import types

# ============================
# ARGUMENTS
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="Gemini tennis match analysis")
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Google AI Studio API key"
    )
    return parser.parse_args()

# ============================
# CONFIG
# ============================
VIDEO_PATH = "../tennis_analysis/output_videos/output_video.mp4"
FRAME_STRIDE = 5
MODEL_NAME = "gemini-3-pro-preview"
OUTPUT_TEXT_PATH = "match_summary.txt"

PROMPT_TEXT = """
You are a professional tennis performance analyst.

You are given sequential annotated video frames from a tennis match.
The annotations include:
- Player bounding boxes and IDs
- Frame number
- Ball and player coordinates
- Court lines
- Labeled key points on the court
- Player and ball speeds
- Distance traveled by each player 

IMPORTANT:
- Treat the frames as a continuous video, not independent images.
- Infer movement, shot selection, and positioning across time.
- Do NOT hallucinate events that are not visually supported.
- Give specific frames in parenthesis when giving tips and observations
- Provide specific numbers/data from output_video.mp4 to support your tips and observations

Analyze the match and output STRICT JSON:

{
  "player_1": {
    "strong_shots": [],
    "weak_shots": [],
    "footwork": "",
    "shot_tendencies": ""
  },
  "player_2": {
    "strong_shots": [],
    "weak_shots": [],
    "footwork": "",
    "shot_tendencies": ""
  },
  "overall_match_summary": ""
}
"""

# ============================
# FRAME EXTRACTION
# ============================
def extract_frames(video_path, stride):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % stride == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))

        idx += 1

    cap.release()
    return frames

# ============================
# JSON CLEANING (NEW FIX)
# ============================
def extract_json(text):
    if not text.strip():
        raise ValueError("Gemini returned empty response")

    # Remove ```json fences if present
    text = re.sub(r"```json|```", "", text).strip()

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Gemini response")

    return match.group(0)

# ============================
# FORMAT OUTPUT
# ============================
def format_summary(raw_text):
    json_text = extract_json(raw_text)
    data = json.loads(json_text)

    def format_player(player_name, player_data):
        text = []
        text.append(f"## **{player_name.replace('_', ' ').title()}**\n")

        text.append("**Strong Shots:**")
        if player_data["strong_shots"]:
            for s in player_data["strong_shots"]:
                text.append(f"- **{s}**")
        else:
            text.append("- *None identified*")

        text.append("\n**Weak Shots:**")
        if player_data["weak_shots"]:
            for w in player_data["weak_shots"]:
                text.append(f"- **{w}**")
        else:
            text.append("- *None identified*")

        text.append(f"\n**Footwork Analysis:**\n*{player_data['footwork']}*")
        text.append(f"\n**Shot Tendencies:**\n*{player_data['shot_tendencies']}*\n")

        return "\n".join(text)

    report = []
    report.append("# **Tennis Match Performance Report**\n")
    report.append(format_player("player_1", data["player_1"]))
    report.append(format_player("player_2", data["player_2"]))

    report.append("## **Overall Match Summary**")
    report.append(f"*{data['overall_match_summary']}*")

    return "\n\n".join(report)

# ============================
# GEMINI CALL
# ============================
def analyze_match(api_key):
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"}
    )

    frames_b64 = extract_frames(VIDEO_PATH, FRAME_STRIDE)

    parts = [types.Part(text=PROMPT_TEXT)]

    for f in frames_b64:
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(f),
                ),
                media_resolution={"level": "media_resolution_high"}
            )
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(parts=parts)]
    )

    raw_text = response.text or ""
    formatted_text = format_summary(raw_text)

    with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    print(f"\nFormatted match summary saved to: {OUTPUT_TEXT_PATH}\n")

# ============================
# RUN
# ============================
if __name__ == "__main__":
    args = parse_args()
    analyze_match(args.api_key)
