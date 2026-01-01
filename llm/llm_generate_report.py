import os
from google import genai

api_key = "AIzaSyAuYyA9Q05tJqYPUkkqD-6W534Us-CELxE"
client = genai.Client(api_key=api_key)

VIDEO_PATH = "/Users/kellynie/Desktop/yolo_project/output_videos/output_video.avi"

system_prompt = """
You analyze tennis match videos that already contain:
- Player speeds
- Ball speed
- Court keypoints
- Player poses / keypoints
- Bounding boxes and annotations

Do NOT perform detection. Your job is to interpret the visual information into structured tennis analytics.

Your output MUST be valid JSON in the following format:

{
  "match_summary": "",
  "player_analysis": {
    "player1": {
      "strong_shots": [],
      "weak_shots": [],
      "footwork_notes": "",
      "distance_coverage_meters": 0
    },
    "player2": {
      "strong_shots": [],
      "weak_shots": [],
      "footwork_notes": "",
      "distance_coverage_meters": 0
    }
  },
  "shot_success_grid": {
    "grid_1": 0.0, "grid_2": 0.0, "grid_3": 0.0,
    "grid_4": 0.0, "grid_5": 0.0, "grid_6": 0.0,
    "grid_7": 0.0, "grid_8": 0.0, "grid_9": 0.0
  }
}
"""

user_prompt = """
Analyze this tennis match video using the annotated data (player speeds, ball speeds, court keypoints).

Return:
1. Match summary
2. Strong and weak shots for each player
3. Footwork quality and distance covered by each player
4. Shot success percentage in each of the 9 court grids
"""

def main():
    # Read video as bytes
    with open(VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    # Send video + prompts to Gemini
    resp = client.models.generate_content(
        model="gemini-2.0",
        contents=[
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt},
            {"inline_data": {"mime_type": "video/avi", "data": video_bytes}}
        ]
    )

    print(resp.text)


if __name__ == "__main__":
    main()

