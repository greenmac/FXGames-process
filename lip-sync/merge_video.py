from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips

def merge_mp4s(input_dir, output_path):
    input_dir = Path(input_dir)
    mp4_files = sorted(input_dir.glob("*.mp4"))  # 排序以確保順序一致

    if not mp4_files:
        raise FileNotFoundError(f"No mp4 files found in {input_dir}")

    print(f"Found {len(mp4_files)} mp4 files. Merging...")

    clips = [VideoFileClip(str(f)) for f in mp4_files]
    final_clip = concatenate_videoclips(clips, method="compose")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4
    )

    for clip in clips:
        clip.close()
    final_clip.close()

if __name__ == "__main__":
    merge_mp4s(
        input_dir=r".\data\lipsync\output\clip_1",
        output_path=Path(r".\data\lipsync\output\final\lipsync_1_2.mp4")
    )
