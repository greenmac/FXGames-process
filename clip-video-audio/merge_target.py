from moviepy import VideoFileClip, concatenate_videoclips
from pathlib import Path

def merge_mp4_folder(input_dir, output_path):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(input_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"No mp4 files found in {input_dir}")
        return

    print("Merging these files in order:")
    for f in mp4_files:
        print("  ", f.name)

    clips = [VideoFileClip(str(f)) for f in mp4_files]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    for c in clips:
        c.close()
    final_clip.close()
    print(f"Merged video saved to: {output_path}")

if __name__ == "__main__":
    input_dir = "./data/LipSync_Baccarat_EN_merged_target"
    output_path = "./data/LipSync_Baccarat_EN_merged_final/LipSync_Baccarat_EN_final.mp4"
    merge_mp4_folder(input_dir, output_path)
