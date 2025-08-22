import subprocess
from pathlib import Path
from utils_tool import timer

@timer
def ffmpeg_to_frames(video_path: str | Path, output_dir: str | Path, fps: int | None = None):
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-q:v", "2",                # 圖片品質 (1~31, 越小越好)
    ]

    if fps:  # 可選：只取固定 fps
        cmd.extend(["-vf", f"fps={fps}"])

    cmd.append(str(output_dir / "frame_%06d.png"))

    subprocess.run(cmd, check=True)
    print(f"Frames saved to {output_dir}")

if __name__ == "__main__":
    ffmpeg_to_frames("./data/output/EVO_ori_flat_hands.mp4", "./data/frames", fps=30)