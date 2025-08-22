import subprocess
from pathlib import Path

def remove_watermark(input_file: str, output_file: str,
                     x: int, y: int, w: int, h: int, show: int = 0):
    """
    Remove watermark using ffmpeg delogo filter.
    
    input_file: 原始影片路徑
    output_file: 輸出影片路徑
    x, y: 浮水印左上角座標
    w, h: 浮水印寬高
    show: 是否顯示紅框 (1=顯示, 0=實際移除)
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-vf", f"delogo=x={x}:y={y}:w={w}:h={h}:show={show}",
        "-c:a", "copy",
        output_file
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    input_path = "clip_1_close_flat_hand.mp4"
    output_path = "clip_1_close_flat_hand_no_watermark.mp4"

    remove_watermark(input_path, output_path,
                     x=1600, y=925, w=300, h=110, show=0)
