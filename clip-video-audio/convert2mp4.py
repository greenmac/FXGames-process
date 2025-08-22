from moviepy import VideoFileClip
from utils_tool import timer

@timer
def mkv_to_mp4(input_path: str, output_path: str, fps: int = 30):
    clip = VideoFileClip(input_path)
    clip.write_videofile(
        output_path,
        codec="libx264",      # 視訊轉 H.264
        # audio_codec="aac",    # 音訊轉 AAC, 如果轉 AAC, vscode 播放就無法聽到聲音
        fps=fps,
        preset="medium",
        ffmpeg_params=["-crf", "14", "-b:a", "192k"]  # 視訊品質 & 音訊比特率
    )

    clip.close()

if __name__ == "__main__":
    mkv_to_mp4('./data/ori/EVO_ori.mkv', './data/output/EVO_ori.mp4')
