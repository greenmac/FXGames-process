from moviepy import VideoFileClip
from pathlib import Path
from utils_tool import timer

@timer
def main(input_path, output_path, start_time, end_time, fps=30):
    clip:VideoFileClip = VideoFileClip(input_path).subclipped(start_time, end_time)
    clip.write_videofile(
        output_path,
        codec="libx264",      # 視訊轉 H.264
        # audio_codec="aac",    # 音訊轉 AAC, 如果轉 AAC, vscode 播放就無法聽到聲音
        fps=fps,
        preset="medium",
        ffmpeg_params=["-crf", "14", "-b:a", "192k"]  # 視訊品質 & 音訊比特率
    )
    clip.close()
    print('Cut-out clip duration:', clip.duration)

if __name__ == "__main__":
    root_path = './data'
    input_path = f'{root_path}/ori/EVO_ori.mp4'
    total_duration = VideoFileClip(input_path).duration
    source = Path(input_path).stem

    output_path = f'{root_path}/output/EVO_ori_flat_hands.mp4'
    start_time = 47.5
    end_time = 50.0
    
    # output_path = f'{root_path}/output/clip_{source}_11_PlayerWins.mp4'
    # start_time = 34.5
    # end_time = total_duration

    main(input_path, output_path, start_time, end_time)