from moviepy import VideoFileClip
from pathlib import Path
from utils_tool import timer

@timer
def main(input_path, output_path, start_time, end_time, fps=30):
    clip: VideoFileClip = VideoFileClip(input_path).subclipped(start_time, end_time)
    print(f'Cut-out clip duration: {clip.duration} seconds')

    clip.write_videofile(
        output_path,
        codec="libx264",
        fps=clip.fps,
        temp_audiofile="temp-audio.mp3",
        remove_temp=True
    )
    clip.close()


if __name__ == "__main__":
    root_path = './data'
    input_path = f'{root_path}/no_editing/60_Round.mp4'

    source = Path(input_path).stem
    
    output_path = f'{root_path}/ori/clip_{source}.mp4'
    start_time = 121.0
    end_time = 144.0

    main(input_path, output_path, start_time, end_time)