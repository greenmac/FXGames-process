from moviepy import VideoFileClip
from moviepy import CompositeVideoClip


def main(input_path, output_path):
    start_time = 9.4
    end_time = 9.5
    clip = (
        VideoFileClip(input_path)
        .subclipped(start_time, end_time)
    )
    clips = [clip.with_start(i * clip.duration) for i in range(25)]
    final = CompositeVideoClip(clips)
    final.write_videofile(output_path, codec="libx264")
    
    clip.reader.close()
    if clip.audio:
        clip.audio.reader.close_proc()
    final.close()

if __name__ == "__main__":
    root_path = './data'
    input_path = f'{root_path}/ori/Dealing_cards_01.mp4'
    output_path = f'{root_path}/adj/clip_Ori_Lip_Sync_BankerPlayerPoints.mp4'
    main(input_path, output_path)