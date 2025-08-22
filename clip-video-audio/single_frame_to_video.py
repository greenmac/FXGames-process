from moviepy import VideoFileClip
from moviepy.video.VideoClip import ImageClip

def single_frame_to_video(input_path, output_path, t, duration=2.5, fps=24):
    video = VideoFileClip(input_path)
    img_clip:VideoFileClip = video.to_ImageClip(t).with_duration(duration)
    img_clip = img_clip.with_fps(fps)
    img_clip.write_videofile(output_path, codec="libx264", fps=fps)
    video.close()
    img_clip.close()

if __name__ == "__main__":
    root_path = './data'
    input_path = f'{root_path}/ori/Dealing_cards_01.mp4'
    output_path = f'{root_path}/adj/Ori_BankerPlayerPoints.mp4'
    single_frame_to_video(
        input_path,
        output_path,
        t=9.4,
        duration=2.5,
        fps=24
    )