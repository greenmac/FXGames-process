from rename_image import rename_images
from moviepy import VideoFileClip
from pathlib import Path
from utils_tool import get_folder_path
from utils_tool import timer

def resized_video(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip_resized: VideoFileClip = clip.resized(new_size=(1280, 720))
    clip_resized = clip_resized.with_fps(30)
    clip_resized.write_videofile(
        output_path,
        codec="libx264",   # H.264 壓縮
        preset="medium",
        ffmpeg_params=["-crf", "14"]  # 品質參數，數值越小品質越好，檔案也越大
    )

def video_to_frames(video_path: str, output_dir: str, fps: int = 30):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(video_path).with_fps(fps)
    clip.write_images_sequence(
        f"{output_dir}/%09d.png",
        fps=fps
    )

    clip.close()
    print(f"Frames saved to {output_dir}")

@timer
def main(root_path, source, source_path, all_jobs):
    for job in all_jobs:
        clip = job['clip']
        start = job['start']
        filename = 'close_flat_hands'
        input_path = f"{source_path}/video_ori/{source}_{clip}_{filename}.mp4"
        output_path = f"{source_path}/video_adj/{clip}_{filename}.mp4"
        resized_video(input_path, output_path)

        part_name = 'part2_open_cards'
        output_dir = rename_input_path = f"{source_path}/frames_adj/{clip}_{part_name}"
        video_to_frames(output_path, output_dir)

        frames_renamed_path = f"{root_path}/{source}/frames_renamed"
        rename_output_path = f"{frames_renamed_path}/{clip}_part2_open_cards"
        rename_images(rename_input_path, rename_output_path, start)

if __name__ == "__main__":
    root_path = './data'
    source = 'kling'
    source_path = get_folder_path(root_path, source)
    all_jobs = [
        # {'clip': 'clip_1', 'start': 111}, 
        # {'clip': 'clip_2', 'start': 175}, 
        # {'clip': 'clip_3', 'start': 141}, 
        {'clip': 'clip_4', 'start': 180},
    ]

    main(root_path, source, source_path, all_jobs)