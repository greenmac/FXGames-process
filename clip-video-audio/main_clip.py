from moviepy import VideoFileClip
from pathlib import Path


def main(input_path, output_path, start_time, end_time):
    clip = VideoFileClip(input_path).subclipped(start_time, end_time)
    print('Cut-out clip duration:', clip.duration)
    clip.write_videofile(output_path, codec="libx264", fps=clip.fps)
    clip.close()

if __name__ == "__main__":
    root_path = './data'
    # input_path = f'{root_path}/ori/Dealing_cards_01.mp4'
    input_path = f'{root_path}/ori/Dealing_20250811.mp4'
    total_duration = VideoFileClip(input_path).duration

    source = Path(input_path).stem

    # output_path = f'{root_path}/output/clip_{source}_01_PleasePlaceYourBets.mp4'
    # start_time = 0.0
    # end_time = 2.6

    # output_path = f'{root_path}/output/clip_{source}_02_NoMoreBets.mp4'
    # start_time = 2.6
    # end_time = 5.6

    # output_path = f'{root_path}/output/clip_{source}_03_Deal.mp4'
    # start_time = 5.6
    # end_time = 9.9

    # output_path = f'{root_path}/output/clip_{source}_04_BankerPlayerPoints.mp4'
    # start_time = 9.9
    # end_time = 12.9

    # output_path = f'{root_path}/output/clip_{source}_05_BankerWins.mp4'
    # start_time = 12.9
    # end_time = 16.2

    # output_path = f'{root_path}/output/clip_{source}_06_GapsInTheHand.mp4'
    # start_time = 16.2
    # end_time = 19.0

    # output_path = f'{root_path}/output/clip_{source}_07_PleasePlaceYourBets.mp4'
    # start_time = 19.0
    # end_time = 21.6

    # output_path = f'{root_path}/output/clip_{source}_08_NoMoreBets.mp4'
    # start_time = 21.6
    # end_time = 24.6
    
    # output_path = f'{root_path}/output/clip_{source}_09_Deal.mp4'
    # start_time = 24.6
    # end_time = 31.5
    
    # output_path = f'{root_path}/output/clip_{source}_10_BankerPlayerPoints.mp4'
    # start_time = 31.5
    # end_time = 34.5
    
    output_path = f'{root_path}/output/clip_{source}_11_PlayerWins.mp4'
    start_time = 34.5
    end_time = total_duration

    main(input_path, output_path, start_time, end_time)