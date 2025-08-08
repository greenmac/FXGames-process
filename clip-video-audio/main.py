from moviepy import VideoFileClip
from moviepy import CompositeVideoClip


def main(input_path, output_path, start_time, end_time):
    clip = VideoFileClip(input_path).subclipped(start_time, end_time)
    print('Cut-out clip duration:', clip.duration)
    clip.write_videofile(output_path, codec="libx264", fps=clip.fps)
    clip.close()

if __name__ == "__main__":
    root_path = './data'
    input_path = f'{root_path}/ori/Dealing_cards_01.mp4'
    
    # output_path = f'{root_path}/adj/clip_01_PleasePlaceYourBets.mp4'
    # start_time = 12.4
    # end_time = 14.6

    # output_path = f'{root_path}/adj/clip_02_NoMoreBets.mp4'
    # start_time = 14.6
    # end_time = 16.9

    # output_path = f'{root_path}/adj/clip_03_Deal.mp4'
    # start_time = 16.9
    # end_time = 20.8

    # output_path = f'{root_path}/adj/clip_04_BankerPlayerPoints.mp4'
    # start_time = 20.8
    # end_time = 23.6

    output_path = f'{root_path}/adj/clip_05_BankerOrPlayerWins.mp4'
    start_time = 23.6
    end_time = 28.2

    main(input_path, output_path, start_time, end_time)