import audiomath as am
from audiomath import Sound
from utils_tool import timer

root_path = './data'
source = 'kokoro'

@timer
def main(clips, start_times, output_file):

    segments = []
    current_time = 0.0

    for s, st in zip(clips, start_times):
        # If the current time is lower than the start time of the audio, insert silence first.
        if st > current_time:
            segments.append(st - current_time)
            current_time = st
        # Then insert the audio
        segments.append(s)
        current_time += s.Duration()
        

    final = am.Concatenate(*segments)
    final.Write(output_file)

if __name__ == "__main__":
    clips = [
        Sound(f"{root_path}/{source}/tts_af_bella_01_PleasePlaceYourBets.wav"),
        Sound(f"{root_path}/{source}/tts_af_bella_02_NoMoreBets.wav"),
        Sound(f"{root_path}/{source}/tts_af_bella_04_Banker8PointsPlayer5Points.wav"),
        Sound(f"{root_path}/{source}/tts_af_bella_05_BankerWins.wav"),
        Sound(f"{root_path}/{source}/tts_af_bella_07_PleasePlaceYourBets.wav"),
        Sound(f"{root_path}/{source}/tts_af_bella_08_NoMoreBets.wav"),
        Sound(f"{root_path}/{source}/tts_af_bella_10_Banker6PointsPlayer9Points.wav"),
        Sound(f"{root_path}/{source}/tts_af_bella_11_PlayerWins.wav"),
    ]
    start_times = [0.1, 2.6, 9.9, 12.9, 19.0, 21.6, 31.5, 34.5]
    output_file = "./data/kokoro/tts_Dealing_20250811.wav"

    main(clips, start_times, output_file)