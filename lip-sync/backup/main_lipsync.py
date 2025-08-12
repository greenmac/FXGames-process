# https://github.com/mowshon/lipsync
from lipsync import LipSync
from pathlib import Path
from utils_tool import timer

Path('cache').mkdir(parents=True, exist_ok=True)

root_path = './data/lipsync'
ori_path = f'{root_path}/ori'
out_path = f'{root_path}/out'

@timer
def main(face, audio_file, out_file):
    lip = LipSync(
        model='wav2lip',
        checkpoint_path='weights/wav2lip.pth',
        nosmooth=True,
        device='cuda',
        cache_dir='cache',
        img_size=96,
        save_cache=True,
    )

    lip.sync(face, audio_file, out_file)
    
if __name__ == "__main__":
    # date = '20250811'
    
    # file_name = '01_PleasePlaceYourBets'
    # file_name = '02_NoMoreBets'
    # file_name = '04_BankerPlayerPoints'

    # face = f'{ori_path}/clip_Dealing_{date}_{file_name}.mp4'
    # audio_file = f'{ori_path}/tts_af_{file_name}.wav'
    # output_file = f'{out_path}/lipsync_Dealing_{date}_{file_name}_kokoro.mp4'
    
    face = f'{ori_path}/Dealing_20250811.mp4'
    audio_file = f'{ori_path}/tts_temp.wav'
    out_file = f'{out_path}/lipsync_Dealing_temp.mp4'
    main(face, audio_file, out_file)
