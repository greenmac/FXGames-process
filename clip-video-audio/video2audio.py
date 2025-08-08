from moviepy import VideoFileClip
from pathlib import Path

def mp4_to_wav(mp4_path, wav_path):
    video = VideoFileClip(str(mp4_path))
    audio = video.audio
    if audio is None:
        print(f"Warning: {mp4_path} has no audio track, skipped.")
    else:
        audio.write_audiofile(str(wav_path), codec='pcm_s16le')
        audio.close()
    video.close()

if __name__ == "__main__":
    mp4_root = Path('./data/LipSync_Baccarat_EN_video')
    wav_root = Path('./data/LipSync_Baccarat_EN_audio')
    wav_root.mkdir(exist_ok=True, parents=True)

    mp4_files = list(mp4_root.glob('*.mp4'))

    for mp4_path in mp4_files:
        wav_path = wav_root / (mp4_path.stem + '.wav')
        print(f'Convert {mp4_path} -> {wav_path}')
        mp4_to_wav(mp4_path, wav_path)
