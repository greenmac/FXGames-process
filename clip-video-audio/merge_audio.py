import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip

def merge_and_time_stretch_to_duration(wav_paths, target_duration_sec, output_audio_dir):
    all_audio = []
    sr = None
    for p in wav_paths:
        y, _sr = sf.read(str(p))
        if sr is None:
            sr = _sr
        elif _sr != sr:
            raise ValueError(f"Sample rate mismatch: {p} ({_sr} vs {sr})")
        all_audio.append(y)
    merged = np.concatenate(all_audio)

    original_duration = len(merged) / sr
    speed_factor = original_duration / target_duration_sec
    print(f"Original duration: {original_duration:.2f}s, Target duration: {target_duration_sec:.2f}s, Speed factor: {speed_factor:.4f}")

    stems = [Path(w).stem for w in wav_paths]
    filename = "merged_" + "_".join(stems) + ".wav"
    output_audio_dir = Path(output_audio_dir)
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_audio_path = output_audio_dir / filename

    merged_stretched = pyrb.time_stretch(merged, sr, speed_factor)
    sf.write(str(output_audio_path), merged_stretched, sr)
    print("Audio Output:", output_audio_path)
    return output_audio_path

def mux_audio_to_video(video_path, audio_path, output_path):
    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path)).with_duration(video.duration)
    video = video.with_audio(audio)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    print("Final video:", output_path)

if __name__ == "__main__":
    video_path = "./data/adj/clip_04_BankerPlayerPoints.mp4"
    video = VideoFileClip(video_path)
    target_duration_sec = video.duration
    print(f"Video duration: {target_duration_sec:.2f}s")

    wavs = [
        './data/LipSync_Baccarat_EN_audio/LipSync_Banker_8_points.wav',
        './data/LipSync_Baccarat_EN_audio/LipSync_Banker_5_points.wav'
    ]
    output_audio_dir = "./data/LipSync_Baccarat_EN_merge_audio"
    output_audio_path = merge_and_time_stretch_to_duration(wavs, target_duration_sec, output_audio_dir)

    output_video_path = "./data/LipSync_Baccarat_EN_merged_target/LipSync_04_BankerPlayerPoints.mp4"
    mux_audio_to_video(video_path, output_audio_path, output_video_path)
