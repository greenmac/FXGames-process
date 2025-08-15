from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips
from moviepy import AudioClip
from lipsync import LipSync
from utils_tool import timer
import cv2
import mediapipe as mp
import numpy as np
import subprocess

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_video_props(path: str):
    """Return (fps, (w, h)) using MoviePy."""
    with VideoFileClip(path) as v:
        fps = getattr(v, "fps", None) or getattr(v.reader, "fps", None) or 30.0
        w, h = v.size
    return fps, (w, h)

def _ensure_same_resolution(src_video: str, candidate_video: str) -> str:
    """
    If candidate_video resolution differs from src_video, resize candidate to match.
    Returns path to a file that matches src resolution (possibly the same input).
    """
    _, (sw, sh) = _get_video_props(src_video)
    _, (cw, ch) = _get_video_props(candidate_video)
    if (sw, sh) == (cw, ch):
        return candidate_video

    fixed_path = str(Path(CACHE_DIR) / (Path(candidate_video).stem + "_resized.mp4"))
    with VideoFileClip(candidate_video) as v:
        v = v.resize((sw, sh))
        # visually-lossless-ish mezzanine for the resized intermediate
        v.write_videofile(
            fixed_path,
            codec="libx264",
            fps=getattr(v, "fps", None) or getattr(v.reader, "fps", None) or 30.0,
            logger=None,
            ffmpeg_params=["-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
                           "-profile:v", "high", "-level", "4.2", "-movflags", "+faststart"]
        )
    return fixed_path

def mux_mp4_with_copy(video_src: str, audio_src: str, out_mp4: str, sr: int = 44100, audio_bitrate: str = "192k"):
    """
    Mux video (copy) + audio (AAC) with ffmpeg CLI to avoid re-encoding the video stream.
    Keeps original quality perfectly.
    """
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command:
    # -c:v copy : keep original video bitstream (no quality loss)
    # -c:a aac  : encode audio to AAC for MP4 compatibility
    # -shortest : stop at the shortest stream (audio already aligned to video)
    # -movflags +faststart : better streaming/startup
    cmd = [
        "ffmpeg", "-y",
        "-i", video_src,
        "-i", audio_src,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(sr),
        "-movflags", "+faststart",
        "-shortest",
        out_mp4
    ]
    subprocess.run(cmd, check=True)

def _silent_clip(duration: float, sr: int = 44100, nch: int = 2) -> AudioClip:
    """Return a silent AudioClip (float32) with the given duration."""
    duration = max(0.0, float(duration))
    return AudioClip(
        lambda t: np.zeros((np.size(t), nch), dtype=np.float32),
        duration=duration,
        fps=sr,
    )

def _ensure_silent_wav_if_needed(audio_path: Path, filename_stem: str, duration_sec: float = 1.0, sr: int = 44100):
    """
    If audio_path does not exist and filename_stem startswith 'Blank',
    create a 1-second silent WAV at audio_path. Otherwise do nothing.
    """
    if audio_path.exists():
        return
    if filename_stem.lower().startswith("blank"):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        _silent_clip(duration_sec, sr=sr, nch=2).write_audiofile(str(audio_path), fps=sr, logger=None)
        print(f"[INFO] Created 1s silent wav: {audio_path}")

def build_audio_aligned_to_video(
    video_path: str,
    audio_path: str,
    out_wav_path: str,
    start_offset_sec: float = 0.0,
    sr: int = 44100
) -> str:
    """
    Build a WAV whose length exactly matches the video's duration:
      - Prepend silence of `start_offset_sec`
      - Append the original audio (trim if it exceeds remaining duration)
      - If still shorter, pad with trailing silence
    """
    with VideoFileClip(video_path) as v:
        target_dur = float(v.duration)

    with AudioFileClip(audio_path) as a0:
        a0_dur = float(a0.duration)
        head = _silent_clip(max(0.0, start_offset_sec), sr=sr)
        remain = max(0.0, target_dur - head.duration)

        if a0_dur > remain:
            a_mid = a0.subclipped(0, remain)
        else:
            a_mid = a0

        used = head.duration + float(a_mid.duration)
        tail_len = max(0.0, target_dur - used)
        tail = _silent_clip(tail_len, sr=sr) if tail_len > 1e-6 else None

        parts = [head, a_mid] if tail is None else [head, a_mid, tail]
        a_final = concatenate_audioclips(parts)

        Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
        a_final.write_audiofile(out_wav_path, fps=sr, logger=None)

    return out_wav_path

def detect_face_runs(video_path: str, min_conf: float = 0.4, min_run_frames: int = 3, fps_override: float | None = None):
    """
    Scan the whole video and return [((t0, t1), has_face), ...] and fps.
    Short flickers (< min_run_frames) are merged to stabilize segments.
    """
    cap = cv2.VideoCapture(video_path)
    fps_cv = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_override or (fps_cv if fps_cv and fps_cv > 0 else 30.0)

    det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_conf
    )
    has = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        has.append(1 if r.detections else 0)
    cap.release()

    has = np.array(has, dtype=np.int32)
    runs = []
    i = 0
    while i < len(has):
        j = i
        while j < len(has) and has[j] == has[i]:
            j += 1
        if j - i < min_run_frames:
            if runs:
                (t0, _), prev_has = runs[-1]
                runs[-1] = ((t0, j / fps), prev_has)
            else:
                runs.append(((i / fps, j / fps), 0))
        else:
            runs.append(((i / fps, j / fps), int(has[i])))
        i = j
    return runs, fps

def has_any_face(video_path: str, min_conf: float = 0.4, min_run_frames: int = 3) -> bool:
    """Quick boolean: True if there exists any stable run with face detected."""
    runs, _ = detect_face_runs(video_path, min_conf=min_conf, min_run_frames=min_run_frames, fps_override=None)
    return any(has_face == 1 for (_t0t1, has_face) in runs)

def write_mp4_aac(video_path: str, audio_path: str, out_file: str,
                  fps: float | None = None, crf: int = 14, preset: str = "slow",
                  audio_bitrate: str = "192k", sr: int = 44100):
    """
    High-quality MP4 (H.264 + AAC).
    """
    temp_audio = str(CACHE_DIR / "temp_audio.m4a")
    with VideoFileClip(video_path) as v, AudioFileClip(audio_path) as a:
        _fps = fps or getattr(v, "fps", None) or getattr(v.reader, "fps", None) or 30.0
        a = a.with_fps(sr)
        v_final = v.with_audio(a).with_duration(float(v.duration)).with_fps(_fps)
        v_final.write_videofile(
            out_file,
            codec="libx264",
            audio_codec="aac",
            fps=_fps,
            logger=None,
            temp_audiofile=temp_audio,
            remove_temp=True,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-preset", preset,
                "-crf", str(crf),
                "-profile:v", "high",
                "-level", "4.2",
                "-g", str(int(round(_fps * 2))),  # GOP ~ 2s
                "-bf", "2",
                "-b_strategy", "1",
                "-sc_threshold", "40",
            ],
        )

def run_lipsync_to_tmp(face_video: str, audio_file: str, tmp_out_video: str, device: str = "cuda"):
    """Run Wav2Lip and write to a temporary video."""
    lip = LipSync(
        model='wav2lip',
        checkpoint_path='weights/wav2lip_gan.pth',
        nosmooth=True,
        device=device,              # "cuda" or "cpu"
        cache_dir=str(CACHE_DIR),
        img_size=96,
        save_cache=True,
    )
    lip.sync(face_video, audio_file, tmp_out_video)

def export_mp4_only(
        video_src: str, 
        audio_src: str, 
        out_mp4: str,
        crf_mp4: int = 14, 
        preset_mp4: str = "slow",
        prefer_copy_video: bool = False
    ):
    """
    If prefer_copy_video is True -> use ffmpeg stream copy (no video re-encode).
    Else -> re-encode via MoviePy with high-quality x264 settings.
    """
    if prefer_copy_video:
        mux_mp4_with_copy(video_src, audio_src, out_mp4)
        return

    fps, _ = _get_video_props(video_src)
    write_mp4_aac(video_src, audio_src, out_mp4, fps=fps, crf=crf_mp4, preset=preset_mp4)


@timer
def main(tasks, source: str, face_root_path: Path, lipsync_root_path: Path):
    for t in tasks:
        part = t["part"]
        filename = t["filename"]
        offset_sec = float(t.get("offset", 0.0))
        do_lipsync = bool(t.get("lipsync", True))

        video_name = f"{source}_{part}_{filename}.mp4"
        audio_name = f"tts_{filename}.wav"

        face  = face_root_path / "output" / source / video_name
        audio = lipsync_root_path / "ori" / "audio" / audio_name

        _ensure_silent_wav_if_needed(audio, filename, duration_sec=1.0, sr=44100)

        out_dir = lipsync_root_path / "output" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / f"lipsync_{source}_{part}_{filename}.mp4"

        if not face.exists():
            raise FileNotFoundError(f"face video not found: {face}")
        if not audio.exists():
            raise FileNotFoundError(f"audio file not found: {audio}")

        padded_wav = build_audio_aligned_to_video(
            video_path=str(face),
            audio_path=str(audio),
            out_wav_path=str(CACHE_DIR / f"tmp_{part}_{filename}.wav"),
            start_offset_sec=offset_sec,
            sr=44100,
        )

        if do_lipsync and has_any_face(str(face), min_conf=0.4, min_run_frames=3):
            tmp_video = str(CACHE_DIR / f"tmp_{part}_{filename}.mp4")
            run_lipsync_to_tmp(str(face), padded_wav, tmp_video, device="cuda")  # or "cpu"
            fixed_tmp = _ensure_same_resolution(str(face), tmp_video)
            export_mp4_only(fixed_tmp, padded_wav, str(out_mp4),
                            crf_mp4=12,    # even lower CRF for more detail
                            preset_mp4="slow",
                            prefer_copy_video=False)
        else:
            export_mp4_only(str(face), padded_wav, str(out_mp4),
                            crf_mp4=14, preset_mp4="slow",
                            prefer_copy_video=True)


if __name__ == "__main__":
    source = "clip_1"
    face_root_path = Path("./data/image2video")
    lipsync_root_path = Path("./data/lipsync")

    # Each task is a dict: {"part": str, "filename": str, "offset": float, "lipsync": bool}
    tasks = [
        {"part": "part1", "filename": "NoMoreBets",                  "offset": 4.0, "lipsync": True},
        {"part": "part2", "filename": "GoodLuck",                    "offset": 3.5, "lipsync": False},
        {"part": "part3", "filename": "Blank_OriBankerOrPlayerWins", "offset": 0.1, "lipsync": False},  # ← auto-create 1s silent tts_*.wav if missing
        {"part": "part4", "filename": "PleasePlaceYourBets",         "offset": 0.1, "lipsync": True},
    ]

    main(tasks, source, face_root_path, lipsync_root_path)
