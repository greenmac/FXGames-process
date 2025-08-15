from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips
from moviepy import AudioClip
from moviepy.audio.AudioClip import AudioArrayClip  # ★ 重要：用來精確寫出 numpy 組裝的音檔
from lipsync import LipSync
from utils_tool import timer
import math
import numpy as np
import subprocess

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def safe_subclip(clip, start, end, eps=1e-3):
    """確保 end 不超過 clip.duration - eps"""
    safe_end = min(end, clip.duration - eps)
    safe_start = max(start, 0)
    return clip.subclipped(safe_start, safe_end)


def _rms_db(x: np.ndarray, eps=1e-12) -> float:
    """計算 RMS dBFS（mono 1D）。"""
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.maximum(x**2, eps))))
    return 20.0 * math.log10(max(rms, eps))

def audio_has_speech(audio_path: str,
                     sr: int = 16000,
                     frame_ms: int = 30,
                     hop_ms: int = 15,
                     db_threshold: float = -45.0,
                     min_speech_ratio: float = 0.10) -> bool:
    """
    粗略檢測音檔是否有語音：
    - 將音訊轉為指定 sr 的 waveform（[-1, 1] 範圍）
    - 分成 frame_ms 毫秒一幀
    - 計算每幀 RMS dBFS，超過 db_threshold 視為有聲
    - 有聲比例 >= min_speech_ratio 即視為有語音
    """
    with AudioFileClip(audio_path) as a:
        y = a.to_soundarray(fps=sr)  # shape: (N, channels)
    if y.ndim > 1:
        y = y.mean(axis=1)  # 轉 mono
    n = len(y)
    if n == 0:
        return False

    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms  / 1000)

    num_frames = 0
    speech_frames = 0
    i = 0
    while i + frame_len <= n:
        seg = y[i:i+frame_len]
        db = _rms_db(seg)
        if db > db_threshold:
            speech_frames += 1
        num_frames += 1
        i += hop_len

    ratio = speech_frames / max(1, num_frames)
    return ratio >= min_speech_ratio

def make_silence_wav_for_video(video_path: str, out_wav_path: str, sr: int = 44100):
    """做一條與 video 等長的全靜音 WAV（供 Wav2Lip 驅動用）。"""
    with VideoFileClip(video_path) as v:
        dur = float(v.duration)
    # 用 AudioArrayClip 產生精確靜音（不走函式型 AudioClip 避免浮點誤差）
    target_samples = max(1, int(np.floor(dur * sr)))
    arr = np.zeros((target_samples, 2), dtype=np.float32)
    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
    AudioArrayClip(arr, fps=sr).write_audiofile(out_wav_path, fps=sr, logger=None)
    return out_wav_path

def _get_video_props(path: str):
    """Return (fps, (w, h)) using MoviePy."""
    with VideoFileClip(path) as v:
        fps = getattr(v, "fps", None) or getattr(v.reader, "fps", None) or 30.0
        w, h = v.size
    return fps, (w, h)

def _ensure_same_resolution(src_video: str, candidate_video: str) -> str:
    """
    若 candidate 與 src 解析度不同就重採到一致，回傳修正後路徑。
    """
    _, (sw, sh) = _get_video_props(src_video)
    _, (cw, ch) = _get_video_props(candidate_video)
    if (sw, sh) == (cw, ch):
        return candidate_video

    fixed_path = str(Path(CACHE_DIR) / (Path(candidate_video).stem + "_resized.mp4"))
    with VideoFileClip(candidate_video) as v:
        v = v.resize((sw, sh))
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
    用 ffmpeg 將 video(copy) + audio(aac) 合併（不重編碼視訊，零畫質損失）。
    """
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
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

def _ensure_silent_wav_if_needed(audio_path: Path, filename_stem: str, duration_sec: float = 1.0, sr: int = 44100):
    """
    若音檔不存在且檔名以 'Blank' 開頭，建立 1 秒靜音 WAV。
    """
    if audio_path.exists():
        return
    if filename_stem.lower().startswith("blank"):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        # 用 AudioArrayClip 產生靜音檔（避免函式型 Clip 的浮點誤差）
        nsamp = max(1, int(np.floor(duration_sec * sr)))
        arr = np.zeros((nsamp, 2), dtype=np.float32)
        AudioArrayClip(arr, fps=sr).write_audiofile(str(audio_path), fps=sr, logger=None)
        print(f"[INFO] Created 1s silent wav: {audio_path}")


def build_audio_aligned_to_video(
    video_path: str,
    audio_path: str,
    out_wav_path: str,
    start_offset_sec: float = 0.0,
    sr: int = 44100
) -> str:
    """
    生成一條「長度精準（樣本級）」且「比影片少 1ms」的 WAV：
      - 前置 start_offset_sec 的靜音
      - 接上原始音檔（超過剩餘長度就截斷）
      - 不足的補尾端靜音
    全程用 NumPy 組裝，避免 MoviePy 內部浮點長度累積誤差。
    """
    # 影片長度（秒）
    with VideoFileClip(video_path) as v:
        vid_dur = float(v.duration)

    # 目標長度：故意比影片短 1ms，避免越界
    safety = 1e-3
    target_samples = max(1, int(np.floor((vid_dur - safety) * sr)))

    # 讀音檔為指定取樣率、浮點 [-1, 1]
    with AudioFileClip(audio_path) as a0:
        y = a0.to_soundarray(fps=sr)  # (N, C) float (-1,1)
    if y.ndim == 1:
        y = y[:, None]
    # 統一用立體聲
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    y = y.astype(np.float32, copy=False)

    # 計算各段樣本數
    head_samples   = max(0, int(round(start_offset_sec * sr)))
    remain_samples = max(0, target_samples - head_samples)
    a0_samples     = min(len(y), remain_samples)

    # 組裝輸出：固定精確 target_samples
    out = np.zeros((target_samples, 2), dtype=np.float32)
    if a0_samples > 0:
        out[head_samples:head_samples + a0_samples, :] = y[:a0_samples, :2]

    # 寫檔（長度=target_samples/sr，重讀時不會顯示成 3.930000）
    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
    AudioArrayClip(out, fps=sr).write_audiofile(out_wav_path, fps=sr, logger=None)
    return out_wav_path

# ----------------------------
# 視訊輸出
# ----------------------------
def write_mp4_aac(
        video_path: str, audio_path: str, out_file: str,
        fps: float | None = None, crf: int = 14, preset: str = "slow",
        sr: int = 44100
    ):
    """
    高品質 MP4 (H.264 + AAC) 重編碼輸出。
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
                "-g", str(int(round(_fps * 2))),
                "-bf", "2",
                "-b_strategy", "1",
                "-sc_threshold", "40",
            ],
        )

def run_lipsync_to_tmp(face_video: str, audio_file: str, tmp_out_video: str, device: str = "cuda"):
    lip = LipSync(
        model='wav2lip',
        checkpoint_path='weights/wav2lip_gan.pth',
        nosmooth=True,
        device=device,
        cache_dir=str(CACHE_DIR),
        img_size=96,
        save_cache=True,
        static=True,           # ★ 只在第一幀偵測人臉，後面沿用
        pads=(0, 10, 0, 0),    # ★ 視情況給上邊界留一點空間，避免嘴巴被裁
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
    prefer_copy_video=True → ffmpeg stream copy（不重編碼視訊）
    否則 → moviepy 重編碼（可控 CRF）
    """
    if prefer_copy_video:
        mux_mp4_with_copy(video_src, audio_src, out_mp4)
        return

    fps, _ = _get_video_props(video_src)
    write_mp4_aac(video_src, audio_src, out_mp4, fps=fps, crf=crf_mp4, preset=preset_mp4)

# ----------------------------
# 主流程
# ----------------------------
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

        # Output path
        out_dir = lipsync_root_path / "output" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / f"lipsync_{source}_{part}_{filename}.mp4"

        if not face.exists():
            raise FileNotFoundError(f"face video not found: {face}")
        if not audio.exists():
            raise FileNotFoundError(f"audio file not found: {audio}")

        # 以「樣本級對齊」方式產生 padded_wav（保證比影片短 1ms）
        padded_wav = build_audio_aligned_to_video(
            video_path=str(face),
            audio_path=str(audio),
            out_wav_path=str(CACHE_DIR / f"tmp_{part}_{filename}.wav"),
            start_offset_sec=offset_sec,
            sr=44100,
        )

        # === 無語音 → 用「純靜音」作為驅動跑 Wav2Lip（畫面照動、嘴巴閉合），再把 padded_wav mux 回去 ===
        silent = not audio_has_speech(padded_wav, sr=16000,
                                      frame_ms=30, hop_ms=15,
                                      db_threshold=-45.0,
                                      min_speech_ratio=0.10)

        if silent:
            driver_sil = str(CACHE_DIR / f"drv_sil_{part}_{filename}.wav")
            make_silence_wav_for_video(str(face), driver_sil, sr=44100)

            tmp_video = str(CACHE_DIR / f"tmp_sil_{part}_{filename}.mp4")
            run_lipsync_to_tmp(str(face), driver_sil, tmp_video, device="cuda")  # or "cpu"

            fixed_tmp = _ensure_same_resolution(str(face), tmp_video)
            export_mp4_only(fixed_tmp, padded_wav, str(out_mp4),
                            crf_mp4=12, preset_mp4="slow",
                            prefer_copy_video=False)
            continue
        # 有語音時
        if do_lipsync:
            tmp_video = str(CACHE_DIR / f"tmp_{part}_{filename}.mp4")
            run_lipsync_to_tmp(str(face), padded_wav, tmp_video, device="cuda")  # or "cpu"
            fixed_tmp = _ensure_same_resolution(str(face), tmp_video)
            export_mp4_only(fixed_tmp, padded_wav, str(out_mp4),
                            crf_mp4=12, preset_mp4="slow",
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
        {"part": "part1", "filename": "NoMoreBets", "offset": 4.0, "lipsync": True},
        {"part": "part2", "filename": "GoodLuck", "offset": 3.5, "lipsync": False},
        {"part": "part3", "filename": "BlankCollectionCards", "offset": 0.1, "lipsync": False},  # auto-create 1s silent wav if missing
        {"part": "transition_1_1_two_times", "filename": "PleasePlaceYourBets", "offset": 0.1, "lipsync": True},
    ]

    main(tasks, source, face_root_path, lipsync_root_path)
