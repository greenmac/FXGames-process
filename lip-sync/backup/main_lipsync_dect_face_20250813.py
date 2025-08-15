# https://github.com/mowshon/lipsync
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip, concatenate_audioclips
from moviepy.audio.AudioClip import AudioClip
from lipsync import LipSync
from utils_tool import timer

def detect_face_runs(video_path, min_conf=0.4, min_run_frames=3, fps_override=None):
    """以 cv2 讀檔偵測臉部區段，回傳 [((t0, t1), has_face), ...] 與 fps"""
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

def _silent_audio(duration, fps=44100, nch=2):
    """產生靜音 AudioClip（float32）"""
    return AudioClip(lambda t: np.zeros((np.size(t), nch), dtype=np.float32),
                     duration=duration, fps=fps)

@timer
def lipsync_with_passthrough(
    face_video: str,
    audio_file: str,
    output_file: str,
    model_ckpt='weights/wav2lip.pth',
    device='cuda',
    img_size=96,
    silence_sr=44100
):
    from math import isfinite

    # 取得可靠 fps/時長
    v = VideoFileClip(face_video)
    orig_fps = getattr(v, "fps", None) or getattr(v.reader, "fps", None) or 30.0
    vid_dur = float(v.duration)
    eps_t = max(1e-4, 0.5 / float(orig_fps))  # 半格安全邊界

    # 【新增】統一的高畫質 x264 參數（CRF 越小越清晰，檔案也越大）
    # 建議 14~20；你可以先用 16，若檔案太大再調到 18 或 20
    x264_hq = [
        "-crf", "14",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        "-vsync", "cfr", # 保持 CFR
        "-g", str(int(orig_fps * 2)), # 合理的 GOP（1~2秒都可以），保留 B-frames 增加壓縮效率與畫質
    ]

    # 臉部區段（時間以相同 fps 計）
    runs, _ = detect_face_runs(face_video, min_conf=0.4, min_run_frames=3, fps_override=orig_fps)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("cache/segments")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    lip = LipSync(model='wav2lip', checkpoint_path=model_ckpt,
                  nosmooth=True, device=device, cache_dir='cache',
                  img_size=img_size, save_cache=True)

    global_audio = AudioFileClip(audio_file)
    aud_dur = float(global_audio.duration)

    out_clips = []
    seg_id = 0
    total_frames = 0  # 以「重編碼後段落」的實際幀數累加

    for (t0, t1), has_face in runs:
        # clamp 邊界
        t0 = max(0.0, min(t0, vid_dur))
        t1 = max(0.0, min(t1, vid_dur))
        if t1 <= t0:
            continue

        # 以 frame 對齊
        n_frames = int(round((t1 - t0) * orig_fps))
        if n_frames <= 0:
            continue
        target_dur = n_frames / orig_fps
        t1_fixed = min(t0 + target_dur, vid_dur)

        seg_id += 1

        # 原片段（無音訊）—給 lipsync 用；【修改】高畫質輸出
        seg_vid = v.subclipped(t0, t1_fixed)
        seg_video_path = str(tmp_dir / f"seg_{seg_id:03d}.mp4")
        seg_audio_path = str(tmp_dir / f"seg_{seg_id:03d}.wav")
        seg_out_path   = str(tmp_dir / f"seg_{seg_id:03d}_out.mp4")
        seg_std_path   = str(tmp_dir / f"seg_{seg_id:03d}_std.mp4")  # 標準化段（用來串接）

        seg_vid.without_audio().write_videofile(
            seg_video_path,
            codec="libx264",
            audio=False,
            fps=orig_fps,
            logger=None,
            ffmpeg_params=x264_hq  # 【修改】高畫質
        )

        # 只做 pad：夾取可用音訊 + 補靜音
        a0 = min(t0, aud_dur)
        a1 = min(t1_fixed, aud_dur)
        if a1 > a0:
            seg_aud = global_audio.subclipped(a0, a1)
            pad_tail = target_dur - (a1 - a0)
            if pad_tail > 1e-6:
                seg_aud = concatenate_audioclips([seg_aud, _silent_audio(pad_tail, fps=silence_sr)])
        else:
            seg_aud = _silent_audio(target_dur, fps=silence_sr)

        # lipsync 與 base_clip（video-only）
        if has_face:
            seg_aud.write_audiofile(seg_audio_path, fps=silence_sr, logger=None)
            lip.sync(seg_video_path, seg_audio_path, seg_out_path)
            base_clip = VideoFileClip(seg_out_path).without_audio()
        else:
            base_clip = VideoFileClip(seg_video_path).without_audio()

        # 【新增】確保 lipsync 輸出的解析度與原段一致（避免被縮小）
        try:
            src_w, src_h = seg_vid.size
            out_w, out_h = base_clip.size
            if (out_w, out_h) != (src_w, src_h):
                base_clip = base_clip.resize((src_w, src_h))
        except Exception:
            pass

        # 三者最小值決定安全長度
        base_dur = float(base_clip.duration) if base_clip.duration else 0.0
        safe_seg_dur = max(0.0, min(target_dur, float(seg_aud.duration), base_dur) - eps_t)
        if safe_seg_dur <= 0:
            base_clip.close()
            seg_vid.close()
            continue

        # 對齊段長
        base_clip = base_clip.with_duration(safe_seg_dur)

        # ⭐ 用新 AudioFileClip 以避免 NoneType stdout
        if has_face:
            seg_aud_clip = AudioFileClip(seg_audio_path).with_duration(safe_seg_dur)
        else:
            tmp_pad_wav = str(tmp_dir / f"seg_{seg_id:03d}_pad.wav")
            seg_aud.with_duration(safe_seg_dur).write_audiofile(tmp_pad_wav, fps=silence_sr, logger=None)
            seg_aud_clip = AudioFileClip(tmp_pad_wav).with_duration(safe_seg_dur)

        # 合成本段並重編碼成高畫質 CFR（鎖 fps）
        combined = base_clip.with_audio(seg_aud_clip).with_fps(orig_fps)
        combined.write_videofile(
            seg_std_path,
            codec="libx264",
            audio_codec="aac",
            fps=orig_fps,
            logger=None,
            audio_bitrate="192k",     # 【新增】音訊品質
            ffmpeg_params=x264_hq     # 【修改】高畫質
        )

        # 讀回標準段加入串接清單
        std_clip = VideoFileClip(seg_std_path)
        if isfinite(float(std_clip.duration or 0)):
            std_clip = std_clip.with_duration(max(0.0, float(std_clip.duration) - eps_t))
        out_clips.append(std_clip)

        # 以「重編碼後段落」實際幀數累加
        enc_frames = int(round(float(std_clip.duration) * orig_fps))
        total_frames += enc_frames

        # 關掉暫存 clip
        seg_aud_clip.close()
        base_clip.close()
        combined.close()
        seg_vid.close()

    # 串接
    if not out_clips:
        v.close(); global_audio.close()
        raise RuntimeError("No output clips produced. Check face detection runs.")
    final = concatenate_videoclips(out_clips, method="chain")

    # 以實際幀數估計期望長度，上限為原片長；略縮 epsilon
    expected_dur = (total_frames / orig_fps)
    target_final = max(0.0, min(expected_dur, vid_dur) - eps_t)
    final = final.with_duration(target_final)

    # 最終音訊（pad）
    if aud_dur >= target_final:
        final_audio = global_audio.subclipped(0, target_final)
    else:
        head = global_audio.subclipped(0, aud_dur)
        tail = _silent_audio(target_final - aud_dur, fps=silence_sr)
        final_audio = concatenate_audioclips([head, tail])

    final = final.with_audio(final_audio)

    # 最終輸出：高畫質 CFR
    final.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        fps=orig_fps,
        logger=None,
        audio_bitrate="192k",            # 【新增】音訊品質
        ffmpeg_params=x264_hq + ["-movflags", "+faststart"]  # 【修改】高畫質 + 快速起播
    )

    # 清理
    v.close()
    global_audio.close()
    for c in out_clips:
        c.close()
    final.close()

if __name__ == "__main__":
    root_path = './data/lipsync'
    ori_path = f'{root_path}/ori'
    out_path = f'{root_path}/output'
    
    face_video = f'{ori_path}/ori_Dealing_20250811.mp4'
    audio_file = f'{ori_path}/tts_Dealing_20250811.wav'
    output_file = f'{out_path}/lipsync_Dealing_20250811.mp4'

    lipsync_with_passthrough(face_video, audio_file, output_file)
