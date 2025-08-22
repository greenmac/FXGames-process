from moviepy import VideoFileClip
from pathlib import Path
from utils_tool import timer
import numpy as np
import shlex
import soundfile as sf
import subprocess
import tempfile

def is_silent_wav(wav_path: Path, db_threshold: float = -50.0) -> bool:
    """用 RMS dBFS 粗略判斷是否靜音（全片低於門檻視為靜音）"""
    y, sr = sf.read(str(wav_path), always_2d=True)
    if y.size == 0:
        return True
    y = y.astype(np.float32, copy=False)
    if y.shape[1] > 1:
        y = y.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.maximum(y**2, 1e-12))))
    db = 20.0 * np.log10(max(rms, 1e-12))
    return db < db_threshold

def run(cmd: list[str]) -> None:
    print('[CMD]', ' '.join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)

def ffprobe_duration_seconds(video_path: Path) -> float:
    out = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1',
         str(video_path)],
        check=True, capture_output=True, text=True
    ).stdout.strip()
    return float(out)

def get_video_duration_moviepy(video_path: Path) -> float:
    '''用 MoviePy 讀取時長（備援），MoviePy 2.x 用法。'''
    with VideoFileClip(str(video_path)) as v:
        return float(v.duration)

def _ffmpeg_resample(in_path: Path, out_path: Path, target_sr: int):
    cmd = [
        'ffmpeg', '-y',
        '-i', str(in_path),
        '-af', f'aresample={target_sr}:resampler=soxr',
        '-ar', str(target_sr),
        '-ac', '2',
        str(out_path),
    ]
    run(cmd)

def build_aligned_audio_wav(
    video_path: Path,
    audio_path: Path,
    out_wav: Path,
    *,
    audio_start_offset_sec: float = 0.0,
    sr: int = 44100,
) -> Path:
    '''
    產出一個「與影片等長」的 WAV，聲音從 offset 開始，不足補尾端靜音、超出就截斷。
    - 長度以 video 為主（-1 ms 安全邊界，避免浮點超長）
    '''
    try:
        vid_dur = ffprobe_duration_seconds(video_path)
    except Exception:
        vid_dur = get_video_duration_moviepy(video_path)

    eps = 1e-3
    target_samples = max(1, int(np.floor((vid_dur - eps) * sr)))

    y, y_sr = sf.read(str(audio_path), always_2d=True)
    if y_sr != sr:
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / "resampled.wav"
            _ffmpeg_resample(audio_path, tmp_wav, sr)
            y, y_sr = sf.read(str(tmp_wav), always_2d=True)

    y = y.astype(np.float32, copy=False)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    head = max(0, int(round(audio_start_offset_sec * sr)))
    out = np.zeros((target_samples, 2), dtype=np.float32)

    remain = max(0, target_samples - head)
    a0 = min(remain, y.shape[0])
    if a0 > 0:
        out[head:head + a0, :] = y[:a0, :2]

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), out, sr)
    return out_wav

def mux_copy_video_aac_audio(video_src: Path, audio_wav: Path, out_mp4: Path, *, sr: int = 44100, audio_bitrate: str = '192k'):
    '''
    用 ffmpeg 合成：video stream copy（零畫質損失）+ AAC 音訊編碼。
    - `-shortest` 可避免任何音訊/視訊殘留導致拖長（但我們前面已經精準截等長）
    參考：ffmpeg 濾鏡與 stream copy 說明。:contentReference[oaicite:1]{index=1}
    '''
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_src),
        '-i', str(audio_wav),
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', audio_bitrate, '-ar', str(sr),
        '-movflags', '+faststart',
        '-shortest',
        str(out_mp4),
    ]
    run(cmd)

def run_latentsync_inference(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    unet_config_path: Path = Path('configs/unet/stage2_512.yaml'),
    ckpt_path: Path = Path('checkpoints/latentsync_unet.pt'),
    inference_steps: int = 50,
    guidance_scale: float = 1.5,
    enable_deepcache: bool = True,
    seed: int = 1247,
    temp_dir: Path = Path('temp'),
):
    temp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        'python', '-m', 'scripts.inference',
        '--unet_config_path', str(unet_config_path),
        '--inference_ckpt_path', str(ckpt_path),
        '--inference_steps', str(inference_steps),
        '--guidance_scale', str(guidance_scale),
        '--video_path', str(video_path),
        '--audio_path', str(audio_path),
        '--video_out_path', str(output_path),
        '--seed', str(seed),
        '--temp_dir', str(temp_dir),
    ]
    if enable_deepcache:
        cmd.append('--enable_deepcache')
    run(cmd)

def process_one(
    video_path: str,
    audio_path: str,
    output_path: str,
    do_lipsync: bool,
    audio_start_offset_sec: float = 0.0,
    sr: int = 44100,
    close_mouth_if_silent: bool = True,  # ★ 新增
):
    video_p = Path(video_path)
    audio_p = Path(audio_path)
    out_p   = Path(output_path)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        aligned_wav = tmp_dir / "aligned.wav"

        # 1) 產生「等長對齊」音訊（含 offset、尾端補靜音）
        build_aligned_audio_wav(
            video_p, audio_p, aligned_wav,
            audio_start_offset_sec=audio_start_offset_sec,
            sr=sr,
        )

        # 2) 決定是否要用「閉嘴 lipsync」
        if not do_lipsync and close_mouth_if_silent and is_silent_wav(aligned_wav):
            # 用純靜音來驅動 lip-sync，讓模型輸出閉嘴畫面
            # 生成一條與影片等長、同取樣率的「純靜音 WAV」當驅動
            silent_driver = tmp_dir / "driver_silence.wav"
            sf.write(str(silent_driver), np.zeros_like(sf.read(str(aligned_wav), always_2d=True)[0]), sr)

            tmp_synced = tmp_dir / "closed_mouth.mp4"
            run_latentsync_inference(
                video_path=video_p,
                audio_path=silent_driver,   # ★ 用純靜音驅動，讓嘴型關閉
                output_path=tmp_synced,
            )
            # 再把「模型輸出的閉嘴畫面」與「對齊後（等長）的音訊」合在一起成最終成品
            mux_copy_video_aac_audio(tmp_synced, aligned_wav, out_p, sr=sr)

        else:
            if do_lipsync:
                run_latentsync_inference(
                    video_path=video_p,
                    audio_path=aligned_wav,
                    output_path=out_p,
                )
            else:
                mux_copy_video_aac_audio(video_p, aligned_wav, out_p, sr=sr)

    print(f'[OK] Saved → {out_p}')

def get_tasks(
        offset_silence_draw_cards: float,
        offset_open_cards: float,
        offset_silence_collect_cards: float,
        do_lipsync_silence_draw_cards: bool = False,
        do_lipsync_open_cards: bool = False,
        do_lipsync_silence_collect_cards: bool = False,
        do_mute_silence_draw_cards: bool = True,
        do_mute_open_cards: bool = False,
        do_mute_silence_collect_cards: bool = True,
    ):
    return [
        {
            'idx': '01',
            'filename': 'draw_cards',
            'offset_sec': offset_silence_draw_cards,
            'offset_sec': [
                {'start': offset_silence_draw_cards, 'tts_content': 'draw_cards'},
            ],
            'do_lipsync': do_lipsync_silence_draw_cards,
            'do_mute': do_mute_silence_draw_cards,
        },
        {
            'idx': '02',
            'filename': 'open_cards',
            'offset_sec': [
                {'start': offset_open_cards, 'tts_content': 'open_cards'},
            ],
            'do_lipsync': do_lipsync_open_cards,
            'do_mute': do_mute_open_cards,
        },
        {
            'idx': '03',
            'filename': 'collect_cards',
            'offset_sec': [
                {'start': offset_silence_collect_cards, 'tts_content': 'collect_cards'},
            ],
            'do_lipsync': do_lipsync_silence_collect_cards,
            'do_mute': do_mute_silence_collect_cards,
        },
    ]

def _load_wav_stereo_resampled(wav_path: Path, sr: int, tmpdir: Path) -> tuple[np.ndarray, int]:
    """讀入 wav，若取樣率不符則用 ffmpeg 重採樣到 sr；確保回傳 float32、2ch。"""
    y, y_sr = sf.read(str(wav_path), always_2d=True)
    if y_sr != sr:
        tmp_wav = tmpdir / f"resamp_{wav_path.stem}.wav"
        _ffmpeg_resample(wav_path, tmp_wav, sr)
        y, y_sr = sf.read(str(tmp_wav), always_2d=True)
    y = y.astype(np.float32, copy=False)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    return y, sr


def _get_video_duration_safe(video_path: Path) -> float:
    try:
        return ffprobe_duration_seconds(video_path)
    except Exception:
        return get_video_duration_moviepy(video_path)


def _render_tts_bed(
    video_path: Path,
    audio_dir: Path,
    offsets: list[dict],
    sr: int,
    tmpdir: Path,
) -> Path:
    """
    將多段 TTS 依 start 秒數貼到同一條音軌上，長度 = 影片長度（-1ms 安全邊界）。
    offsets 例：
      [
        {'start': 0.1, 'tts_content': 'please_place_your_bets'},
        {'start': 9.0, 'tts_content': 'no_more_bets'},
      ]
    對應檔名：audio_dir / f'tts_{tts_content}.wav'
    """
    eps = 1e-3
    vid_dur = _get_video_duration_safe(video_path)
    target_samples = max(1, int(np.floor((vid_dur - eps) * sr)))
    bed = np.zeros((target_samples, 2), dtype=np.float32)

    # 逐段貼上
    for seg in offsets:
        t0 = float(seg['start'])
        key = str(seg['tts_content'])
        wav_path = audio_dir / f"tts_{key}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"TTS wav not found: {wav_path}")
        y, _ = _load_wav_stereo_resampled(wav_path, sr, tmpdir)
        head = max(0, int(round(t0 * sr)))
        if head >= target_samples:
            continue
        a0 = min(target_samples - head, y.shape[0])
        if a0 <= 0:
            continue
        # 疊加（允許重疊）
        bed[head:head + a0, :] += y[:a0, :2]

    # 簡單避免爆音：clip 到 [-1, 1]
    np.clip(bed, -1.0, 1.0, out=bed)

    out_wav = tmpdir / "tts_bed.wav"
    sf.write(str(out_wav), bed, sr)
    return out_wav


@timer
def main(all_jobs, ori_root_path, output_root_path):
    for job in all_jobs:
        source = job['source']
        tasks = job['tasks']

        # 允許 tasks 是 list 或單一 dict（像你貼的例子）
        if isinstance(tasks, dict):
            tasks_iter = [tasks]
        else:
            tasks_iter = tasks

        for t in tasks_iter:
            idx = t.get('idx', '')
            filename = t['filename']
            offset_sec = t['offset_sec']   # 可能是 float 或 list[dict]
            do_lipsync = t['do_lipsync']
            do_mute = t['do_mute']

            video_name = f'{source}_{idx}_{filename}.mp4' if idx else f'{source}_{filename}.mp4'
            output_name = f'lipsync_{source}_{idx}_{filename}.mp4' if idx else f'lipsync_{source}_{filename}.mp4'

            video_path  = ori_root_path / 'video' / video_name
            audio_dir   = ori_root_path / 'audio'
            output_path = output_root_path / output_name

            with tempfile.TemporaryDirectory() as td:
                tmpdir = Path(td)

                # 判斷三種情況：
                # A) do_mute=True -> 直接用 tts_silence.wav
                # B) offset_sec 是 list -> 合成一條多段 TTS 音軌
                # C) offset_sec 是數值 -> 維持舊邏輯（單一 TTS 檔名：tts_{filename}.wav + offset）
                if do_mute:
                    audio_path = audio_dir / 'tts_silence.wav'
                    if not audio_path.exists():
                        # 沒有靜音檔就現做一條與影片等長的純靜音
                        sr = 44100
                        dur = _get_video_duration_safe(video_path)
                        samples = max(1, int(np.floor((dur - 1e-3) * sr)))
                        silence = np.zeros((samples, 2), dtype=np.float32)
                        audio_path = tmpdir / 'tts_silence_autogen.wav'
                        sf.write(str(audio_path), silence, sr)

                    # 靜音不需要 offset，直接送進去、audio_start_offset_sec=0
                    process_one(
                        video_path=video_path,
                        audio_path=audio_path,
                        output_path=output_path,
                        do_lipsync=do_lipsync,
                        audio_start_offset_sec=0.0,
                        sr=44100,
                    )

                elif isinstance(offset_sec, list):
                    # B) 多段 TTS：合成一條 bed，再以 0 offset 丟進去
                    bed_wav = _render_tts_bed(
                        video_path=video_path,
                        audio_dir=audio_dir,
                        offsets=offset_sec,
                        sr=44100,
                        tmpdir=tmpdir,
                    )
                    process_one(
                        video_path=video_path,
                        audio_path=bed_wav,
                        output_path=output_path,
                        do_lipsync=do_lipsync,
                        audio_start_offset_sec=0.0,  # 重要：已經合成到正確時間點，這裡就 0
                        sr=44100,
                    )

                else:
                    # C) 單一 offset（舊行為）
                    audio_path = audio_dir / f'tts_{filename}.wav'
                    process_one(
                        video_path=video_path,
                        audio_path=audio_path,
                        output_path=output_path,
                        do_lipsync=do_lipsync,
                        audio_start_offset_sec=float(offset_sec),
                        sr=44100,
                    )

if __name__ == '__main__':
    root_path = Path('./data')
    ori_root_path = Path(f'{root_path}/ori')
    output_root_path = Path(f'{root_path}/output')
    
    all_jobs = [
        {
            "source": "clip_1",
            "tasks": get_tasks(
                offset_silence_draw_cards=0.0,
                offset_open_cards=3.5,
                offset_silence_collect_cards=0.0,
            ),
        },
        {
            "source": "clip_2",
            "tasks": get_tasks(
                offset_silence_draw_cards=0.0,
                offset_open_cards=4.6,
                offset_silence_collect_cards=0.0,
            ),
        },
        {
            "source": "clip_3",
            "tasks": get_tasks(
                offset_silence_draw_cards=0.0,
                offset_open_cards=4.5,
                offset_silence_collect_cards=0.0,
            ),
        },
        {
            "source": "clip_4",
            "tasks": get_tasks(
                offset_silence_draw_cards=0.0,
                offset_open_cards=5.6,
                offset_silence_collect_cards=0.0,
            ),
        },
        {
            "source": "clip",
            "tasks": {
                'idx': '',
                'filename': 'game_start',
                'offset_sec': [
                    {'start': 0.1, 'tts_content': 'please_place_your_bets'},
                    {'start': 9.0, 'tts_content': 'no_more_bets'},
                ],
                'do_lipsync': True,
                'do_mute': False,
            },
        },
        {
            "source": "clip",
            "tasks": {
                'idx': '',
                'filename': 'loop',
                'offset_sec': [
                    {'start': 0.0, 'tts_content': 'silence'},
                ],
                'do_lipsync': True,
                'do_mute': True,
            },
        },
    ]
    main(all_jobs, ori_root_path, output_root_path)
