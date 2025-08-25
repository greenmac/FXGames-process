from moviepy import ImageSequenceClip
from pathlib import Path
from utils_tool import timer
import cv2
import re
import numpy as np


def _natural_key(p: Path):
    """Natural sort: 1.png, 2.png, 10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.stem)]


def _alpha_blend_to_bg(bgra, bg_color=(0, 0, 0)):
    """BGRA -> BGR，將透明度疊到底色（B, G, R）。"""
    b, g, r, a = cv2.split(bgra)
    a = a.astype(np.float32) / 255.0
    a3 = cv2.merge([a, a, a])
    bg = np.full_like(bgra[:, :, :3], bg_color, dtype=np.uint8).astype(np.float32)
    fg = cv2.merge([b, g, r]).astype(np.float32)
    out = (fg * a3 + bg * (1.0 - a3)).clip(0, 255).astype(np.uint8)
    return out


def _ensure_size_and_rgb(img, target_wh, bg_color=None):
    """
    Resize 到 target_wh (W,H) 並轉成 RGB 3 通道。
    若有 alpha 且提供 bg_color，會做 alpha blend；否則直接丟棄 alpha。
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    tw, th = target_wh
    if (w, h) != (tw, th):
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    # 先確保 3 通道 BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = _alpha_blend_to_bg(img, bg_color) if bg_color is not None else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 3:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # MoviePy 需要 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _prep_frames_in_memory(image_paths, target_wh=None, bg_color=None):
    """
    讀入所有圖片，統一尺寸並轉成 RGB（numpy arrays），全部留在記憶體。
    回傳: (frames, base_wh)
    """
    # 決定基準尺寸
    if target_wh is None:
        first = None
        for p in image_paths:
            first = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if first is not None:
                break
        if first is None:
            raise RuntimeError("Cannot read any image to determine base size.")
        h, w = first.shape[:2]
        target_wh = (w, h)

    frames = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read: {p}")
        frames.append(_ensure_size_and_rgb(img, target_wh, bg_color=bg_color))
    return frames, target_wh


def image_to_video(image_folder: Path, output_path: Path, fps: int = 30, crf: int = 14, preset: str = 'medium', target_wh=None, bg_color=None):
    images = sorted(image_folder.glob('*.png'), key=_natural_key)
    if not images:
        raise ValueError('No PNG files found in the folder.')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames, base_wh = _prep_frames_in_memory(images, target_wh=target_wh, bg_color=bg_color)
    print(f'[Info] Unified size: {base_wh[0]}x{base_wh[1]}, Channels=3 (RGB), Frames={len(frames)}')

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        str(output_path),
        codec='libx264',
        audio=False,
        preset=preset,
        ffmpeg_params=['-crf', str(crf)],
        fps=fps,
    )
    clip.close()

    print(f'Exported video: {output_path}')


@timer
def main(images_path, video_path, clips):
    DEFAULTS = dict(fps=30, crf=14, preset='medium', target_wh=None)
    for c in clips:
        cfg = {**DEFAULTS, **{k: c[k] for k in DEFAULTS if isinstance(c, dict) and k in c}}
        folder = images_path / (c if isinstance(c, str) else c['name'])
        output_name = f"{(c if isinstance(c, str) else c['name'])}_close_flat_hands.mp4"
        output_path = video_path / output_name
        image_to_video(folder, output_path, **cfg)


if __name__ == '__main__':
    root_path = Path('./data')
    images_path = root_path / 'kling' / 'adj'
    video_path = root_path / 'kling' / 'video_adj'

    clips = ["clip_1", "clip_2", "clip_3", "clip_4"]

    main(images_path, video_path, clips)
