from moviepy import ImageSequenceClip
from pathlib import Path
from utils_tool import timer
import cv2
import re


def _natural_key(p: Path):
    '''Natural sort: 1.png, 2.png, 10.png.'''
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.stem)]

def _to_bgra_4ch(img):
    '''Convert image to 4-channel BGRA (Alpha=255).'''
    if img is None:
        return None
    if img.ndim == 2:
        # Grayscale -> BGR -> BGRA
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 3:
        # BGR -> BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 4:
        # Already BGRA
        pass
    else:
        # Any other unexpected channel count: force convert to BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def _ensure_size_and_4ch(img, target_wh):
    '''Resize to target_wh (W, H) and ensure BGRA 4-channel.'''
    img = _to_bgra_4ch(img)
    if img is None:
        return None
    h, w = img.shape[:2]
    tw, th = target_wh
    if (w, h) != (tw, th):
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return img  # BGRA 4ch

def _prep_images_4ch_fixed_size(image_paths, cache_dir: Path, target_wh=None):
    '''
    Convert all images to fixed size + 4 channels, output to cache_dir.
    If target_wh=None, use the first readable image's size as the base.
    Returns: (list of processed image paths, base_wh)
    '''
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine base size (W,H)
    if target_wh is None:
        first = None
        for p in image_paths:
            first = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if first is not None:
                break
        if first is None:
            raise RuntimeError('Cannot read any image to determine base size.')
        h, w = first.shape[:2]
        target_wh = (w, h)

    fixed_paths = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Failed to read: {p}')
        fixed = _ensure_size_and_4ch(img, target_wh)
        out_p = cache_dir / p.name
        # Save as PNG; OpenCV will store alpha correctly for BGRA images
        ok = cv2.imwrite(str(out_p), fixed)
        if not ok:
            raise RuntimeError(f'Failed to write: {out_p}')
        fixed_paths.append(out_p)
    return fixed_paths, target_wh

def image_to_video(image_folder: Path, output_path: Path, fps: int = 30, crf: int = 14, preset: str = 'medium', target_wh=None):
    images = sorted(image_folder.glob('*.png'), key=_natural_key)
    if not images:
        raise ValueError('No PNG files found in the folder.')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert all frames to fixed size + 4 channels (RGBA)
    cache_dir = image_folder / '_fixed_tmp'
    fixed_paths, base_wh = _prep_images_4ch_fixed_size(images, cache_dir, target_wh=target_wh)
    print(f'[Info] Unified size: {base_wh[0]}x{base_wh[1]}, Channels=4 (RGBA)')

    # Create video from processed frames
    clip = ImageSequenceClip([str(p) for p in fixed_paths], fps=fps)
    clip.write_videofile(
        str(output_path),
        codec='libx264',        # If transparent background is needed, use 'libx264rgb' + '-pix_fmt rgba' (most players do not support it)
        audio=False,
        preset=preset,
        ffmpeg_params=['-crf', str(crf)]
    )
    clip.close()

    print(f'Exported video: {output_path}')

@timer
def main(images_path, video_path, clips):
    DEFAULTS = dict(fps=30, crf=14, preset='medium', target_wh=None)

    for clip in clips:
        cfg = {**DEFAULTS, **{k: clip[k] for k in DEFAULTS if k in clip}}
        images_folder = images_path / clip
        output_name = f'{clip}_close_flat_hands.mp4'
        output_path = video_path / output_name
        image_to_video(images_folder, output_path, **cfg)


if __name__ == '__main__':
    root_path = Path('./data')
    images_path = root_path / 'kling' / 'adj'
    video_path = root_path / 'kling' / 'video_adj'

    clips = ["clip_1", "clip_2", "clip_3", "clip_4"]

    main(images_path, video_path, clips)