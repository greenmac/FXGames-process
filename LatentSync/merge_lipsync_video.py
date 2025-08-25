from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips
from utils_tool import timer

@timer
def merge_with_moviepy(inputs: list[str | Path], output: str | Path, fps: int = 30, crf: int = 18, preset: str = "medium"):
    """
    Merge MP4 files using MoviePy, forcing fps=30 and re-encoding.
    """
    clips = []
    for p in inputs:
        clip = VideoFileClip(str(p))
        clip = clip.with_fps(fps)  # 強制 fps=30
        clips.append(clip)

    final_clip = concatenate_videoclips(clips, method="compose")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    final_clip.write_videofile(
        str(out),
        fps=fps,
        codec="libx264",
        # audio_codec="aac",
        preset=preset,
        ffmpeg_params=["-crf", str(crf), "-movflags", "+faststart"],
    )

    final_clip.close()
    for c in clips:
        c.close()

    print(f"[OK] merged → {out}")


if __name__ == "__main__":
    root_path = Path('./data')
    lipsync_clip_path = Path(f"{root_path}/output_lipsync")
    inputs = [
        f'{lipsync_clip_path}/lipsync_clip_game_start.mp4',
        f'{lipsync_clip_path}/lipsync_clip_loop.mp4',
        f'{lipsync_clip_path}/lipsync_clip_1_part1_draw_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_1_part2_open_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_1_part3_collect_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_game_start.mp4',
        f'{lipsync_clip_path}/lipsync_clip_loop.mp4',
        f'{lipsync_clip_path}/lipsync_clip_2_part1_draw_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_2_part2_open_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_2_part3_collect_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_game_start.mp4',
        f'{lipsync_clip_path}/lipsync_clip_loop.mp4',
        f'{lipsync_clip_path}/lipsync_clip_3_part1_draw_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_3_part2_open_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_3_part3_collect_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_game_start.mp4',
        f'{lipsync_clip_path}/lipsync_clip_loop.mp4',
        f'{lipsync_clip_path}/lipsync_clip_4_part1_draw_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_4_part2_open_cards.mp4',
        f'{lipsync_clip_path}/lipsync_clip_4_part3_collect_cards.mp4',
    ]

    output_dir = root_path / "lipsync_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lipsync_clip_final.mp4"
    merge_with_moviepy(inputs, output_path, fps=30, crf=14, preset="medium")
