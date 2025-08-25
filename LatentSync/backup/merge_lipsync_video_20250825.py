from pathlib import Path
import shlex, subprocess

def run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True)

def _write_concat_list(paths: list[Path], list_file: Path) -> None:

    lines = []
    for p in paths:
        ap = Path(p).resolve().as_posix()  # Windows 也轉成 POSIX 路徑，避免跳脫地獄
        lines.append(f"file '{ap}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")

def merge_mp4(
    inputs: list[str | Path],
    output: str | Path,
    *,
    mode: str = "fast",     # "fast" -> concat demuxer + copy；"safe" -> concat filter + re-encode
    crf: int = 18,
    preset: str = "medium",
    include_audio: bool = True,  # 若有些片段沒有音軌，建議設 False（只合併影像）
):
    paths = [Path(p) for p in inputs]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if mode == "fast":
        # 以 concat demuxer 無損串接
        list_file = out.with_suffix(".concat.txt")
        _write_concat_list(paths, list_file)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",                # 影片/音訊皆 stream copy
            "-movflags", "+faststart",
            str(out),
        ]
        run(cmd)

    elif mode == "safe":
        # 用 concat filter 串接並重新編碼
        cmd = ["ffmpeg", "-y"]
        for p in paths:
            cmd += ["-i", str(p)]

        n = len(paths)
        if include_audio:
            # 需要每個輸入都至少有一條音軌；若沒有，請設 include_audio=False
            inputs_va = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
            filtergraph = f"{inputs_va}concat=n={n}:v=1:a=1[v][a]"
            cmd += [
                "-filter_complex", filtergraph,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(out),
            ]
        else:
            inputs_v = "".join(f"[{i}:v:0]" for i in range(n))
            filtergraph = f"{inputs_v}concat=n={n}:v=1:a=0[v]"
            cmd += [
                "-filter_complex", filtergraph,
                "-map", "[v]",
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                "-movflags", "+faststart",
                str(out),
            ]
        run(cmd)

    else:
        raise ValueError("mode 必須是 'fast' 或 'safe'")

    print(f"[OK] merged → {out}")

if __name__ == "__main__":
    root_path = Path('./data')
    lipsync_clip_path = Path(f"{root_path}/output")
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

    output_dir = root_path / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lipsync_clip_final.mp4"
    merge_mp4(inputs, output_path, mode="fast", crf=14, preset="medium", include_audio=True)