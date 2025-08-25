from pathlib import Path
from utils_tool import timer
import shutil

@timer
def rename_images(input_path: str, output_folder: str, start: int, *, keep_original=True, dry_run=False):
    input_path = Path(input_path)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.png"))  # 你的檔名已是 000000000.png 格式，字典序即可
    if not files:
        print("No PNG files found."); return

    for i, file in enumerate(files):
        new_index = start + i
        new_name = f"{new_index:09d}.png"
        new_path = output_path / new_name

        if new_path.exists():
            raise FileExistsError(f"Destination exists: {new_path}")

        print(f"{file.name} -> {new_name}")

        if dry_run:
            continue

        if keep_original:
            shutil.copy2(file, new_path)   # 複製，保留原檔
        else:
            file.replace(new_path)         # 等同於安全覆蓋式移動（會把原檔移走）


if __name__ == "__main__":
    # clip = 'clip_1'
    # start = 111

    # clip = 'clip_2'
    start = 155
    
    # clip = 'clip_3'
    # start = 141

    # clip = 'clip_4'
    # start = 180
    
    root_path = './data'
    source = 'kling'
    frames_adj_path = f"{root_path}/{source}/frames_adj"
    frames_renamed_path = f"{root_path}/{source}/frames_renamed"

    input_path = f"{frames_adj_path}/{clip}_part2_open_cards"
    output_path = f"{frames_renamed_path}/{clip}_part2_open_cards"
    
    rename_images(input_path, output_path, start)
