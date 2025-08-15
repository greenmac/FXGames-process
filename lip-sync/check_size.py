from pathlib import Path
import re
import imageio.v3 as iio

# 資料夾：調整成你的路徑
FOLDER = Path(r".\data\image2video\ori\clip_1\part4")

def _natural_key(p: Path):
    """自然排序：1.png, 2.png, 10.png"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]

def list_image_files(folder: Path):
    # 若要支援 jpg，可改成 folder.glob("*.png") -> folder.glob("*.*")
    return sorted(folder.glob("*.png"), key=_natural_key)

def main():
    files = list_image_files(FOLDER)
    if not files:
        print(f"[空資料夾] {FOLDER} 沒有 PNG 檔")
        return

    print("=== 每張圖片的 frame.shape（imageio 讀取）===")
    shapes = {}  # {shape: [filenames]}
    all_shapes = []

    for p in files:
        try:
            arr = iio.imread(p)  # shape 可能為 (H,W)、(H,W,C)
        except Exception as e:
            print(f"[讀取失敗] {p.name}: {e}")
            shapes.setdefault(("unreadable",), []).append(p.name)
            continue

        print(f"{p.name}: shape={arr.shape}, dtype={arr.dtype}")
        shapes.setdefault(arr.shape, []).append(p.name)
        all_shapes.append(arr.shape)

    # 統計與結論
    uniq = list(shapes.keys())
    print("\n=== 統計 ===")
    for shp in uniq:
        print(f"{shp}: {len(shapes[shp])} 張")

    if len([s for s in uniq if s != ('unreadable',)]) == 1:
        print("\n✅ 所有可讀圖片的 shape 一致")
    else:
        print("\n❌ 圖片 shape 不一致（可能是通道數不同或旋轉導致）")
        # 額外把異常分組列出
        for shp, names in shapes.items():
            print(f"\n[shape={shp}]")
            for n in names[:20]:
                print("  -", n)
            if len(names) > 20:
                print(f"  ...共 {len(names)} 張")

if __name__ == "__main__":
    main()
