import os
import numpy as np
import cv2
from moviepy import ImageSequenceClip
from ccvfi import AutoModel, ConfigType, VFIBaseModel
from utils_tool import timer

# ----------------------------
# 1) 參數：輸入/輸出與 ROI 區域
# ----------------------------
ROOT = "./data"
IMG0_PATH = f"{ROOT}/ori/000000110.png"  # 第 1 張（基準：要一路維持的樣貌）
IMG1_PATH = f"{ROOT}/ori/frame_000001.png"  # 第 2 張（先被覆寫三塊 ROI，再插補）
FRAMES_DIR = f"{ROOT}/frames_out"
OUTPUT_MP4 = f"{ROOT}/output/out_4s_30fps.mp4"
FPS = 30
N_FRAMES = 4 * FPS  # 4 秒 * 30 fps = 120 幀
FEATHER = 21        # ROI 邊緣羽化半徑（可調）

# 依你的紅框填入 (x, y, w, h)
ROI_LEFT_DECK    = (0,   135, 120, 240)  # 左邊牌海
ROI_CENTER_CARDS = (307, 223, 715, 210)  # 中間四張牌區（保持四張）
ROI_RIGHT_SHOE   = (1027,180, 253, 224)  # 右邊牌靴
ROIS = [ROI_LEFT_DECK, ROI_CENTER_CARDS, ROI_RIGHT_SHOE]

# ----------------------------
# 2) 影像讀取與 ROI 鎖定工具
# ----------------------------
def imread_any(path: str):
    """Windows 路徑友善讀檔（支援中文路徑）。"""
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def paste_with_feather(dst: np.ndarray, src: np.ndarray, rect, feather: int = 15):
    """把 src 的 ROI 貼到 dst 的相同位置，邊界做高斯羽化。"""
    x, y, w, h = rect
    patchA = src[y:y+h, x:x+w].copy()
    roi    = dst[y:y+h, x:x+w]
    mask = np.full((h, w), 255, np.uint8)
    if feather and feather > 0:
        k = max(1, feather | 1)  # odd kernel
        mask = cv2.GaussianBlur(mask, (k, k), feather)
    mask3 = cv2.merge([mask]*3).astype(np.float32) / 255.0
    out = (patchA.astype(np.float32) * mask3 + roi.astype(np.float32) * (1 - mask3)).astype(np.uint8)
    dst[y:y+h, x:x+w] = out

def lock_rois_to_imgA(imgA: np.ndarray, imgB: np.ndarray, rois, feather=21) -> np.ndarray:
    """回傳把 imgB 的指定區塊改成 imgA 樣貌後的新影像。"""
    assert imgA.shape == imgB.shape, "兩張圖尺寸不同，請先對位或 resize。"
    out = imgB.copy()
    for rect in rois:
        paste_with_feather(out, imgA, rect, feather=feather)
    return out

# ----------------------------
# 3) RIFE 插補（用 ccvfi）
# ----------------------------
model: VFIBaseModel = AutoModel.from_pretrained(ConfigType.RIFE_IFNet_v426_heavy)

def mid(model: VFIBaseModel, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """t=0.5 的中間幀（RIFE）。"""
    return model.inference_image_list(img_list=[a, b])[0]

def frames_with_exact_count(model, a, b, n_frames: int):
    """
    先遞迴二分細化到 >= n_frames，再等距抽樣成剛好 n_frames（含首尾）。
    """
    seq = [a, b]
    while len(seq) < n_frames:
        nxt = []
        for i in range(len(seq) - 1):
            L, R = seq[i], seq[i + 1]
            M = mid(model, L, R)
            nxt.extend([L, M])
        nxt.append(seq[-1])
        seq = nxt
        if len(seq) == n_frames:
            break
    if len(seq) != n_frames:
        idx = np.linspace(0, len(seq) - 1, n_frames)
        idx = np.round(idx).astype(int)
        seq = [seq[i] for i in idx]
    return seq

# ----------------------------
# 4) 主流程：鎖 ROI → 插補 120 幀 → MoviePy 寫 mp4
# ----------------------------
@timer
def main():
    img0 = imread_any(IMG0_PATH)
    img1 = imread_any(IMG1_PATH)

    # 尺寸不一致就把第二張 resize 成第一張大小（也可改成先做特徵對位）
    if img0.shape != img1.shape:
        img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_AREA)

    # 先把第二張的三個區域改成第一張的樣貌（四張牌 + 深色牌海/牌靴）
    img1_locked = lock_rois_to_imgA(img0, img1, ROIS, feather=FEATHER)

    # 生成 120 幀（含首尾）
    frames = frames_with_exact_count(model, img0, img1_locked, N_FRAMES)

    # 寫出影像序列（可省略；但便於除錯）
    os.makedirs(FRAMES_DIR, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(FRAMES_DIR, f"out_{i:04d}.png"), f)

    # 用 MoviePy 組成 4 秒 mp4（libx264 + yuv420p 相容性好）
    os.makedirs(os.path.dirname(OUTPUT_MP4), exist_ok=True)
    seq_files = [os.path.join(FRAMES_DIR, f"out_{i:04d}.png") for i in range(N_FRAMES)]
    clip = ImageSequenceClip(seq_files, fps=FPS)
    clip.write_videofile(OUTPUT_MP4, codec="libx264", fps=FPS, audio=False)

    print("Video exported successfully:", OUTPUT_MP4)

if __name__ == "__main__":
    main()
