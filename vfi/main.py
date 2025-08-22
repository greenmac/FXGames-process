import os
import math
import numpy as np
import cv2
from moviepy import ImageSequenceClip
from ccvfi import AutoModel, ConfigType, VFIBaseModel

# =========================
# Config
# =========================
ROOT = "./data"
IMG0_PATH = f"{ROOT}/ori/000000110.png"    # 基準影像（要維持樣貌）
IMG1_PATH = f"{ROOT}/ori/frame_000001.png" # 另一端影像（會先對齊+鎖ROI）
FRAMES_DIR = f"{ROOT}/frames_out"
OUTPUT_MP4 = f"{ROOT}/output/out_4s_30fps.mp4"

FPS = 30
TARGET_FRAMES = 4 * FPS        # 4 秒 -> 120 幀（維持）
FEATHER = 21                   # ROI 羽化半徑
USE_TEMPORAL_MEDIAN = False    # 如仍有淡淡殘影，可改 True 試試 3 幀時域中值

# 三個 ROI：左牌海 / 中間四張牌 / 右牌靴（x,y,w,h）
ROI_LEFT_DECK    = (0,   135, 120, 240)
ROI_CENTER_CARDS = (307, 223, 715, 210)
ROI_RIGHT_SHOE   = (1027,180, 253, 224)
ROIS = [ROI_LEFT_DECK, ROI_CENTER_CARDS, ROI_RIGHT_SHOE]

# =========================
# Utils
# =========================
def imread_any(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def paste_with_feather(dst: np.ndarray, src: np.ndarray, rect, feather: int = 15):
    x, y, w, h = rect
    patch = src[y:y+h, x:x+w].copy()
    roi   = dst[y:y+h, x:x+w]
    mask = np.full((h, w), 255, np.uint8)
    if feather and feather > 0:
        k = max(1, feather | 1)
        mask = cv2.GaussianBlur(mask, (k, k), feather)
    mask3 = cv2.merge([mask]*3).astype(np.float32) / 255.0
    out = (patch.astype(np.float32)*mask3 + roi.astype(np.float32)*(1-mask3)).astype(np.uint8)
    dst[y:y+h, x:x+w] = out

def lock_rois_to_imgA(imgA: np.ndarray, imgB: np.ndarray, rois, feather=21) -> np.ndarray:
    out = imgB.copy()
    for rect in rois:
        paste_with_feather(out, imgA, rect, feather=feather)
    return out

def align_img2_to_img1(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    ORB + RANSAC Homography，把 img2 對齊到 img1，減少大位移（避免鬼影/重影）。
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(gray1, None)
    k2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None:
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in matches if m.distance < 0.7*n.distance]
    if len(good) < 8:
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

# =========================
# RIFE 插補（ccvfi）
# =========================
model: VFIBaseModel = AutoModel.from_pretrained(ConfigType.RIFE_IFNet_v426_heavy)

def rife_mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # t=0.5 中間幀；RIFE 的理論支援任意時刻插補，這裡用二分構出「2^d 網格」。:contentReference[oaicite:3]{index=3}
    return model.inference_image_list(img_list=[a, b])[0]

def densify_to_depth(a: np.ndarray, b: np.ndarray, depth: int) -> list[np.ndarray]:
    """
    生成長度 2^depth + 1 的序列（dyadic grid）。
    depth=7 -> 129 幀，足以再抽為 120 幀。
    """
    seq = [a, b]
    for _ in range(depth):
        nxt = []
        for i in range(len(seq) - 1):
            L, R = seq[i], seq[i + 1]
            M = rife_mid(L, R)
            nxt.extend([L, M])
        nxt.append(seq[-1])
        seq = nxt
    return seq  # len = 2^depth + 1

def ease_out_cubic(t: np.ndarray) -> np.ndarray:
    # t ∈ [0,1]，越後面越慢（快進慢出）
    return 1 - (1 - t) ** 3

def resample_with_ease(frames, n_out: int, head_hold=10, tail_hold=10, ease_fn=ease_out_cubic):
    """
    先在開頭/結尾保持（hold）幾幀，核心區段用 easing 曲線重分配時間。
    - frames: 幀池（例如 dyadic 129 幀）
    - n_out:  輸出幀數（你的目標 120）
    - head_hold/tail_hold: 端點各保留的靜止幀數（可微調 5~15）
    """
    core = max(n_out - head_hold - tail_hold, 1)
    t = np.linspace(0, 1, core)
    te = ease_fn(t)  # 非線性時間分佈（easing）
    idx = np.round(te * (len(frames) - 1)).astype(int)

    seq = [frames[0]] * head_hold + [frames[i] for i in idx] + [frames[-1]] * tail_hold
    # 長度校正
    if len(seq) != n_out:
        seq = seq[:n_out] if len(seq) > n_out else seq + [frames[-1]] * (n_out - len(seq))
    return seq


def temporal_median_3(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    輕量「去鬼影」：三幀時域中值（保邊、不易出現拖影）。如仍有殘影可開。:contentReference[oaicite:4]{index=4}
    """
    out = frames.copy()
    for i in range(1, len(frames)-1):
        stack = np.stack([frames[i-1], frames[i], frames[i+1]], axis=0)
        out[i] = np.median(stack, axis=0).astype(np.uint8)
    return out

# =========================
# Main
# =========================
def main():
    os.makedirs(os.path.dirname(OUTPUT_MP4), exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    img0 = imread_any(IMG0_PATH)
    img1 = imread_any(IMG1_PATH)

    # 尺寸統一
    if img0.shape != img1.shape:
        img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_AREA)

    # 1) 對齊 + 2) 鎖 ROI（四張牌＋兩側深色，降低遮擋/內容不一致帶來的鬼影）:contentReference[oaicite:5]{index=5}
    img1_aligned = align_img2_to_img1(img0, img1)
    img1_locked  = lock_rois_to_imgA(img0, img1_aligned, ROIS, feather=FEATHER)

    # 3) 只建「最小充分」的 2^d 網格：找到 d 使 2^d + 1 >= 120 → d=7（129 幀）
    needed = TARGET_FRAMES
    depth = math.ceil(math.log2(needed - 1))   # 119 -> 7
    pool  = densify_to_depth(img0, img1_locked, depth)  # 129 幀（比你之前的 960 快很多）

    # 4) 等距抽到 120 幀（4 秒 @ 30fps）
    frames = resample_with_ease(pool, TARGET_FRAMES, head_hold=10, tail_hold=10)


    # 5) 可選：溫和去鬼影（預設關）
    if USE_TEMPORAL_MEDIAN:
        frames = temporal_median_3(frames)

    # 6) 直接用 MoviePy 封裝（免落地 PNG）
    rgb_seq = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    clip = ImageSequenceClip(rgb_seq, fps=FPS)
    clip.write_videofile(OUTPUT_MP4, codec="libx264", fps=FPS, audio=False)

    print("Exported:", OUTPUT_MP4)

if __name__ == "__main__":
    main()
