import cv2, numpy as np
import re

# --- feather α-blend（簡潔版）：把 A 的 ROI 貼到 B 上 ---
def paste_with_feather(dst, src, rect, feather=15):
    x,y,w,h = rect
    patchA = src[y:y+h, x:x+w].copy()
    roi    = dst[y:y+h, x:x+w]

    mask = np.zeros((h, w), np.uint8)
    mask[:] = 255
    if feather > 0:
        # 邊緣羽化
        k = max(1, feather|1)
        mask = cv2.GaussianBlur(mask, (k,k), feather)

    mask3 = cv2.merge([mask]*3).astype(np.float32)/255.0
    out = (patchA.astype(np.float32)*mask3 + roi.astype(np.float32)*(1-mask3)).astype(np.uint8)
    dst[y:y+h, x:x+w] = out

def main(imgA, imgB, output_path):
    imgB_locked = imgB.copy()
    for rect in ROIS:
        paste_with_feather(imgB_locked, imgA, rect, feather=21)
    cv2.imwrite(output_path, imgB_locked)

if __name__ == "__main__":
    n = 4
    
    root_path = './data'
    img0_path = f"{root_path}/ori/clip_{n}_close_hand.png"
    img1_path = f"{root_path}/ori/clip_all_flat_hand.png"
    output_path = f"{root_path}/ori/clip_{n}_flat_hand.png"

    imgA = cv2.imread(img0_path)  # 第 1 張（基準）
    imgB = cv2.imread(img1_path)  # 第 2 張（要被覆寫三塊 ROI）
    
    assert imgA is not None and imgB is not None
    assert imgA.shape == imgB.shape, "兩張圖必須同尺寸（不然先 resize 或對位）"

    H, W = imgA.shape[:2]

    # === 這三個 ROI 你要自己填（依你標紅框的區域） ===
    # (x, y, w, h)
    ROI_LEFT_DECK    = (0,   135, 120, 240)
    ROI_CENTER_CARDS = (307, 223, 715, 210)
    ROI_RIGHT_SHOE   = (1027,180, 253, 224)
    ROIS = [ROI_LEFT_DECK, ROI_CENTER_CARDS, ROI_RIGHT_SHOE]

    main(imgA, imgB, output_path)

