# main.py
import gc, subprocess, shutil
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# YOLO: 用來抓第1幀的人框（初始化 SAM2）
from ultralytics import YOLO

# SAM2: 直接用 HF 權重，省去手動 config/ckpt
from sam2.sam2_video_predictor import SAM2VideoPredictor

# BiRefNet: HF AutoModel
from transformers import AutoModelForImageSegmentation
from utils_tool import timer
import math
import torch.nn.functional as F
# ------------------------ Utils ------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def mask_to_uint8(m: np.ndarray) -> np.ndarray:
    return (np.clip(m, 0, 1) * 255).astype(np.uint8)

def read_first_frame(video_path: str) -> np.ndarray:
    """讀取影片首幀（RGB），供 YOLO 初始人框."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV 無法開啟影片：{video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("讀取首幀失敗")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def auto_person_box(frame_rgb: np.ndarray, conf=0.25) -> Tuple[int,int,int,int]:
    """用 YOLOv11 自動抓 person 方框（避免包含桌面：略微上縮底邊）."""
    h, w = frame_rgb.shape[:2]
    yolo = YOLO("yolo11n.pt")  # 未下載則自動抓
    res = yolo.predict(source=frame_rgb, verbose=False, conf=conf)
    best = None; best_area = 0
    for r in res:
        if not hasattr(r, "boxes"): continue
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(cls) != 0:  # 0 = person
                continue
            x1, y1, x2, y2 = box.astype(int)
            y2 = int(y2 - 0.08 * h)  # 底邊上縮，避免桌面
            x1 = max(0, x1); y1=max(0, y1); x2=min(w-1, x2); y2=min(h-1, y2)
            area = max(0, x2-x1) * max(0, y2-y1)
            if area > best_area:
                best_area = area; best = (x1, y1, x2, y2)
    if best is None:
        # 退而求其次：畫面中央區域
        cx, cy = w//2, int(h*0.45)
        ww, hh = int(w*0.5), int(h*0.7)
        best = (max(0,cx-ww//2), max(0,cy-hh//2), min(w-1,cx+ww//2), min(h-1,cy+hh//2))
    return best

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def extract_frames(video_path: str, frames_dir: Path, quality: int = 2):
    """
    修法 A：把 MP4 轉為幀資料夾。
    1) 優先用 ffmpeg（速度與相容性最佳）
    2) 若沒 ffmpeg，退回 OpenCV 逐幀寫檔
    """
    ensure_dir(frames_dir)
    # 若已存在則略過
    existing = list(frames_dir.glob("*.jpg"))
    if len(existing) > 0:
        return len(existing)

    if has_ffmpeg():
        # -q:v 2 ≈ 高品質JPEG；-vsync 0 保留原始幀率
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-q:v", str(quality), "-vsync", "0",
            str(frames_dir / "%06d.jpg")
        ]
        subprocess.run(cmd, check=True)
    else:
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok: break
            cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            idx += 1
        cap.release()

    return len(list(frames_dir.glob("*.jpg")))

# ------------------------ SAM2：追蹤人像（以幀資料夾） ------------------------

def sam2_track_masks(
    frames_dir: str,
    init_box_xyxy: Tuple[int,int,int,int],
    device: str = "cuda",
    chunk_size: int = 300,
    prob_thresh: float = 0.5
) -> Dict[int, np.ndarray]:
    """
    輸入：幀資料夾路徑（%06d.jpg）
    回傳：{frame_idx: binary_mask(H,W)}
    策略：async 載幀 + 分段 propagate，避免 RAM 爆掉。
    """
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")  # HF 自動載權重、配置 :contentReference[oaicite:4]{index=4}

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda)

    out: Dict[int, np.ndarray] = {}
    frames_dir = str(frames_dir)

    with torch.inference_mode(), amp_ctx:
        state = predictor.init_state(
            frames_dir,
            async_loading_frames=True,   # 逐幀載入
            offload_state_to_cpu=True    # 大型狀態丟 CPU，省 VRAM
        )

        obj_id = 1
        _f0, _obj_ids, _mask_logits = predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=obj_id,
            box=np.array(init_box_xyxy, dtype=np.float32)
        )

        total_frames = len(list(Path(frames_dir).glob("*.jpg"))) or 0

        start = 0
        while start < total_frames or total_frames == 0:
            max_num = None if total_frames == 0 else min(chunk_size, total_frames - start)
            gen = predictor.propagate_in_video(
                state,
                start_frame_idx=None if total_frames == 0 else start,
                max_frame_num_to_track=max_num,
                reverse=False,
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in gen:
                if obj_id not in out_obj_ids:
                    continue
                k = list(out_obj_ids).index(obj_id)
                logits_k = out_mask_logits[k]
                if isinstance(logits_k, (list, tuple)):
                    logits_k = logits_k[0]
                if hasattr(logits_k, "detach"):
                    logits_np = torch.sigmoid(torch.as_tensor(logits_k)).float().cpu().numpy()
                else:
                    logits_np = 1 / (1 + np.exp(-logits_k))
                if logits_np.ndim == 3 and logits_np.shape[0] == 1:
                    logits_np = logits_np[0]
                m = (logits_np > prob_thresh).astype(np.uint8)
                out[int(out_frame_idx)] = m

            if total_frames == 0:
                break
            start += max_num if max_num is not None else chunk_size

    return out

# ------------------------ RVM：產生 base alpha（時間穩定） ------------------------

def rvm_base_alpha(video_path: str, device="cuda", downsample_ratio=0.25) -> Tuple[List[np.ndarray], int, int, float]:
    """
    用 Torch Hub 載入 RVM（mobilenetv3），逐幀估 alpha。
    參考官方建議：torch.hub.load(..., "mobilenetv3")。:contentReference[oaicite:5]{index=5}
    回傳 (alpha_list, W, H, fps)
    """
    # 載入一次即可（也可搬到全域）
    rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").eval().to(device)  # :contentReference[oaicite:6]{index=6}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rec = [None]*4
    alphas: List[np.ndarray] = []
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(frame_rgb).float()/255.0   # HxWx3
            src = src.permute(2,0,1).unsqueeze(0).to(device)  # 1x3xHxW
            fgr, pha, *rec = rvm(src, *rec, downsample_ratio) # RNN 有記憶，時序穩定
            a = pha[0,0].clamp(0,1).detach().cpu().numpy()
            alphas.append(a)
    cap.release()
    return alphas, W, H, fps

# ------------------------ BiRefNet：邊緣精修（手指） ------------------------

def load_birefnet(device="cuda", hf_repo="ZhengPeng7/BiRefNet_dynamic-matting"):
    net = AutoModelForImageSegmentation.from_pretrained(
        hf_repo, trust_remote_code=True
    ).to(device).eval()
    try:
        net.half()
    except Exception:
        pass
    return net

def _pad_to_32(img_np: np.ndarray):
    H, W = img_np.shape[:2]
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32
    if pad_h or pad_w:
        img_np = cv2.copyMakeBorder(
            img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )
    return img_np, (0, pad_w, 0, pad_h)  # (pad_left, pad_right, pad_top, pad_bottom)

def refine_with_birefnet(net, frame_rgb: np.ndarray, alpha0: np.ndarray, band_px: int=8, device="cuda") -> np.ndarray:
    H, W = alpha0.shape

    # 邊緣帶
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px, band_px))
    dil = cv2.dilate(alpha0, k); ero = cv2.erode(alpha0, k)
    edge_band = cv2.dilate(((dil - ero) > 0.01).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px+2, band_px+2)))

    # 整幀 padding 到 32 的倍數
    frame_pad, (pl, pr, pt, pb) = _pad_to_32(frame_rgb)

    # 打 tensor
    inp = torch.from_numpy(frame_pad).to(device)
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
    inp = inp.permute(0,3,1,2).contiguous().float() / 255.0
    if next(net.parameters()).dtype == torch.float16:
        inp = inp.half()

    with torch.no_grad():
        out = net(inp)

        # 取 logits（相容 list / dict）
        if isinstance(out, dict):
            ten = out.get("logits", out.get("alpha", out.get("pred", out.get("pred_masks", None))))
        else:
            ten = out
        if isinstance(ten, (list, tuple)):
            ten = ten[-1]
        if ten.ndim == 4 and ten.shape[1] == 1:
            ten = ten[:, 0]

        matte = torch.sigmoid(ten).float()
        matte = F.interpolate(
            matte.unsqueeze(1),
            size=(frame_pad.shape[0], frame_pad.shape[1]),
            mode="bilinear", align_corners=False
        )[:, 0]
        matte_np = matte[0].clamp(0, 1).detach().cpu().numpy()

    # 裁回原始大小
    if pb or pr:
        matte_np = matte_np[pt:pt+H, pl:pl+W]

    # 只覆寫邊緣帶
    alpha_ref = alpha0.copy()
    band = edge_band.astype(bool)
    alpha_ref[band] = matte_np[band]

    # 輕微平滑
    alpha_ref = cv2.GaussianBlur(alpha_ref, (0,0), sigmaX=0.8)
    return np.clip(alpha_ref, 0, 1)
# ------------------------ 主流程 ------------------------
@timer
def run(
    video_path: str,
    out_dir: str,
    biref_hf_repo="ZhengPeng7/BiRefNet-matting",
    manual_box: Tuple[int,int,int,int] = None,
    device="cuda",
):
    torch.backends.cudnn.benchmark = True
    out_dir = Path(out_dir)
    frames_dir = out_dir / "sam2_frames"
    rgba_dir   = out_dir / "rgba_frames"
    ensure_dir(out_dir); ensure_dir(frames_dir); ensure_dir(rgba_dir)

    # 0) 先拆幀（修法 A）
    n_frames = extract_frames(video_path, frames_dir)
    if n_frames == 0:
        raise RuntimeError("拆幀失敗或沒有任何幀輸出")

    # 1) 取第1幀 → YOLO 抓人框（用原影片的首幀）
    first_rgb = read_first_frame(video_path)
    box = manual_box or auto_person_box(first_rgb)

    # 2) SAM2：用幀資料夾進行追蹤，得到 per-frame person mask（二值）
    sam2_masks = sam2_track_masks(str(frames_dir), box, device=device)

    # 3) RVM：對原影片逐幀估 alpha（時序穩定）  :contentReference[oaicite:8]{index=8}
    alphas, W, H, fps = rvm_base_alpha(video_path, device=device)
    if len(alphas) == 0:
        raise RuntimeError("RVM 沒有產生 alpha")

    # 4) BiRefNet：邊緣精修（重點在手指）
    biref = load_birefnet(device=device, hf_repo=biref_hf_repo)  # :contentReference[oaicite:9]{index=9}

    # 5) 合成 RGBA 序列
    cap = cv2.VideoCapture(video_path)
    idx = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(alphas), desc="Compositing & edge refining"):
        while True:
            ok, frame_bgr = cap.read()
            if not ok or idx >= len(alphas): break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            m_person = sam2_masks.get(idx, None)
            if m_person is None:
                m_person = sam2_masks.get(idx-1, np.ones_like(alphas[idx], dtype=np.uint8))
            m_person = m_person.astype(np.float32)

            a0 = np.clip(alphas[idx], 0, 1) * m_person
            a_ref = refine_with_birefnet(biref, frame_rgb, a0, band_px=8, device=device)

            rgba = np.dstack([frame_rgb, mask_to_uint8(a_ref)])
            Image.fromarray(rgba, mode="RGBA").save(rgba_dir / f"{idx:06d}.png")

            idx += 1
            tqdm.write(f"frame {idx} done") if idx % 100 == 0 else None
            gc.collect()
    cap.release()

    # 6) 預覽 MP4（黑底）
    out_mp4 = str(out_dir / "preview_black_bg.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (W, H))
    for i in range(idx):
        rgba = np.array(Image.open(rgba_dir / f"{i:06d}.png"))
        rgb = rgba[...,:3]; a = (rgba[...,3:4] / 255.0)
        comp = (rgb * a + np.zeros_like(rgb) * (1 - a)).astype(np.uint8)
        vw.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    vw.release()

    print(f"[DONE] 幀資料夾：{frames_dir}")
    print(f"[DONE] RGBA PNG 序列：{rgba_dir}")
    print(f"[DONE] 預覽 MP4（黑底）：{out_mp4}")

# ------------------------ CLI ------------------------

if __name__ == "__main__":
    video_path = "data/ori/clip_60_Round.mp4"
    out_dir = "data/output"
    biref_model = "ZhengPeng7/BiRefNet_dynamic-matting" # or "ZhengPeng7/BiRefNet-matting"
    device = "cuda"  # 或 cpu
    
    # 若要指定手動方框：
    # manual_box = (50, 100, 780, 350)
    manual_box = None

    run(video_path, out_dir, biref_model, manual_box, device)

