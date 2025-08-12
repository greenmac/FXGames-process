from ultralytics import YOLO
import cv2, numpy as np
import mediapipe as mp

model = YOLO("yolo11s-seg.pt") # COCO model pre-training, including the person category

def get_background_median(video_path, sample_size=200, max_frames=5000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"The video cannot be opened: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"The total number of frames in the video is {frame_count}")

    sample_n = min(frame_count, sample_size)
    try:
        ids = np.random.choice(frame_count, size=sample_n, replace=False)
    except ValueError:
        ids = np.random.choice(frame_count, size=sample_size, replace=True)

    frames = []
    for fid in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()

    if not frames:
        raise ValueError("Unable to read any reference images")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

from ultralytics import YOLO
import cv2, numpy as np

# Load YOLOv11 instance segmentation model, COCO pre-training includes person category
model = YOLO("yolo11s-seg.pt")

def get_person_mask_yolo_seg(frame, conf=0.3):
    results = model.predict(frame, conf=conf, classes=[0])
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for r in results:
        for m, cls in zip(r.masks.data, r.boxes.cls):
            if int(cls) == 0:
                npm = m.cpu().numpy()  # shape e.g. (N, H_model, W_model)
                resized = cv2.resize(npm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask |= (resized > 0.5).astype(np.uint8)
    return mask

def process_frame(frame, bg_ref, person_mask, dark_thresh=40, dilate_size=5, inpaint_radius=3, alpha=0.9):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < dark_thresh).astype(np.uint8)
    fix = cv2.bitwise_and(dark_mask, 1 - person_mask)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    fix_mask = cv2.dilate(fix, K, iterations=1) * 255

    inp = cv2.inpaint(frame, fix_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)

    clean = inp.astype(np.float32)
    bg_only = (fix_mask == 0) & (person_mask == 0)
    for c in range(3):
        diff = bg_ref[..., c].astype(np.float32) - inp[..., c].astype(np.float32)
        clean[..., c][bg_only] = inp[..., c][bg_only] + alpha * diff[bg_only]

    clean = np.clip(clean, 0, 255).astype(np.uint8)
    clean[person_mask == 1] = frame[person_mask == 1]
    return clean

def process_video(video_path, output_path, sample_bg_frames=200):
    bg_ref = get_background_median(video_path, sample_size=sample_bg_frames)
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        person_mask = get_person_mask_yolo_seg(frame)
        cleaned = process_frame(frame, bg_ref, person_mask)
        out.write(cleaned)

    cap.release()
    out.release()



if __name__ == "__main__":
    input_path = './data/1_part2_swap_2.mp4'
    output_path = './data/output_clean.mp4'
    process_video(input_path, output_path)
