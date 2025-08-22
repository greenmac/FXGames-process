import os
import numpy as np
import cv2
from moviepy import ImageSequenceClip
from ccvfi import AutoModel, ConfigType, VFIBaseModel
from utils_tool import timer


model: VFIBaseModel = AutoModel.from_pretrained(ConfigType.RIFE_IFNet_v426_heavy)

# 生成恰好 120 幀方法（同之前）
def mid(model: VFIBaseModel, a, b):
    return model.inference_image_list(img_list=[a, b])[0]

def frames_with_exact_count(model, a, b, n_frames):
    seq = [a, b]
    while len(seq) < n_frames:
        nxt = []
        for i in range(len(seq)-1):
            nxt.append(seq[i])
            nxt.append(mid(model, seq[i], seq[i+1]))
        nxt.append(seq[-1])
        seq = nxt
        if len(seq) == n_frames:
            break
    if len(seq) != n_frames:
        idx = np.linspace(0, len(seq)-1, n_frames)
        idx = np.round(idx).astype(int)
        seq = [seq[i] for i in idx]
    return seq

@timer
def main(root_path, img0_path, img1_path):
    img0 = cv2.imdecode(np.fromfile(img0_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    frames = frames_with_exact_count(model, img0, img1, 120)

    os.makedirs(f"{root_path}/frames_out", exist_ok=True)
    for i, f in enumerate(frames):
        path = os.path.join(f"{root_path}/frames_out", f"out_{i:04d}.png")
        cv2.imwrite(path, f)

    frame_files = [os.path.join(f"{root_path}/frames_out", f"out_{i:04d}.png") for i in range(120)]
    output_path = f"{root_path}/output/out_4s_30fps.mp4"
    clip = ImageSequenceClip(frame_files, fps=30)
    clip.write_videofile(output_path, codec="libx264", fps=30)

    print("Video exported successfully:", output_path)

if __name__ == "__main__":
    root_path = './data'
    img0_path = f"{root_path}/ori/000000110.png"
    img1_path = f"{root_path}/ori/frame_000001.png"

    main(root_path, img0_path, img1_path)