import cv2
import os
import numpy as np

root_path = './data'
clip_folder = 'clip_4'
kling_ori_path = f"{root_path}/kling/ori/{clip_folder}"
kling_adj_path = f"{root_path}/kling/adj/{clip_folder}"
os.makedirs(kling_ori_path, exist_ok=True)
os.makedirs(kling_adj_path, exist_ok=True)

video_path = f"{root_path}/kling/video_ori/kling_{clip_folder}_close_flat_hands.mp4"
overlay_image_path = f"./data/ori/{clip_folder}_close_hand.png"  # 要補的圖片
scale_y = 1.0 / 1.023  # 縮小比例

cap = cv2.VideoCapture(video_path)
frame_count = 0

def resize_and_pad_to_xxx(image):
    target_w, target_h = 1280, 720
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)  # 黑底
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return result

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame is not None:
        padded_frame = resize_and_pad_to_xxx(frame)
        frame_path = os.path.join(kling_ori_path, f'{frame_count:05d}.png')
        cv2.imwrite(frame_path, padded_frame)
        frame_count += 1

cap.release()
print(f'✅ 擷取完成，共儲存到第 {frame_count - 1:05d}.png，儲存在 {kling_ori_path}/')


# 讀取 overlay 圖片（作為補圖）
overlay_img = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
if overlay_img is None:
    print(f"❌ 無法讀取補圖：{overlay_image_path}")
    exit()
if overlay_img.shape[2] == 3:
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)

# 遍歷資料夾中的所有圖片
for filename in os.listdir(kling_ori_path):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(kling_ori_path, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(kling_adj_path, output_filename)

        img = cv2.imread(input_path)
        if img is None:
            print(f"❌ 無法讀取圖片：{input_path}")
            continue

        orig_h, orig_w = img.shape[:2]
        new_h = int(orig_h * scale_y)

        # 縮小圖像
        resized = cv2.resize(img, (orig_w, new_h))

        # 建立透明背景
        output_img = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)

        # 貼上縮小圖像到中間
        start_y = (orig_h - new_h) // 2
        resized_bgra = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
        output_img[start_y:start_y + new_h, :] = resized_bgra

        # 調整 overlay 圖到相同大小
        overlay_resized = cv2.resize(overlay_img, (orig_w, orig_h))

        # 把透明處補上 overlay 圖內容
        alpha_channel = output_img[:, :, 3]
        transparent_mask = (alpha_channel == 0)

        # 將透明處用 overlay 補上
        output_img[transparent_mask] = overlay_resized[transparent_mask]

        # 儲存結果
        cv2.imwrite(output_path, output_img)
        print(f"✅ 處理完成：{output_filename}")

print("🎉 所有圖片處理與補圖完成。")
