import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =====================
# 入力動画ファイルと出力ディレクトリ
# =====================
VIDEO_PATH = "/workspaces/python/hassi-/TLC00011.AVI"
SAVE_DIR = "video_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# 動画読み込み
# =====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"動画が開けません: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_to_save = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames_to_save.append((frame_idx, frame))
    frame_idx += 1

cap.release()

# =====================
# フレーム保存関数
# =====================
def save_frame(args):
    idx, frame = args
    filename = f"frame_{idx:04d}.jpg"  # JPEGで高速化
    cv2.imwrite(os.path.join(SAVE_DIR, filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return 1  # 成功したフレーム数

# =====================
# 並列保存 + tqdmで進捗バー
# =====================
processed = 0
with ThreadPoolExecutor(max_workers=8) as executor:  # CPUコアに応じて調整
    futures = [executor.submit(save_frame, f) for f in frames_to_save]
    for f in tqdm(as_completed(futures), total=len(futures), desc="フレーム保存中"):
        processed += f.result()

print(f"{processed} 枚のフレームを保存しました")
