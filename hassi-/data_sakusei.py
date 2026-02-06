import cv2
import numpy as np
import os

# =====================
# 定数
# =====================
WIDTH, HEIGHT = 1000, 250
DST_PTS = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype=np.float32)

# 左右ワープ座標
LEFT_PTS = np.array([[275*1.875, 435*1.875], [280*1.875, 160*1.875], 
                     [405*1.875, 180*1.875], [395*1.875, 455*1.875]], dtype=np.float32)
RIGHT_PTS = np.array([[878*1.875, 190*1.875], [818*1.875, 440*1.875], 
                      [717*1.875, 400*1.875], [772*1.875, 160*1.875]], dtype=np.float32)

# 数字領域（ワープ後画像上の4点）
digit_regions_left = [
    ("1", [(165, 0), (305, 0), (290, 250), (135, 250)]),
    ("8", [(320, 0), (475, 0), (460, 250), (295, 250)]),
    ("0", [(480, 0), (650, 0), (635, 250), (470, 250)]),
    ("3", [(650, 0), (800, 0), (790, 250), (645, 250)]),
    ("5", [(835, 0), (975, 0), (975, 250), (830, 250)])
]

digit_regions_right = [
    ("1", [(0, 0), (190, 0), (190, 250), (0, 250)]),
    ("6", [(190, 0), (375, 0), (375, 250), (190, 250)]),
    ("2", [(375, 0), (565, 0), (565, 250), (375, 250)]),
    ("9", [(565, 0), (740, 0), (740, 250), (565, 250)]),
    ("0", [(750, 0), (960, 0), (990, 250), (775, 250)])
]

# =====================
# 前処理関数
# =====================
def adjust_contrast_brightness(img, alpha=1.2, beta=20):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def normalize_image_safe(img):
    img_float = img.astype(np.float32)
    if len(img.shape) == 3:
        out = np.zeros_like(img_float)
        for c in range(3):
            min_val, max_val = img_float[:, :, c].min(), img_float[:, :, c].max()
            if max_val - min_val > 1e-5:
                out[:, :, c] = (img_float[:, :, c] - min_val) * 255 / (max_val - min_val)
            else:
                out[:, :, c] = img_float[:, :, c]
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        min_val, max_val = img_float.min(), img_float.max()
        if max_val - min_val > 1e-5:
            out = (img_float - min_val) * 255 / (max_val - min_val)
        else:
            out = img_float
        return np.clip(out, 0, 255).astype(np.uint8)

def preprocess_image(img, alpha=1.5, beta=15):
    img = adjust_contrast_brightness(img, alpha=alpha, beta=beta)
    img = normalize_image_safe(img)
    return img

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def gamma_correction(img, gamma=1.3):
    inv = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


# =====================
# 画像切り出し・ワープ関数
# =====================
def warp_image(img, src_pts, width=WIDTH, height=HEIGHT):
    M = cv2.getPerspectiveTransform(src_pts, DST_PTS)
    return cv2.warpPerspective(img, M, (width, height))

def crop_by_4points(img, pts):
    pts = np.array(pts, dtype=np.float32)
    width = int(max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])))
    height = int(max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])))
    dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (width, height))

def crop_and_save_digits(img, save_dir, digit_regions, side_label, base_name):
    os.makedirs(save_dir, exist_ok=True)
    for idx, (digit, pts) in enumerate(digit_regions, start=1):
        cropped = crop_by_4points(img, pts)
        filename = f"{base_name}_{side_label}_{idx}_{digit}.png"
        cv2.imwrite(os.path.join(save_dir, filename), cropped)

# =====================
# メイン処理
# =====================
def process_image(img_path, save_dir):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {img_path}")

    # 左右ワープ
    left_warped = warp_image(img, LEFT_PTS)
    right_warped = warp_image(img, RIGHT_PTS)

    # 前処理
    #left_warped  = apply_clahe(left_warped)
    #right_warped = apply_clahe(right_warped)

    #left_warped  = gamma_correction(left_warped, gamma=1.2)
    #right_warped = gamma_correction(right_warped, gamma=1.2)

    # save_dir の一つ上のディレクトリ
    parent_dir = os.path.dirname(save_dir)
    # ワープ画像保存
    cv2.imwrite(os.path.join(parent_dir, f"left_warped.png"), left_warped)
    cv2.imwrite(os.path.join(parent_dir, f"right_warped.png"), right_warped)

    # 数字切り出し保存
    crop_and_save_digits(left_warped, save_dir, digit_regions_left, "left", base_name)
    crop_and_save_digits(right_warped, save_dir, digit_regions_right, "right", base_name)

    print("左右ワープ＆数字切り出し処理が完了しました")
    print(f"元画像: {img_path}")

# =====================
# 実行例
# =====================
if __name__ == "__main__":
    IMG_PATH = "/workspaces/python/video_frames/frame_0080.jpg"
    SAVE_DIR = "/workspaces/python/hassi-/digits"
    process_image(IMG_PATH, SAVE_DIR)