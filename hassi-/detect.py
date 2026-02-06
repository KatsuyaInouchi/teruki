import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms

# =====================
# 設定
# =====================
SAVE_DIR = "recognized_digits"
os.makedirs(SAVE_DIR, exist_ok=True)

WIDTH, HEIGHT = 1000, 250
DST_PTS = np.array([[0,0],[WIDTH,0],[WIDTH,HEIGHT],[0,HEIGHT]], dtype=np.float32)

# 左右ワープ座標
LEFT_PTS = np.array([[275*1.875, 435*1.875], [280*1.875, 160*1.875], 
                     [405*1.875, 180*1.875], [395*1.875, 455*1.875]], dtype=np.float32)
RIGHT_PTS = np.array([[878*1.875, 190*1.875], [818*1.875, 440*1.875], 
                      [717*1.875, 400*1.875], [772*1.875, 160*1.875]], dtype=np.float32)

# 数字領域（ワープ後画像上の4点）
digit_regions_left = [
    ("1", [(165,0),(305,0),(290,250),(135,250)]),
    ("8", [(320,0),(475,0),(460,250),(295,250)]),
    ("2", [(480,0),(650,0),(635,250),(470,250)]),
    ("1", [(650,0),(800,0),(790,250),(645,250)]),
    ("0", [(835,0),(975,0),(975,250),(830,250)])
]

digit_regions_right = [
    ("1", [(0,0),(190,0),(190,250),(0,250)]),
    ("6", [(190,0),(375,0),(375,250),(190,250)]),
    ("4", [(375,0),(565,0),(565,250),(375,250)]),
    ("4", [(565,0),(740,0),(740,250),(565,250)]),
    ("9", [(750,0),(960,0),(990,250),(775,250)])
]

# =====================
# モデル準備（学習済みResNet18）
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(torch.load("/workspaces/python/hassi-/digit_classifier.pth", map_location=device))
model.eval()

# 入力前処理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# =====================
# ユーティリティ関数
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

def recognize_digit(img):
    """ResNet18で数字認識"""
    # BGR → RGB
    if len(img.shape)==3 and img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # 前処理
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred = torch.argmax(outputs, dim=1).item()
    return pred

def recognize_digits_from_warped(img, digit_regions, side="left"):
    """切り出し → 認識 → 文字列結合"""
    digits_str = ""
    for idx, (_, pts) in enumerate(digit_regions):
        cropped = crop_by_4points(img, pts)

        # 保存
        save_dir = os.path.join(SAVE_DIR, side)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir, f"{side}_digit{idx}.png"
        )
        cv2.imwrite(save_path, cropped)

        pred = recognize_digit(cropped)
        digits_str += str(pred)
    return digits_str

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
# フルパイプライン
# =====================
def full_pipeline(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {img_path}")

    # 左右ワープ
    left_warped  = warp_image(img, LEFT_PTS)
    right_warped = warp_image(img, RIGHT_PTS)

    # ワープ後
    #left_warped  = apply_clahe(left_warped)
    #right_warped = apply_clahe(right_warped)

    # 明るすぎる場合だけ
    #left_warped  = gamma_correction(left_warped, gamma=1.2)
    #right_warped = gamma_correction(right_warped, gamma=1.2)

    left_save_dir = os.path.join(SAVE_DIR, "left")
    right_save_dir = os.path.join(SAVE_DIR, "right")

    cv2.imwrite(os.path.join(left_save_dir, "left_warped.png"), left_warped)
    cv2.imwrite(os.path.join(right_save_dir, "right_warped.png"), right_warped)

    # 数字認識
    left_digits  = recognize_digits_from_warped(left_warped, digit_regions_left, side="left")
    right_digits = recognize_digits_from_warped(right_warped, digit_regions_right, side="right")

    return left_digits, right_digits

# =====================
# 実行例
# =====================
if __name__ == "__main__":
    IMG_PATH = "/workspaces/python/video_frames/frame_0076.jpg"
    #150    200 73  76
        
    left_str, right_str = full_pipeline(IMG_PATH)
    print(f"左側数字: {left_str}")
    print(f"右側数字: {right_str}")
