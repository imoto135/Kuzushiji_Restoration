import os
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# 設定: ディレクトリパス
# ==========================================
# 1. 正解の文字画像 (Clean Image)
DIR_GT = "datasets/hiragana_dataset/gt/test"
# 2. 入力画像 (Damaged Image)
DIR_LQ = "datasets/hiragana_dataset/lq/test"
# 3. 正解マスク (Ground Truth Mask)
DIR_GT_MASK = "datasets/hiragana_dataset/gt_mask/test"
# 4. 予測マスク (Predicted Mask)
DIR_PRED = "outputs/lastmask_unet++/test"

# 出力先
OUTPUT_DIR = "outputs/maskcompare"
# 画像サイズ (統一してリサイズ)
TARGET_SIZE = (256, 256)

# ==========================================
# 選定された画像リスト (ID, カテゴリ, 文字)
# ==========================================
TARGET_CASES = [
    # (File_ID, Category, Suffix)
    ("U+3051_200021071_00095_1_X0528_Y2149", "Missing", "_Missing"),
    ("U+3066_hnsd003_002_X0103_Y1071", "Scratch", "_Scratch"),
    ("U+304D_200020019_00032_1_X0025_Y1890", "Stain", "_Stain"),
    ("U+3088_200003076_00138_1_X1273_Y0989", "Ghosting", "_Ghosting"),
    ("U+3044_brsk001_011_X0282_Y3237", "Transparent (Failure)", "_Transparent_Stain")
]

def find_image(directory, filename_stem):
    """
    指定ディレクトリから jpg/png を探して読み込む
    """
    path = Path(directory) / f"{filename_stem}.jpg"
    if not path.exists():
        path = Path(directory) / f"{filename_stem}.png"
    
    if path.exists():
        return cv2.imread(str(path))
    return None

def main():
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(TARGET_CASES)} cases...")

    for file_id, category, suffix in TARGET_CASES:
        # --- 1. 画像の検索と読み込み ---
        
        # GT (文字): IDのみ
        img_gt = find_image(DIR_GT, file_id)
        
        # GT Mask: IDのみ
        img_gt_mask = find_image(DIR_GT_MASK, file_id)
        
        # LQ (入力): ID + Suffix
        lq_name = f"{file_id}{suffix}"
        img_lq = find_image(DIR_LQ, lq_name)
        
        # Pred (予測): ID + Suffix
        img_pred = find_image(DIR_PRED, lq_name)

        # 画像が揃っているかチェック
        if any(x is None for x in [img_gt, img_lq, img_gt_mask, img_pred]):
            print(f"[Skip] Missing files for {category}: {file_id}")
            if img_lq is None: print(f"  - LQ not found: {lq_name}")
            if img_gt is None: print(f"  - GT not found: {file_id}")
            if img_gt_mask is None: print(f"  - GT Mask not found: {file_id}")
            if img_pred is None: print(f"  - Pred not found: {lq_name}")
            continue

        # --- 2. リサイズと整形 ---
        images = [img_lq, img_gt, img_gt_mask, img_pred]
        resized_images = []
        
        for img in images:
            # リサイズ
            img = cv2.resize(img, TARGET_SIZE)
            # グレースケールならRGBに変換 (結合のため)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized_images.append(img)

        # --- 3. 結合 (Input | GT | GT Mask | Pred) ---
        # 白い区切り線を入れる
        spacer = np.ones((TARGET_SIZE[1], 10, 3), dtype=np.uint8) * 255
        
        combined = np.hstack([
            resized_images[0], spacer, # Input
            resized_images[1], spacer, # GT (Clean)
            resized_images[2], spacer, # GT Mask
            resized_images[3]          # Pred Mask
        ])

        # --- 4. 保存 ---
        # ファイル名を見やすく整形 (例: Stage1_Missing.png)
        # "Transparent (Failure)" -> "Stage1_Transparent_Failure.png"
        safe_cat = category.replace(" (Failure)", "_Failure").replace(" ", "")
        out_filename = f"Stage1_{safe_cat}.png"
        out_path = out_root / out_filename
        
        cv2.imwrite(str(out_path), combined)
        print(f"[OK] Saved: {out_filename}")

    print("\nDone. Images are saved in 'figures' folder.")

if __name__ == "__main__":
    main()