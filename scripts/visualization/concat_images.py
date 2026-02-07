# import os
# import cv2
# import numpy as np
# from pathlib import Path

# # ==========================================
# # 設定
# # ==========================================
# COLLECTION_DIR = "sotsuron/collection"
# OUTPUT_FILENAME = "Combined_Figure_3x2.png"

# # 画像サイズ（このサイズにリサイズして統一します）
# TARGET_SIZE = (256, 256)

# # レイアウト定義 (3行2列)
# # ここに書かれた順に配置されます。"Mask_GT" を外して6枚にしています。
# LAYOUT = [
#     ["Input",      "GT"],
#     ["Baseline",   "Ours"],
#     ["Mask_P",     "Oracle"]
# ]

# # 画像の上に表示するラベル名
# LABELS = {
#     "Input":    "Input",
#     "GT":       "Ground Truth",
#     "Baseline": "Baseline",
#     "Ours":     "Ours(Proposed)",
#     "Mask_P":   "Predicted Mask",
#     "Oracle":   "Ideal",
#     "Mask_GT":  "GT Mask"
# }

# # デザイン設定
# FONT = cv2.FONT_HERSHEY_SIMPLEX
# FONT_SCALE = 1.0
# THICKNESS = 2
# TEXT_PADDING = 40  # テキスト領域の高さ
# IMG_PADDING = 10   # 画像間の隙間（白）

# def main():
#     root = Path(COLLECTION_DIR)
    
#     # フォルダ探索
#     for folder in root.glob("*/*"): 
#         if not folder.is_dir(): continue
        
#         print(f"Processing: {folder.name} ({folder.parent.name})")

#         # 画像読み込み関数
#         def load(name):
#             for ext in ['.png', '.jpg', '.jpeg']:
#                 p = folder / f"{name}{ext}"
#                 if p.exists():
#                     return cv2.imread(str(p))
#             # ズーム画像があるならそれを使う場合 (例: "Zoom_Input")
#             for ext in ['.png', '.jpg', '.jpeg']:
#                 p = folder / f"Zoom_{name}{ext}"
#                 if p.exists():
#                     return cv2.imread(str(p))
#             return None

#         # --- 画像の結合処理 ---
#         rows_images = []
        
#         for row_keys in LAYOUT:
#             row_imgs = []
#             for key in row_keys:
#                 img = load(key)
                
#                 if img is None:
#                     print(f"  [WARN] Missing image: {key}")
#                     # 画像がない場合は白い空白を埋める
#                     img = np.ones((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8) * 255
                
#                 # リサイズ
#                 img = cv2.resize(img, TARGET_SIZE)
                
#                 # ラベルエリア（白い帯）を作成
#                 h, w = img.shape[:2]
#                 canvas = np.ones((h + TEXT_PADDING, w, 3), dtype=np.uint8) * 255
#                 canvas[TEXT_PADDING:, :] = img  # 画像を下に配置
                
#                 # テキスト描画（中央揃え）
#                 text = LABELS.get(key, key)
#                 (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
#                 tx = (w - tw) // 2
#                 ty = (TEXT_PADDING + th) // 2 # 少し上に調整
#                 cv2.putText(canvas, text, (tx, ty - 5), FONT, FONT_SCALE, (0, 0, 0), THICKNESS)
                
#                 row_imgs.append(canvas)

#             # 横方向の結合（パディング付き）
#             if row_imgs:
#                 row_concat = row_imgs[0]
#                 for i in range(1, len(row_imgs)):
#                     spacer = np.ones((row_concat.shape[0], IMG_PADDING, 3), dtype=np.uint8) * 255
#                     row_concat = np.hstack([row_concat, spacer, row_imgs[i]])
#                 rows_images.append(row_concat)

#         # 縦方向の結合（パディング付き）
#         if rows_images:
#             final_img = rows_images[0]
#             for i in range(1, len(rows_images)):
#                 spacer = np.ones((IMG_PADDING, final_img.shape[1], 3), dtype=np.uint8) * 255
#                 final_img = np.vstack([final_img, spacer, rows_images[i]])
            
#             save_path = folder / OUTPUT_FILENAME
#             cv2.imwrite(str(save_path), final_img)

#     print(f"\n[DONE] 全フォルダに '{OUTPUT_FILENAME}' を作成しました。")

# if __name__ == "__main__":
#     main()

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