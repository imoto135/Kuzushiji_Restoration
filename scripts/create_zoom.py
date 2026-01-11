import os
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# 設定
# ==========================================
COLLECTION_DIR = "sotsuron/collection"
ZOOM_SIZE = 128  # 拡大領域のサイズ (128x128)

# 手動で座標を指定したい場合はここに記述 (フォルダ名: (x, y))
# ※ x, y は拡大したい領域の【中心座標】を指定してください
# ※ 自動検出がうまくいかない場合のみ、ここに追加すればOKです
MANUAL_COORDS = {
    # 例: "Best_PSNR": (250, 150),
    # "Failure_Case_Discussion": (100, 100),
}

def get_best_zoom_area(img_gt, img_base, img_ours, size=128):
    """
    BaselineとGTの差が大きく、かつOursとGTの差が小さい場所（＝改善が顕著な場所）を自動探索
    """
    # グレースケール変換
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
    gray_base = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
    gray_ours = cv2.cvtColor(img_ours, cv2.COLOR_RGB2GRAY)

    # 誤差の絶対値
    diff_base = cv2.absdiff(gray_gt, gray_base)
    diff_ours = cv2.absdiff(gray_gt, gray_ours)
    
    # 改善度マップ (Baseline誤差 - Ours誤差)
    # 値が大きいほど「Baselineは間違っているが、Oursは合っている」場所
    improvement = diff_base.astype(np.float32) - diff_ours.astype(np.float32)
    
    # ノイズ除去（平滑化）して、局所的なピークではなく全体的な改善を見る
    improvement = cv2.GaussianBlur(improvement, (9, 9), 0)
    
    # 指定サイズの窓で「合計改善量」が最大になる場所を探す
    kernel = np.ones((size, size), np.float32)
    score_map = cv2.filter2D(improvement, -1, kernel)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score_map)
    
    # filter2Dの結果、max_loc はウィンドウの中心座標に相当する
    cx, cy = max_loc
    return cx, cy

def main():
    root = Path(COLLECTION_DIR)
    
    # collectionフォルダ以下の全サブフォルダを探索
    for folder in root.glob("*/*"): 
        if not folder.is_dir(): continue
        
        folder_name = folder.name
        print(f"Processing zoom for: {folder_name} ({folder.parent.name})")
        
        # 画像読み込み関数
        def load(name):
            for ext in ['.png', '.jpg', '.jpeg']:
                p = folder / f"{name}{ext}"
                if p.exists():
                    return cv2.imread(str(p))
            return None

        img_gt = load("GT")
        img_base = load("Baseline")
        img_ours = load("Ours")
        
        # 必須画像がない場合はスキップ
        if img_gt is None or img_base is None or img_ours is None:
            print(f"  [SKIP] Necessary images not found in {folder_name}")
            continue

        h, w = img_gt.shape[:2]

        # --- 座標決定ロジック ---
        # 1. 手動指定があるか確認
        if folder_name in MANUAL_COORDS:
            cx, cy = MANUAL_COORDS[folder_name]
            print(f"  [INFO] Manual coordinate used: ({cx}, {cy})")
        
        # 2. なければ自動探索
        else:
            try:
                cx, cy = get_best_zoom_area(img_gt, img_base, img_ours, ZOOM_SIZE)
                print(f"  [INFO] Auto coordinate found: ({cx}, {cy})")
            except Exception as e:
                print(f"  [WARN] Auto detection failed: {e}. Using center.")
                cx, cy = w // 2, h // 2

        # --- 切り抜き範囲の計算 (画像からはみ出さないように) ---
        x1 = max(0, min(cx - ZOOM_SIZE // 2, w - ZOOM_SIZE))
        y1 = max(0, min(cy - ZOOM_SIZE // 2, h - ZOOM_SIZE))
        
        # --- 画像の切り抜きと保存 ---
        # 収集スクリプトで保存したファイル名を使用
        targets = ["Input", "GT", "Baseline", "Ours", "Oracle", "Mask_P", "Mask_GT"]
        
        for target in targets:
            img = load(target)
            if img is not None:
                # 万が一サイズが違う場合はリサイズ
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                
                crop = img[y1:y1+ZOOM_SIZE, x1:x1+ZOOM_SIZE]
                cv2.imwrite(str(folder / f"Zoom_{target}.png"), crop)
        
        # 確認用: Ours画像に赤枠を描画して保存
        img_rect = img_ours.copy()
        cv2.rectangle(img_rect, (x1, y1), (x1+ZOOM_SIZE, y1+ZOOM_SIZE), (0, 0, 255), 2)
        cv2.imwrite(str(folder / "CHECK_ZOOM_AREA.png"), img_rect)

    print("\n[DONE] 全てのフォルダに 'Zoom_xxx.png' と確認用の 'CHECK_ZOOM_AREA.png' が作成されました。")

if __name__ == "__main__":
    main()