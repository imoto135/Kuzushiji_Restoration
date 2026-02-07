import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 設定
# ==========================================
# 予測画像があるフォルダ (ここに _Scratch.jpg などがある)
PRED_DIR = "outputs/lastmask_unet++/test"

# 正解マスクがあるフォルダ (ここに ID.jpg がある)
GT_DIR = "datasets/hiragana_dataset/gt_mask/test"

# 出力ファイル名
OUTPUT_CSV = "Stage1_Evaluation_Log.csv"
OUTPUT_SUMMARY_CSV = "Stage1_Evaluation_Summary.csv"

# サフィックスとカテゴリ名の対応表
# ※ 長いサフィックスから順に記述することが必須
SUFFIX_MAP = {
    "_Transparent_Stain": "Transparent",
    "_Missing": "Missing",
    "_Scratch": "Scratch",
    "_Stain": "Stain",
    "_Ghosting": "Ghosting"
}

def parse_filename(filename):
    """
    ファイル名からサフィックスを除去して (ID, カテゴリ) を返す
    例: "U+304A..._Scratch.jpg" -> ("U+304A...", "Scratch")
    """
    stem = Path(filename).stem  # 拡張子なしのファイル名
    
    # サフィックスのマッチング
    for suffix, category in SUFFIX_MAP.items():
        if stem.endswith(suffix):
            file_id = stem[:-len(suffix)]
            return file_id, category
            
    return None, None  # サフィックスが見つからない場合

def calculate_metrics(gt_path, pred_path):
    """
    正解マスクと予測マスクを比較して指標を計算する
    """
    # グレースケール読み込み
    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

    if gt_img is None or pred_img is None:
        return None

    # サイズ合わせ (予測を正解に合わせる)
    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # 正規化 (0-1)
    gt = gt_img.astype(np.float32) / 255.0
    pred = pred_img.astype(np.float32) / 255.0

    # --- Soft Metrics ---
    soft_intersection = np.sum(pred * gt)
    soft_union = np.sum(pred + gt - pred * gt)
    soft_iou = soft_intersection / (soft_union + 1e-6)
    mae = np.mean(np.abs(gt - pred))

    # --- Hard Metrics (閾値0.5) ---
    gt_bin = (gt > 0.5).astype(np.float32)
    pred_bin = (pred > 0.5).astype(np.float32)

    hard_intersection = np.sum(gt_bin * pred_bin)
    hard_union = np.sum(gt_bin) + np.sum(pred_bin) - hard_intersection
    hard_iou = hard_intersection / (hard_union + 1e-6)
    f1 = (2.0 * hard_intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-6)

    return {
        "F1": round(f1, 4),
        "HardIoU": round(hard_iou, 4),
        "SoftIoU": round(soft_iou, 4),
        "MAE": round(mae, 4)
    }

def main():
    pred_root = Path(PRED_DIR)
    gt_root = Path(GT_DIR)

    if not pred_root.exists():
        print(f"[Error] Prediction Folder '{PRED_DIR}' not found.")
        return
    if not gt_root.exists():
        print(f"[Error] GT Folder '{GT_DIR}' not found.")
        return

    results = []
    
    # .jpg と .png 両方を検索対象にする
    # 再帰的(rglob)ではなく直下(glob)を見る
    pred_files = list(pred_root.glob("*.jpg")) + list(pred_root.glob("*.png"))
    print(f"Scanning '{PRED_DIR}'... Found {len(pred_files)} images.")

    count_calculated = 0

    for pred_path in pred_files:
        # ファイル名を解析
        file_id, category = parse_filename(pred_path.name)
        
        if file_id is None:
            # サフィックスがないファイルは無視 (評価対象外)
            continue

        # 正解マスクを探す (jpg -> png の順)
        gt_path = gt_root / f"{file_id}.jpg"
        if not gt_path.exists():
            gt_path = gt_root / f"{file_id}.png"
        
        if gt_path.exists():
            metrics = calculate_metrics(gt_path, pred_path)
            
            if metrics:
                metrics["Category"] = category
                metrics["File_ID"] = file_id
                metrics["Filename_Pred"] = pred_path.name
                results.append(metrics)
                count_calculated += 1
        else:
            # デバッグ: 紐付け失敗した理由を表示
            # print(f"  [Skip] GT not found for ID: {file_id} (Pred: {pred_path.name})")
            pass

    print(f"Calculated metrics for {count_calculated} pairs.")

    if not results:
        print("[Warning] No matched pairs found.")
        print("ヒント: PRED_DIR のパスが正しいか、GTフォルダに同じIDのファイルがあるか確認してください。")
        return

    # --- CSV保存 ---
    df = pd.DataFrame(results)
    
    # 1. 全データ
    cols = ["Category", "File_ID", "F1", "HardIoU", "SoftIoU", "MAE", "Filename_Pred"]
    df = df[cols]
    
    out_path = pred_root / OUTPUT_CSV
    df.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved raw data: {out_path}")

    # 2. カテゴリ別平均 (サマリ)
    summary_df = df.groupby("Category")[["F1", "HardIoU", "SoftIoU", "MAE"]].mean()
    summary_path = pred_root / OUTPUT_SUMMARY_CSV
    summary_df.to_csv(summary_path)
    print(f"[DONE] Saved summary: {summary_path}")

    print("\n--- Category Summary (Average) ---")
    print(summary_df)

if __name__ == "__main__":
    main()