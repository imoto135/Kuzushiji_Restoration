"""
Restormer vs NAFNet 比較画像生成スクリプト

左から: GT | LQ | Restormer | NAFNet の順で横に連結した画像を生成
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import re

# ==========================================
# 設定
# ==========================================
# 各ディレクトリのパス
DIR_GT = "hiragana_fulldataset_5stain/gt/test"
DIR_LQ = "hiragana_fulldataset_5stain/lq/test"
DIR_RESTORMER = "outputs/restormer_restored_predtest"
DIR_NAFNET = "outputs/nafnet_mask_charbpercep"

# 出力先
OUTPUT_DIR = "co"

# サンプル数
NUM_SAMPLES = 50

# 画像サイズ (統一してリサイズ)
TARGET_SIZE = (256, 256)

# 画像間のパディング
IMG_PADDING = 5

# ラベル設定
LABELS = ["GT", "LQ", "Restormer", "NAFNet"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
TEXT_PADDING = 25  # テキスト領域の高さ


def extract_base_id(filename: str) -> str:
    """
    ファイル名からベースID（損傷タイプを除く部分）を抽出
    例: U+3042_100241706_00006_1_X0668_Y1844_Transparent_Stain.jpg
        -> U+3042_100241706_00006_1_X0668_Y1844
    """
    # 拡張子を除去
    name = Path(filename).stem
    
    # 損傷タイプパターンを除去
    damage_types = ['_Transparent_Stain', '_Stain', '_Missing', '_Scratch', '_Ghosting']
    for damage in damage_types:
        if name.endswith(damage):
            name = name[:-len(damage)]
            break
    
    return name


def find_matching_files(lq_files: list) -> list:
    """
    LQファイルに対応するGT, Restormer, NAFNetファイルを見つける
    """
    matches = []
    
    for lq_file in lq_files:
        base_id = extract_base_id(lq_file)
        lq_stem = Path(lq_file).stem  # 損傷タイプ込みのファイル名
        
        # GT: ベースIDのみ
        gt_path = Path(DIR_GT) / f"{base_id}.jpg"
        
        # Restormer: 損傷タイプ込み (.png)
        restormer_path = Path(DIR_RESTORMER) / f"{lq_stem}.png"
        
        # NAFNet: 損傷タイプ込み (.jpg)
        nafnet_path = Path(DIR_NAFNET) / f"{lq_stem}.jpg"
        
        # 全てのファイルが存在する場合のみ追加
        if gt_path.exists() and restormer_path.exists() and nafnet_path.exists():
            matches.append({
                'base_id': base_id,
                'lq_stem': lq_stem,
                'gt': str(gt_path),
                'lq': str(Path(DIR_LQ) / lq_file),
                'restormer': str(restormer_path),
                'nafnet': str(nafnet_path)
            })
    
    return matches


def load_and_resize(path: str) -> np.ndarray:
    """画像を読み込んでリサイズ"""
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Could not load: {path}")
        return np.ones((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8) * 255
    
    img = cv2.resize(img, TARGET_SIZE)
    
    # グレースケールの場合はBGRに変換
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return img


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    """画像の上にラベルを追加"""
    h, w = img.shape[:2]
    
    # ラベルエリアを上に追加
    canvas = np.ones((h + TEXT_PADDING, w, 3), dtype=np.uint8) * 255
    canvas[TEXT_PADDING:, :] = img
    
    # テキストを中央揃えで描画
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    tx = (w - tw) // 2
    ty = (TEXT_PADDING + th) // 2
    cv2.putText(canvas, label, (tx, ty), FONT, FONT_SCALE, (0, 0, 0), THICKNESS)
    
    return canvas


def concat_images(match: dict) -> np.ndarray:
    """GT, LQ, Restormer, NAFNetを横に連結"""
    images = []
    paths = [match['gt'], match['lq'], match['restormer'], match['nafnet']]
    
    for i, (path, label) in enumerate(zip(paths, LABELS)):
        img = load_and_resize(path)
        img_with_label = add_label(img, label)
        images.append(img_with_label)
    
    # 横方向に連結（パディング付き）
    h = images[0].shape[0]
    spacer = np.ones((h, IMG_PADDING, 3), dtype=np.uint8) * 255
    
    result = images[0]
    for img in images[1:]:
        result = np.hstack([result, spacer, img])
    
    return result


def main():
    # 出力ディレクトリを作成
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # LQディレクトリのファイル一覧を取得
    lq_files = [f for f in os.listdir(DIR_LQ) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(lq_files)} LQ files")
    
    # マッチングファイルを検索
    matches = find_matching_files(lq_files)
    print(f"Found {len(matches)} complete matches (GT + LQ + Restormer + NAFNet)")
    
    if len(matches) == 0:
        print("[ERROR] No matching files found!")
        return
    
    # ランダムに選択
    num_samples = min(NUM_SAMPLES, len(matches))
    selected = random.sample(matches, num_samples)
    print(f"Selected {num_samples} random samples")
    
    # 各サンプルを処理
    for i, match in enumerate(selected):
        print(f"[{i+1}/{num_samples}] Processing: {match['lq_stem']}")
        
        # 画像を連結
        combined = concat_images(match)
        
        # 保存
        out_filename = f"compare_{match['lq_stem']}.png"
        out_path = out_dir / out_filename
        cv2.imwrite(str(out_path), combined)
    
    print(f"\n[DONE] Saved {num_samples} comparison images to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    # 再現性のためにシードを設定（必要に応じてコメントアウト）
    random.seed(42)
    main()
