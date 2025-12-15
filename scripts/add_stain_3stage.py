import os
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from pathlib import Path
from tqdm import tqdm

# ==========================================
#               設定エリア
# ==========================================

# 【入力】元画像が入っているフォルダ
INPUT_DIR = "dataset_final_hiragana/gt/train"

# 【出力】保存先のフォルダ
OUTPUT_DIR = "dataset_final_hiragana/lq_3stage/train"

# 【汚れの大きさ】 (デフォルト: 30, 大きくするなら 50~100程度)
BLOB_SIZE = 80

# 【乱数シード】
SEED = 42

# 【透明度とサフィックスの設定】
# キー: ファイル名のサフィックス, 値: 不透明度 (0.0~1.0)
# この中からランダムに1つが選ばれます
ALPHA_SETTINGS = {
    "alpha30": 0.3,
    "alpha60": 0.6,
    "alpha90": 0.9
}

# ==========================================
#            関数定義
# ==========================================

def random_blobs(shape, blobdensity=3e-4, size=39, roughness=2.0):
    from random import randint
    h, w = shape
    numblobs = int(blobdensity * w * h)
    
    # numblobsが0にならないように最低1つは確保
    numblobs = max(1, numblobs)
    
    mask = np.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[randint(0, h-1), randint(0, w-1)] = 1
    dt = ndi.distance_transform_edt(1-mask)
    mask =  np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size/(2*roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    # pylab.rand -> np.random.rand に変更
    noise = np.random.rand(h, w)
    noise = ndi.gaussian_filter(noise, size/(2*roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'f')

def apply_stain(img, alpha=0.5, blob_size=30):
    """
    画像に汚れを付与する
    img: 入力画像 (BGR, uint8)
    alpha: 汚れの不透明度 (0.0 ~ 1.0)
    blob_size: 汚れの大きさ
    """
    h, w, c = img.shape
    
    # 1. 汚れの形状（マスク）を生成
    # サイズを大きくするため blob_size を使用
    stain_pos = random_blobs((h, w), blobdensity=3e-4, size=blob_size)
    
    # 2. 汚れの色をランダムに決定
    r_rand = np.random.randint(134, 135)
    g_rand = np.random.randint(74, 75)
    b_rand = np.random.randint(43, 44)
    
    # BGR順の汚れレイヤーを作成
    stain_layer = np.zeros((h, w, 3), dtype=np.float32)
    stain_layer[:, :, 0] = stain_pos * b_rand # B
    stain_layer[:, :, 1] = stain_pos * g_rand # G
    stain_layer[:, :, 2] = stain_pos * r_rand # R

    # 3. マスクの生成
    final_mask = stain_pos 
    final_mask = np.stack([final_mask]*3, axis=-1) # 3チャンネルに拡張

    # 4. アルファブレンディングによる合成
    img_float = img.astype(np.float32)
    
    # 汚れレイヤー自体は (0~255) の値を持っているので、それを alpha の割合で混ぜる
    blended = img_float * (1 - alpha) + stain_layer * alpha
    
    # マスク部分だけブレンド結果を採用、それ以外は元の画像
    output = img_float * (1 - final_mask) + blended * final_mask
    
    return np.clip(output, 0, 255).astype(np.uint8)

def main():
    # シード固定
    random.seed(SEED)
    np.random.seed(SEED)

    input_path = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"エラー: 入力フォルダが見つかりません: {INPUT_DIR}")
        return

    os.makedirs(output_root, exist_ok=True)

    # 画像ファイルリストの取得
    image_files = [p for p in input_path.glob('**/*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
    
    if not image_files:
        print("画像ファイルが見つかりませんでした。")
        return

    print(f"処理開始: 全{len(image_files)}枚")
    print(f"設定: 汚れサイズ={BLOB_SIZE}, 設定候補={list(ALPHA_SETTINGS.keys())} (この中からランダムに1つ適用)")

    for file_path in tqdm(image_files, desc="Processing"):
        try:
            # 画像読み込み (日本語パス対応)
            img_array = np.fromfile(str(file_path), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None: continue

            # === 変更点: 設定リストからランダムに1つ選ぶ ===
            suffix, alpha_val = random.choice(list(ALPHA_SETTINGS.items()))
            
            # 汚れ付与
            stained_img = apply_stain(image, alpha=alpha_val, blob_size=BLOB_SIZE)
            
            # 保存ファイル名生成 (例: original_alpha60.png)
            # どの汚れが適用されたかわかるようにサフィックスは残しています
            filename = f"{file_path.stem}_{suffix}{file_path.suffix}"
            save_path = output_root / filename
            
            # 保存 (日本語パス対応)
            _, buf = cv2.imencode(file_path.suffix, stained_img)
            with open(save_path, "wb") as f:
                buf.tofile(f)

        except Exception as e:
            print(f"エラー発生 {file_path.name}: {e}")

    print(f"\n完了しました！ 出力先: {output_root.resolve()}")

if __name__ == "__main__":
    main()