import os
import glob
import random
import warnings
import numpy as np
import scipy.ndimage as ndi
import pylab
from skimage import io, color
from PIL import Image
import argparse

# 警告を抑制
warnings.filterwarnings("ignore")

def set_seed(seed):
    """シード値を固定して再現性を確保する"""
    random.seed(seed)
    np.random.seed(seed)
    # 必要に応じて以下のツールに影響を与えないよう設定
    # imgaugや他の乱数生成器を使用している場合はここに追加


# ---------------------------------------------------------
# 損傷付与関数の定義 (提供されたコードベース)
# ---------------------------------------------------------

def random_blobs(shape, blobdensity=3e-4, size=39, roughness=2.0):
    from random import randint
    h, w = shape
    numblobs = int(blobdensity * w * h)
    mask = np.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[randint(0, h-1), randint(0, w-1)] = 1
    dt = ndi.distance_transform_edt(1-mask)
    mask =  np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size/(2*roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = pylab.rand(h, w)
    noise = ndi.gaussian_filter(noise, size/(2*roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'f')

def random_ghost(img):
    image_flip = np.flip(img, axis=1)
    gray = color.rgb2gray(image_flip)*2-1
    ghost = np.clip((gray > 0).astype(float) + 0.5, 0, 1)
    ghost = ghost[..., None]
    img_degrad = img * ghost
    img_degrad = (np.clip(img_degrad,0,255)).astype(np.uint8)
    return img_degrad

def random_stain(img, severity=0.5):
    if len(img.shape) == 3:
        H,W,C = img.shape
    else:
        H,W = img.shape
        img = img[:,:,None]
        C = 1

    # 汚れの形状（マスク）を作成 (0.0 ~ 1.0)
    # severityを使って汚れの密度や大きさを調整しても良いですが、
    # ここでは元のロジックのまま 3e-4, 30 を使用します
    stain_pos = random_blobs((H,W), 3e-4, 30)
    
    # 汚れの色をランダム決定
    r_rand = np.random.randint(128,169)
    g_rand = np.random.randint(128,169)
    b_rand = np.random.randint(128,169)
    
    # 汚れ画像を作成
    stain_color = np.zeros((H,W,3), dtype=np.float32)
    stain_color[:,:,0] = r_rand
    stain_color[:,:,1] = g_rand
    stain_color[:,:,2] = b_rand
    
    # マスクの次元を合わせる (H,W) -> (H,W,1)
    mask = stain_pos[..., None]

    # 【修正点】
    # 元の画像(img)と汚れ(stain_color)を、マスクを使って合成します。
    # maskが1の部分は汚れの色になり、0の部分は元の画像になります。
    # これにより、文字の有無に関係なく汚れが上書きされます。
    img = img.astype(np.float32)
    img_stain = img * (1 - mask) + stain_color * mask
    
    img_stain = (np.clip(img_stain, 0, 255)).astype(np.uint8)
    
    # マスクも返す (可視化や学習用)
    return img_stain, mask

def random_transparent_stain(img):
    """透明度が30%から80%のランダムな汚れ（Stain）を画像に付与する"""
    if len(img.shape) == 3:
        H,W,C = img.shape
    else:
        H,W = img.shape
        img = np.stack((img,)*3, axis=-1) # グレーをRGB化
        C = 3

    # 1. Stainの位置生成
    stain_pos = (random_blobs((H,W), 3e-4, 30))
    stain_mask = stain_pos[:,:,None]

    # 2. 汚れの色生成
    r_rand = np.random.randint(128,169)
    g_rand = np.random.randint(128,169)
    b_rand = np.random.randint(128,169)

    stain_color = np.zeros((H,W,3)).astype(np.float32)
    stain_color[:,:,0] = (stain_pos[:,:] * r_rand)
    stain_color[:,:,1] = (stain_pos[:,:] * g_rand)
    stain_color[:,:,2] = (stain_pos[:,:] * b_rand)
    
    # 3. 透明度(アルファ) 30% ~ 80%
    alpha = np.random.uniform(0.3, 0.8)

    # 4. 合成処理
    img_float = img.astype(np.float32)
    
    # アルファブレンド: (1 - alpha) * img + alpha * stain
    # stain_maskがある部分だけブレンド計算を行う
    img_degrad = (1 - alpha * stain_mask) * img_float + (alpha * stain_mask) * stain_color
    
    img_degrad = (np.clip(img_degrad, 0, 255)).astype(np.uint8)
    return img_degrad

def random_missing(img):
    if len(img.shape) == 3:
        H,W,C = img.shape
    else:
        H,W = img.shape
        img = np.stack((img,)*3, axis=-1)

    degrad_pos = (random_blobs((H,W), 3e-4, 30))

    vals_r, cnt_r = np.unique(img[:,:,0], return_counts=True)
    vals_g, cnt_g = np.unique(img[:,:,1], return_counts=True)
    vals_b, cnt_b = np.unique(img[:,:,2], return_counts=True)
    r_mode = vals_r[np.argmax(cnt_r)]
    g_mode = vals_g[np.argmax(cnt_g)]
    b_mode = vals_b[np.argmax(cnt_b)]
    
    degrad = np.zeros((H,W,3)).astype(np.uint8)
    degrad[:,:,0] = (degrad_pos[:,:] * r_mode)
    degrad[:,:,1] = (degrad_pos[:,:] * g_mode)
    degrad[:,:,2] = (degrad_pos[:,:] * b_mode)
    
    img = (np.clip(img,0,255)).astype(np.uint8)
    img_degrad = img * (1-degrad_pos[:,:,None]) + degrad
    img_degrad = (np.clip(img_degrad,0,255)).astype(np.uint8)
    return img_degrad

def random_scratch(img, freq=5, size=1e-4):
    if len(img.shape) == 3:
        H,W,C = img.shape
    else:
        H,W = img.shape
        img = np.stack((img,)*3, axis=-1)

    degrad_pos = (random_blobs((H,W), freq, size))

    vals_r, cnt_r = np.unique(img[:,:,0], return_counts=True)
    vals_g, cnt_g = np.unique(img[:,:,1], return_counts=True)
    vals_b, cnt_b = np.unique(img[:,:,2], return_counts=True)
    r_mode = vals_r[np.argmax(cnt_r)]
    g_mode = vals_g[np.argmax(cnt_g)]
    b_mode = vals_b[np.argmax(cnt_b)]
    
    degrad = np.zeros((H,W,3)).astype(np.uint8)
    degrad[:,:,0] = (degrad_pos[:,:] * r_mode)
    degrad[:,:,1] = (degrad_pos[:,:] * g_mode)
    degrad[:,:,2] = (degrad_pos[:,:] * b_mode)
    
    img = (np.clip(img,0,255)).astype(np.uint8)
    img_degrad = img * (1-degrad_pos[:,:,None]) + degrad
    img_degrad = (np.clip(img_degrad,0,255)).astype(np.uint8)
    return img_degrad

# ---------------------------------------------------------
# メイン処理: ディレクトリ一括変換
# ---------------------------------------------------------

def process_images(input_dir, output_dir):
    # 出力ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)

    # 対応する拡張子
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"発見された画像数: {len(image_files)}枚")
    print(f"入力: {input_dir}")
    print(f"保存先: {output_dir}")

    # 損傷タイプの定義
    degradation_types = ['Missing', 'Stain', 'Scratch', 'Ghosting', 'Transparent_Stain']

    for i, img_path in enumerate(image_files):
        try:
            # ファイル名の取得
            file_name = os.path.basename(img_path)
            name_only, ext = os.path.splitext(file_name)
            
            # 画像の読み込み (RGBに統一)
            img = io.imread(img_path)
            
            # Alphaチャンネルがある場合は削除 (RGBA -> RGB)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:,:,:3]
            
            # グレースケールの場合はRGB化 (処理の統一のため)
            if len(img.shape) == 2:
                img = color.gray2rgb(img)
                
            img = img.astype(np.uint8)

            # ランダムに損傷を選択
            selected_deg = random.choice(degradation_types)

            # 損傷の適用
            if selected_deg == 'Missing':
                img_result = random_missing(img)
            elif selected_deg == 'Stain':
                img_result, _ = random_stain(img)
            elif selected_deg == 'Scratch':
                img_result = random_scratch(img)
            elif selected_deg == 'Ghosting':
                img_result = random_ghost(img)
            elif selected_deg == 'Transparent_Stain':
                img_result = random_transparent_stain(img)
            
            # 保存 (ファイル名に損傷タイプを付与)
            save_name = f"{name_only}_{selected_deg}{ext}"
            save_path = os.path.join(output_dir, save_name)
            
            Image.fromarray(img_result).save(save_path)
            print(f"[{i+1}/{len(image_files)}] Processed: {file_name} -> Type: {selected_deg}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def process_splits(input_root, output_root, splits=("train", "test", "val")):
    """
    input_root/{split} 内の画像を読み込み、output_root/{split}に出力する
    """
    for split in splits:
        in_dir = os.path.join(input_root, split)
        out_dir = os.path.join(output_root, split)

        if not os.path.exists(in_dir):
            print(f"[SKIP] 入力ディレクトリが見つかりません: {in_dir}")
            continue

        print("\n" + "-" * 60)
        print(f"Split: {split}")
        process_images(in_dir, out_dir)

# ---------------------------------------------------------
# 実行設定
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", required=True, help="入力データセットのルート (train, val, testを含むディレクトリ)")
    parser.add_argument("-o", "--output_root", required=True, help="出力先のルートディレクトリ")
    parser.add_argument("--splits", default="train,val,test", help="カンマ区切りの split")
    parser.add_argument("--seed", type=int, default=42, help="乱数のシード値 (再現性のため)")
    args = parser.parse_args()

    # シードの設定
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    process_splits(
        input_root=args.input_root,
        output_root=args.output_root,
        splits=splits,
    )