import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import pandas as pd

# 関連するモデルの設計図をインポート
from basicsr.models.archs.restormer_arch import Restormer

def main():
    ### パスとパラメータ設定 ###
    # 使用する学習済みモデルのパス (最適な重みファイルを選択してください)
    # 【要設定】マルチタスク学習させたモデルのパス
    MODEL_PATH = "experiments/AncientTextRestoration_MultiTask_Run02/models/net_g_latest.pth"
    # 【入力】修復したい本物の損傷画像が入ったフォルダ
    INPUT_DIR  = "make_damage/sample_damaged/padded"
    # 【出力】修復後の画像を保存するフォルダ
    OUTPUT_DIR = "results/restored_with_multitask_model02"
    # 【重要】学習時に使用したクラスのマッピングファイルを指定
    CLASS_MAP_CSV = "class_map.csv"
    # 【重要】学習時に.ymlファイルで指定したクラス総数をここに設定
    NUM_CLASSES = 3108
    ### ---

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- クラスIDと文字のマッピングを読み込む ---
    try:
        class_map_df = pd.read_csv(CLASS_MAP_CSV)
        char_to_id = pd.Series(class_map_df.class_id.values, index=class_map_df.char_unicode).to_dict()
        num_classes = len(char_to_id)
        print(f"クラスマッピング {CLASS_MAP_CSV} をロードしました。クラス総数: {num_classes}")
    except FileNotFoundError:
        print(f"エラー: {CLASS_MAP_CSV} が見つかりません。このスクリプトと同じディレクトリに配置してください。")
        return
    
    # --- モデルの準備 (条件付き修復モデル) ---
    model = Restormer(
        inp_channels=4, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint.get('params', checkpoint))
    model.eval()
    model = model.to(device)
    print(f"モデル {os.path.basename(MODEL_PATH)} をロードしました。")
    print(f"推論デバイス: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in tqdm(image_files, desc="画像を修復中"):
        try:
            input_path = os.path.join(INPUT_DIR, filename)
            
            # 1. ファイル名から文字ラベルを取得
            char_label_unicode = filename.split('_')[0]
            if char_label_unicode not in char_to_id:
                print(f"\n警告: {filename} のラベルがクラスマップにありません。スキップします。")
                continue
            class_id = char_to_id[char_label_unicode]

            # 2. RGB画像を読み込み、テンソルに変換
            img = Image.open(input_path).convert('RGB')
            img_tensor = transform(img)
            
            # 3. ラベルマップを作成
            c, h, w = img_tensor.shape
            normalized_label = float(class_id) / (num_classes - 1)
            label_map = torch.full((1, h, w), normalized_label)

            # 4. RGBテンソルとラベルマップを結合して4チャンネル入力を作成
            input_tensor = torch.cat([img_tensor, label_map], dim=0).unsqueeze(0).to(device)

            # 5. 推論実行
            with torch.no_grad():
                # --- ★★★ 変更点: モデルの出力をタプルとして正しく受け取る ★★★ ---
                # 最初の要素が修復画像、2番目の要素（_）は無視する
                output = model(input_tensor)
                if isinstance(output, tuple):
                    restored_tensor = output[0]
                else:
                    restored_tensor = output
                # --- ここまで ---

            # 6. 後処理
            restored_tensor = torch.clamp(restored_tensor, 0, 1)
            restored_img_rgb = restored_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            restored_img_bgr = (cv2.cvtColor(restored_img_rgb, cv2.COLOR_RGB2BGR) * 255.0).round().astype(np.uint8)
            
            # 7. 比較画像を生成して保存
            original_img_bgr = cv2.imread(input_path)
            comparison_image = np.concatenate((original_img_bgr, restored_img_bgr), axis=1)
            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, comparison_image)
            
        except Exception as e:
            print(f"\nエラー: {filename} の処理中に問題が発生しました - {e}")

    print(f"\n修復が完了しました。結果を {OUTPUT_DIR} に保存しました。")

if __name__ == "__main__":
    main()
