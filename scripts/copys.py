import splitfolders
import os

# --- 設定 ---
# 【入力元】分割したい、全ての画像が入っているフォルダ
# 例: "01_gt_originals" や "kmnist_dataset_clean/train" など
INPUT_FOLDER = "dataset/" 

# 【出力先】分割後の'train', 'val', 'test'フォルダが作成される場所
OUTPUT_FOLDER = "dataset/gt_split"

# 【分割比率】(学習用, 検証用, テスト用)
# この例では 8:1:1 (80%, 10%, 10%) に設定
RATIO = (0.8, 0.1, 0.1) 
# --- ---

def main():
    """
    メインの処理関数
    """
    if not os.path.isdir(INPUT_FOLDER):
        print(f"エラー: 入力元フォルダが見つかりません - {INPUT_FOLDER}")
        return

    print(f"'{INPUT_FOLDER}'フォルダを以下の比率で分割します...")
    print(f"Train: {int(RATIO[0]*100)}%")
    print(f"Validation: {int(RATIO[1]*100)}%")
    print(f"Test: {int(RATIO[2]*100)}%")

    try:
        # 実行
        # group_prefix=None を指定することで、クラスごとのサブフォルダが作られないようにする
        splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER, seed=1337, ratio=RATIO, group_prefix=None)
        
        print(f"\nデータセットの分割が完了しました。出力先: {OUTPUT_FOLDER}")
        print("出力先の各フォルダ (train, val, test) の中に、分割された画像が入っています。")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("入力フォルダ内に画像ファイルが存在するか確認してください。")


if __name__ == "__main__":
    main()
