import os
import glob
import pandas as pd
import re
from collections import Counter

def count_chars_in_dataset(dataset_root, output_csv="dataset_counts.csv"):
    """
    データセットのディレクトリ構造から文字ごとの枚数を集計し、CSVに出力する。
    ディレクトリ構造の想定:
    dataset_root/
        train/ (または train/gt/)
        val/   (または val/gt/)
        test/  (または test/gt/)
    """
    
    # Unicodeからひらがなへのマッピング（表示用）
    char_map = {
        'U+3042': 'あ', 'U+3044': 'い', 'U+3046': 'う', 'U+3048': 'え', 'U+304A': 'お',
        'U+304B': 'か', 'U+304D': 'き', 'U+304F': 'く', 'U+3051': 'け', 'U+3053': 'こ',
        'U+3055': 'さ', 'U+3057': 'し', 'U+3059': 'す', 'U+305B': 'せ', 'U+305D': 'そ',
        'U+305F': 'た', 'U+3061': 'ち', 'U+3064': 'つ', 'U+3066': 'て', 'U+3068': 'と',
        'U+306A': 'な', 'U+306B': 'に', 'U+306C': 'ぬ', 'U+306D': 'ね', 'U+306E': 'の',
        'U+306F': 'は', 'U+3072': 'ひ', 'U+3075': 'ふ', 'U+3078': 'へ', 'U+307B': 'ほ',
        'U+307E': 'ま', 'U+307F': 'み', 'U+3080': 'む', 'U+3081': 'め', 'U+3082': 'も',
        'U+3084': 'や', 'U+3086': 'ゆ', 'U+3088': 'よ', 'U+3089': 'ら', 'U+308A': 'り',
        'U+308B': 'る', 'U+308C': 'れ', 'U+308D': 'ろ', 'U+308F': 'わ', 'U+3092': 'を',
        'U+3093': 'ん', 'U+304C': 'が', 'U+304E': 'ぎ', 'U+3050': 'ぐ', 'U+3052': 'げ',
        'U+3054': 'ご', 'U+3056': 'ざ', 'U+3058': 'じ', 'U+305E': 'ぞ', 'U+3060': 'だ',
        'U+3062': 'ぢ', 'U+3065': 'づ', 'U+3067': 'で', 'U+3069': 'ど', 'U+3070': 'ば',
        'U+3073': 'び', 'U+3076': 'ぶ', 'U+3079': 'べ', 'U+307C': 'ぼ', 'U+3071': 'ぱ',
        'U+3074': 'ぴ', 'U+3077': 'ぷ', 'U+307A': 'ぺ', 'U+307D': 'ぽ', 'U+309D': 'ゝ',
        'U+309E': 'ゞ', 'U+309F': 'より', 'U+3041': 'ぁ', 'U+3043': 'ぃ', 'U+3045': 'ぅ',
        'U+3047': 'ぇ', 'U+3049': 'ぉ', 'U+3063': 'っ', 'U+3083': 'ゃ', 'U+3085': 'ゅ',
        'U+3087': 'ょ'
    }

    splits = ["train", "val", "test"]
    extensions = ("*.png", "*.jpg", "*.jpeg")
    
    # 集計用辞書: {unicode: {split: count}}
    stats = {}

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        
        # サブディレクトリ(gtなど)がある場合にも対応
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(split_dir, "**", ext), recursive=True))
        
        print(f"Checking {split}: {len(files)} files found.")
        
        for fpath in files:
            fname = os.path.basename(fpath)
            # ファイル名から Unicode (U+XXXX) を抽出
            match = re.search(r'(U\+[0-9A-F]+)', fname)
            if match:
                unicode_id = match.group(1)
                if unicode_id not in stats:
                    stats[unicode_id] = {s: 0 for s in splits}
                stats[unicode_id][split] += 1

    # データを整形
    rows = []
    for uid, counts in stats.items():
        row = {
            "Unicode": uid,
            "Character": char_map.get(uid, "Unknown"),
            "Train": counts["train"],
            "Val": counts["val"],
            "Test": counts["test"],
            "Total": sum(counts.values())
        }
        rows.append(row)

    # DataFrameを作成して保存
    df = pd.DataFrame(rows)
    # Unicode順にソート
    df = df.sort_values("Unicode")
    
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n集計完了: {output_csv} に保存しました。")
    print(f"全クラス数: {len(df)}")
    print(f"全画像枚数: {df['Total'].sum()}")

if __name__ == "__main__":
    # データセットのルートディレクトリを指定してください
    target_dir = "hiragana_dataset/gt" 
    
    if os.path.exists(target_dir):
        count_chars_in_dataset(target_dir)
    else:
        print(f"エラー: ディレクトリ {target_dir} が見つかりません。")