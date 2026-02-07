import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# ファイルパス設定
# ==========================================
# 1. 正解マスク (GT Mask) を使用した結果
PATH_GT = 'outputs/restored_gt/eval_results.csv'
# 2. マスクなし (Baseline) の結果
PATH_NOMASK = 'outputs/restored_nomask_test/eval_results.csv'
# 3. 推測マスク (Pred Mask) を使用した結果 【追加】
# ※実際のパスに合わせて変更してください
PATH_PRED = 'outputs/restored_pred_mask/eval_results.csv' 

# ==========================================
# 関数定義
# ==========================================
def extract_damage_type(suffix_str):
    """ファイル名から損傷タイプを抽出する"""
    if not isinstance(suffix_str, str): return 'Other'
    if 'Stain' in suffix_str: return 'Stain (汚れ)'
    if 'Missing' in suffix_str: return 'Missing (欠損)'
    if 'Ghosting' in suffix_str: return 'Bleed-through (裏移り)'
    if 'Scratch' in suffix_str: return 'Scratch (傷)'
    return 'Other'

def load_and_process(path, method_name):
    """CSVを読み込み、前処理を行う"""
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    
    df = pd.read_csv(path)
    # 損傷タイプ列を作成
    df['Damage Type'] = df['損傷サフィックスの種類'].apply(extract_damage_type)
    # 手法名を設定
    df['Method'] = method_name
    return df

# ==========================================
# メイン処理
# ==========================================

# 1. データの読み込み
df_gt = load_and_process(PATH_GT, 'Ours (Ideal)')
df_nomask = load_and_process(PATH_NOMASK, 'Baseline')
df_pred = load_and_process(PATH_PRED, 'Ours (Real)') # 【追加】

# データフレームの結合 (Noneを除外)
dfs = [d for d in [df_gt, df_nomask, df_pred] if d is not None]
if not dfs:
    raise FileNotFoundError("CSVファイルが見つかりませんでした。パスを確認してください。")

df_all = pd.concat(dfs)

# 評価指標のリスト
metrics = ['psnr', 'masked psnr', 'ssim', 'masked ssim', 'lpips']

# -------------------------------------------------------
# 2. 平均スコアの算出
# -------------------------------------------------------
print("\n" + "="*50)
print("1. 全体平均スコア (Overall Average)")
print("="*50)
# 手法ごとの平均
overall_mean = df_all.groupby('Method')[metrics].mean()
print(overall_mean)

print("\n" + "="*50)
print("2. 損傷タイプ別平均スコア (Average per Damage Type)")
print("="*50)
# 損傷タイプ x 手法ごとの平均
damage_mean = df_all.groupby(['Damage Type', 'Method'])[metrics].mean().unstack()
print(damage_mean)

# -------------------------------------------------------
# 3. 改善幅の計算 (Difference)
# -------------------------------------------------------
# 計算用にインデックスをセット
df_nomask_idx = df_nomask.set_index('画像名')[metrics]
df_gt_idx = df_gt.set_index('画像名')[metrics]

# Ours (Real) が存在する場合のみ計算
if df_pred is not None:
    df_pred_idx = df_pred.set_index('画像名')[metrics]
    
    # 共通の画像のみで計算（念のため）
    common_indices_real = df_pred_idx.index.intersection(df_nomask_idx.index)
    
    # --- Ours (Real) vs Baseline ---
    diff_real = df_pred_idx.loc[common_indices_real] - df_nomask_idx.loc[common_indices_real]
    
    print("\n" + "="*50)
    print("3-1. 改善幅: Ours (Real) - Baseline")
    print("   (正の値 = Ours(Real)が勝っている)")
    print("="*50)
    print(diff_real.mean())
    
    # 損傷タイプごとの改善幅 (Ours Real)
    # 損傷タイプ情報を結合して集計
    diff_real_type = diff_real.copy()
    diff_real_type['Damage Type'] = df_nomask.set_index('画像名').loc[common_indices_real, 'Damage Type']
    print("\n[損傷タイプ別改善幅: Ours (Real) - Baseline]")
    print(diff_real_type.groupby('Damage Type')[metrics].mean())


# --- Ours (Ideal) vs Baseline ---
common_indices_ideal = df_gt_idx.index.intersection(df_nomask_idx.index)
diff_ideal = df_gt_idx.loc[common_indices_ideal] - df_nomask_idx.loc[common_indices_ideal]

print("\n" + "="*50)
print("3-2. 改善幅 (アッパーバウンド): Ours (Ideal) - Baseline")
print("   (正の値 = Ours(Ideal)が勝っている)")
print("="*50)
print(diff_ideal.mean())

# 損傷タイプごとの改善幅 (Ours Ideal)
diff_ideal_type = diff_ideal.copy()
diff_ideal_type['Damage Type'] = df_nomask.set_index('画像名').loc[common_indices_ideal, 'Damage Type']
print("\n[損傷タイプ別改善幅: Ours (Ideal) - Baseline]")
print(diff_ideal_type.groupby('Damage Type')[metrics].mean())

# ==========================================
# グラフ化 (オプション)
# ==========================================
# Masked PSNRの比較グラフを作成して保存
try:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_all, x='Damage Type', y='masked psnr', hue='Method', errorbar=None)
    plt.title('Comparison of Masked PSNR by Damage Type')
    plt.ylabel('Masked PSNR (dB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('masked_psnr_comparison.png')
    print("\nグラフを 'masked_psnr_comparison.png' に保存しました。")
except Exception as e:
    print(f"\nグラフ作成エラー: {e}")