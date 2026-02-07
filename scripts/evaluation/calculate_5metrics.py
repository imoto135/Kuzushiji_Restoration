import os
import csv
import torch
import lpips
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def calculate_masked_metrics(img_true, img_test, mask):
    """
    Masked PSNR と Masked SSIM を計算する関数（安全化・型変換・空マスク処理追加）
    
    Args:
        img_true (np.array): 正解画像 (BGR or Grayscale, 0-255)
        img_test (np.array): 修復画像 (BGR or Grayscale, 0-255)
        mask (np.array): マスク画像 (Grayscale, 0-255, 文字=255, 背景=0)
    
    Returns:
        tuple: (masked_psnr, masked_ssim) -- mask が空なら (np.nan, np.nan)
    """
    # サイズ合わせ（既存）
    if img_true.shape != img_test.shape:
        h, w = img_true.shape[:2]
        img_test = cv2.resize(img_test, (w, h))

    if mask.shape[:2] != img_true.shape[:2]:
        h, w = img_true.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # マスクをブール値に変換 (閾値128)
    mask_bool = (mask > 128)
    if mask_bool.sum() == 0:
        # マスク領域が空なら計算不可（NaN を返す）
        return float('nan'), float('nan')

    # --- Masked PSNR ---
    # 明示的に float にキャストして差分を計算（オーバーフロー回避）
    true_f = img_true.astype(np.float32)
    test_f = img_test.astype(np.float32)

    # boolean indexing: 2D mask applied to HxWxC -> returns (N, C)
    true_pixels = true_f[mask_bool]
    test_pixels = test_f[mask_bool]

    # MSE（チャンネル混在での MSE をそのまま平均）
    mse = np.mean((true_pixels - test_pixels) ** 2)
    if mse == 0 or np.isnan(mse):
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    # --- Masked SSIM ---
    # グレースケール化して SSIM マップを計算（既存）
    if len(img_true.shape) == 3:
        img_true_gray = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        img_test_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    else:
        img_true_gray = img_true
        img_test_gray = img_test

    # skimage の ssim で full map を取得
    _, ssim_map = ssim(img_true_gray, img_test_gray, full=True, data_range=255)

    # マスク領域内の平均を取る（mask_bool は 2D）
    try:
        masked_ssim = float(ssim_map[mask_bool].mean())
    except Exception:
        masked_ssim = float('nan')

    return psnr, masked_ssim


def calculate_psnr(img_true, img_test):
    """
    通常のPSNRを計算（画像全体）
    
    Args:
        img_true (np.array): 正解画像 (BGR, 0-255)
        img_test (np.array): 修復画像 (BGR, 0-255)
    
    Returns:
        float: PSNR値
    """
    # サイズ合わせ
    if img_true.shape != img_test.shape:
        h, w = img_true.shape[:2]
        img_test = cv2.resize(img_test, (w, h))
    
    # MSE計算
    mse = np.mean((img_true.astype(np.float32) - img_test.astype(np.float32)) ** 2)
    
    if mse == 0:
        return 100.0
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img_true, img_test):
    """
    通常のSSIMを計算（画像全体）
    
    Args:
        img_true (np.array): 正解画像 (BGR, 0-255)
        img_test (np.array): 修復画像 (BGR, 0-255)
    
    Returns:
        float: SSIM値
    """
    # サイズ合わせ
    if img_true.shape != img_test.shape:
        h, w = img_true.shape[:2]
        img_test = cv2.resize(img_test, (w, h))
    
    # グレースケール化
    if len(img_true.shape) == 3:
        img_true_gray = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        img_test_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    else:
        img_true_gray = img_true
        img_test_gray = img_test
    
    # SSIM計算
    ssim_value = ssim(img_true_gray, img_test_gray, data_range=255)
    return float(ssim_value)


def normalize_stem(fname):
    """
    ファイル名のstemを正規化して突合しやすくする。
    座標情報(_X..._Y...)は保持し、指定された損傷サフィックスのみを除去する。
    """
    import re
    stem = os.path.splitext(os.path.basename(fname))[0]

    # 1. 指定された損傷サフィックスを削除
    # (_Transparent_Stain を _Stain より先に判定させるため、長い順に記述しています)
    stem = re.sub(r"(_Transparent_Stain|_Ghosting|_Scratch|_Missing|_Stain)$", "", stem, flags=re.IGNORECASE)

    # 2. 推論時に付与されがちなその他のサフィックスも除去
    stem = re.sub(r"(_restored|_pred|_out)$", "", stem, flags=re.IGNORECASE)

    # 3. 末尾に残ったアルファベット1文字などのゴミ（_a, -bなど）があれば除去
    # (座標情報 _Y1234 などを消さないよう、数字を含まないものだけに限定)
    stem = re.sub(r"[_-][A-Za-z]+$", "", stem)

    return stem


def build_index(dir_path):
    """
    ディレクトリ内の画像をstem正規化でインデックス化
    normalized_stem -> filepath のマッピングを作成
    """
    index = {}
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(dir_path, f)
            key = normalize_stem(f)
            if key not in index:
                index[key] = full_path
    return index


def calculate_all_metrics(dir_gt, dir_pred, dir_mask, output_csv, args=None, use_gpu=True):
    """
    ディレクトリ内の画像を比較し、LPIPS, Masked PSNR, Masked SSIMを計算してCSVに出力
    ファイル名の正規化を行い、損傷サフィックスの有無に関わらずマッチング
    """
    
    # 1. LPIPSモデルの準備
    print("LPIPSモデルを読み込んでいます...")
    loss_fn = lpips.LPIPS(net='alex')
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    loss_fn.to(device)
    loss_fn.eval()

    # LPIPS用画像前処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 2. インデックスを構築
    print("ファイルインデックスを構築中...")
    gt_index = build_index(dir_gt)
    pred_index = build_index(dir_pred)
    mask_index = build_index(dir_mask)
    
    # 共通のキー（正規化されたstem）を取得
    common_keys = sorted(set(gt_index.keys()) & set(pred_index.keys()) & set(mask_index.keys()))
    
    if not common_keys:
        print("エラー: マッチするファイルが見つかりませんでした。")
        print(f"  GT: {len(gt_index)} 枚")
        print(f"  Pred: {len(pred_index)} 枚")
        print(f"  Mask: {len(mask_index)} 枚")
        return

    results = []
    total_lpips = 0.0
    total_psnr = 0.0
    total_masked_psnr = 0.0
    total_ssim = 0.0
    total_masked_ssim = 0.0
    count = 0

    print(f"計算開始: {len(common_keys)} 枚の画像を処理します...")
    print(f"  GT: {len(gt_index)} 枚, Pred: {len(pred_index)} 枚, Mask: {len(mask_index)} 枚")

    # 3. ループ処理
    for key in tqdm(common_keys, desc="Processing"):
        path_gt = gt_index[key]
        path_pred = pred_index[key]
        path_mask = mask_index[key]
        
        filename = os.path.basename(path_pred)  # 出力用にpredのファイル名を使用

        try:
            # --- LPIPS計算 (PyTorch/PIL使用) ---
            img_gt_pil = Image.open(path_gt).convert('RGB')
            img_pred_pil = Image.open(path_pred).convert('RGB')
            # make sure prediction has same size as GT for LPIPS
            if img_pred_pil.size != img_gt_pil.size:
                img_pred_pil = img_pred_pil.resize(img_gt_pil.size, Image.BICUBIC)
 
            tensor_gt = transform(img_gt_pil).unsqueeze(0).to(device)
            tensor_pred = transform(img_pred_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                score_lpips = loss_fn(tensor_gt, tensor_pred).item()

            # --- Masked PSNR/SSIM計算 (OpenCV/Numpy使用) ---
            # 画像を読み込み (BGR 0-255)
            cv_gt = cv2.imread(path_gt)
            cv_pred = cv2.imread(path_pred)
            # マスクを読み込み (グレースケール)
            cv_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)

            # --- 通常のPSNR/SSIM計算（画像全体） ---
            score_psnr = calculate_psnr(cv_gt, cv_pred)
            score_ssim = calculate_ssim(cv_gt, cv_pred)

            # --- Masked PSNR/SSIM計算（マスク領域のみ） ---
            score_masked_psnr, score_masked_ssim = calculate_masked_metrics(cv_gt, cv_pred, cv_mask)

            # NaN ガード（必要ならスキップまたは 0 代入）
            if np.isnan(score_masked_psnr) or np.isnan(score_masked_ssim):
                print(f"Warning: empty mask for {filename}, skipping masked metrics")
                continue

            # 結果を保存 [Filename, LPIPS, PSNR, Masked_PSNR, SSIM, Masked_SSIM]
            results.append([filename, score_lpips, score_psnr, score_masked_psnr, score_ssim, score_masked_ssim])
            
            total_lpips += score_lpips
            total_psnr += score_psnr
            total_masked_psnr += score_masked_psnr
            total_ssim += score_ssim
            total_masked_ssim += score_masked_ssim
            count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 4. 平均計算と出力
    if count > 0:
        avg_lpips = total_lpips / count
        avg_psnr = total_psnr / count
        avg_masked_psnr = total_masked_psnr / count
        avg_ssim = total_ssim / count
        avg_masked_ssim = total_masked_ssim / count
        
        print(f"\n--- 処理完了 ---")
        print(f"処理枚数: {count}")
        print(f"平均 LPIPS (Lower is better): {avg_lpips:.4f}")
        print(f"平均 PSNR (Higher is better): {avg_psnr:.4f}")
        print(f"平均 Masked PSNR (Higher is better): {avg_masked_psnr:.4f}")
        print(f"平均 SSIM (Higher is better): {avg_ssim:.4f}")
        print(f"平均 Masked SSIM (Higher is better): {avg_masked_ssim:.4f}")
    else:
        print("計算可能な画像がありませんでした。")
        avg_lpips = avg_psnr = avg_masked_psnr = avg_ssim = avg_masked_ssim = 0

    # 5. CSV保存
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ヘッダー
            writer.writerow(['Filename', 'LPIPS', 'PSNR', 'Masked_PSNR', 'SSIM', 'Masked_SSIM'])
            
            # データ
            writer.writerows(results)
            
            # 平均行
            writer.writerow([])
            writer.writerow(['Average', avg_lpips, avg_psnr, avg_masked_psnr, avg_ssim, avg_masked_ssim])
            
        print(f"結果を {output_csv} に保存しました。")
        
    except Exception as e:
        print(f"CSV書き込みエラー: {e}")

    # 6. wandbログ保存
    if args and args.use_wandb:
        print("wandbに結果を送信中...")
        wandb_results = {
            "Average/LPIPS": avg_lpips,
            "Average/PSNR": avg_psnr,
            "Average/Masked_PSNR": avg_masked_psnr,
            "Average/SSIM": avg_ssim,
            "Average/Masked_SSIM": avg_masked_ssim,
            "Count": count
        }
        
        # テーブル作成（詳細データ）
        table = wandb.Table(columns=['Filename', 'LPIPS', 'PSNR', 'Masked_PSNR', 'SSIM', 'Masked_SSIM'])
        for res in results:
            table.add_data(*res)
        
        # ログをまとめて送信
        log_data = wandb_results.copy()
        log_data["Evaluation_Results"] = table
        wandb.log(log_data)

        # Runsテーブル（一覧）に表示されるようにsummaryを明示的に更新
        for key, value in wandb_results.items():
            # "Average/" プレフィックスがついている場合、プレフィックスなしでも登録しておくと表で見やすい場合があります
            # ここでは元のキー(Average/...)で登録します
            wandb.run.summary[key] = value
            
        wandb.finish()
        print("wandb送信完了")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='画像修復の評価指標 (LPIPS, PSNR, SSIM) を計算')
    
    # パス設定
    parser.add_argument('--gt_dir', type=str, 
                        default="/home/imoto/Kuzushiji_Restoration/hiragana_fulldataset_5stain/gt/test",
                        help='正解画像 (Ground Truth) のフォルダ')
    parser.add_argument('--pred_dir', type=str, 
                        default="/home/imoto/Kuzushiji_Restoration/outputs/nafnet_mask_MaskMorph",
                        help='修復画像 (Restored/Output) のフォルダ')
    parser.add_argument('--mask_dir', type=str, 
                        default="/home/imoto/Kuzushiji_Restoration/hiragana_fulldataset_5stain/gt_mask/test",
                        help='文字領域マスク (Mask) のフォルダ')
    parser.add_argument('--output_csv', type=str, 
                        default="/home/imoto/Kuzushiji_Restoration/outputs/nafnet_mask_charbpercep/evaluation_nafnet_mask_MaskMorph.csv",
                        help='出力CSVファイルパス')
    
    # wandb設定
    parser.add_argument('--use_wandb', action='store_true', help='wandbに結果を記録する')
    parser.add_argument('--wandb_project', type=str, default='Kuzushiji_Restoration', help='wandbプロジェクト名')
    parser.add_argument('--wandb_name', type=str, default='eval_nafnet_mask_MaskMorph', help='wandb run名 (未指定なら自動生成)')
    parser.add_argument('--wandb_job_type', type=str, default='evaluation', help='wandbジョブタイプ')

    args = parser.parse_args()

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, job_type=args.wandb_job_type, config=vars(args))

    if os.path.exists(args.gt_dir) and os.path.exists(args.pred_dir) and os.path.exists(args.mask_dir):
        # 出力ディレクトリの作成
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        
        # calculate_all_metrics に args を渡せるように少し修正が必要だが、
        # ここでは calculate_all_metrics で wandb を直接参照せず、戻り値や引数で渡す形にするか、
        # グローバルな args を参照する形にするか... 
        # シンプルに calculate_all_metrics の引数を修正せず、関数内から args にアクセスするのはdirtyなので、
        # calculate_all_metrics の引数を変更するのが筋だが、今回は関数定義も変更する。
        # 上記の replace ブロックですでに関数末尾に wandb 処理を追加しているので、
        # 関数定義のシグネチャ変更も必要。
        pass # 下記の関数定義変更で対応
    else:
        print("エラー: 指定されたディレクトリが見つかりません。パスを確認してください。")
        if not os.path.exists(args.gt_dir): print(f"Missing: {args.gt_dir}")
        if not os.path.exists(args.pred_dir): print(f"Missing: {args.pred_dir}")
        if not os.path.exists(args.mask_dir): print(f"Missing: {args.mask_dir}")
        import sys; sys.exit(1)

    calculate_all_metrics(args.gt_dir, args.pred_dir, args.mask_dir, args.output_csv, args=args)