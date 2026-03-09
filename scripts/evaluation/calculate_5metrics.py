import os
import re
import csv
import math
import torch
import lpips
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_sk
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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
    _, ssim_map = ssim_sk(img_true_gray, img_test_gray, full=True, data_range=255)

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
    ssim_value = ssim_sk(img_true_gray, img_test_gray, data_range=255)
    return float(ssim_value)


def normalize_stem(fname):
    """
    ファイル名のstemを正規化して突合しやすくする。
    座標情報(_X..._Y...)は保持し、指定された損傷サフィックスのみを除去する。
    """
    import re
    stem = os.path.splitext(os.path.basename(fname))[0]

    # 1. 指定された損傷サフィックスを削除（複数回適用して全て除去）
    # (_Transparent_Stain を _Stain より先に判定させるため、長い順に記述しています)
    while True:
        old_stem = stem
        stem = re.sub(r"(_Transparent_Stain|_Ghosting|_Scratch|_Missing|_Stain)$", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"(_restored|_pred|_out)$", "", stem, flags=re.IGNORECASE)
        if stem == old_stem:
            break

    # 2. 末尾のアンダースコアやハイフンを削除
    stem = stem.rstrip('_-')

    return stem


def build_index(dir_path):
    """
    ディレクトリ内の画像をstem正規化でインデックス化
    normalized_stem -> filepath のマッピングを作成
    """
    index = {}
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            full_path = os.path.join(dir_path, f)
            key = normalize_stem(f)
            if key in index:
                # 重複する場合は警告（デバッグ用）
                # print(f"Warning: Duplicate key '{key}' for files: {os.path.basename(index[key])} and {f}")
                pass
            else:
                index[key] = full_path
    return index


def count_parameters(model):
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters())


def _cpu_worker(key_paths):
    """
    並列ワーカー: CPU で PSNR / SSIM / Masked 指標を計算して返す。
    LPIPS は GPU バッチで別途計算するため含まない。
    引数: (key, path_gt, path_pred, path_mask)
    戻り値: (key, score_psnr, score_ssim, score_masked_psnr, score_masked_ssim) or None
    """
    key, path_gt, path_pred, path_mask = key_paths
    try:
        cv_gt   = cv2.imread(path_gt)
        cv_pred = cv2.imread(path_pred)
        cv_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
        if cv_gt is None or cv_pred is None or cv_mask is None:
            return None

        score_psnr = calculate_psnr(cv_gt, cv_pred)
        score_ssim = calculate_ssim(cv_gt, cv_pred)
        score_masked_psnr, score_masked_ssim = calculate_masked_metrics(cv_gt, cv_pred, cv_mask)

        if np.isnan(score_masked_psnr) or np.isnan(score_masked_ssim):
            return None

        return (key, score_psnr, score_ssim, score_masked_psnr, score_masked_ssim)
    except Exception:
        return None


def calculate_flops(model, input_size=(1, 4, 128, 128)):
    """
    FLOPsを計算（thopまたはfvcore使用）
    """
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except ImportError:
        try:
            from fvcore.nn import FlopCountAnalysis
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size).to(device)
            flops = FlopCountAnalysis(model, dummy_input).total()
            params = count_parameters(model)
            return flops, params
        except ImportError:
            print("Warning: thop or fvcore not installed. FLOPs calculation skipped.")
            return None, None


def calculate_all_metrics(dir_gt, dir_pred, dir_mask, output_csv, args=None, use_gpu=True, model=None):
    """
    ディレクトリ内の画像を比較し、LPIPS, Masked PSNR, Masked SSIMを計算してCSVに出力
    ファイル名の正規化を行い、損傷サフィックスの有無に関わらずマッチング
    """
    
    # 1. LPIPSモデルの準備
    print("LPIPSモデルを読み込んでいます...", flush=True)
    loss_fn = lpips.LPIPS(net='alex')
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    loss_fn.to(device)
    loss_fn.eval()


    # 2. インデックスを構築
    print("ファイルインデックスを構築中...")
    gt_index = build_index(dir_gt)
    pred_index = build_index(dir_pred)
    mask_index = build_index(dir_mask)
    
    # 共通のキー（正規化されたstem）を取得
    common_keys = sorted(set(gt_index.keys()) & set(pred_index.keys()) & set(mask_index.keys()))
    
    # デバッグ: 不一致ファイルを確認
    gt_only = set(gt_index.keys()) - set(pred_index.keys())
    pred_only = set(pred_index.keys()) - set(gt_index.keys())
    mask_only = set(mask_index.keys()) - set(gt_index.keys())
    
    if gt_only or pred_only or mask_only:
        print(f"\n--- ファイル不一致の詳細 ---")
        if gt_only:
            print(f"GTのみに存在: {len(gt_only)}枚")
            # 最初の5個を表示
            for i, key in enumerate(list(gt_only)[:5]):
                print(f"  例: {key} -> {os.path.basename(gt_index[key])}")
        if pred_only:
            print(f"Predのみに存在: {len(pred_only)}枚")
            for i, key in enumerate(list(pred_only)[:5]):
                print(f"  例: {key} -> {os.path.basename(pred_index[key])}")
        if mask_only:
            print(f"Maskのみに存在: {len(mask_only)}枚")
            for i, key in enumerate(list(mask_only)[:5]):
                print(f"  例: {key} -> {os.path.basename(mask_index[key])}")
        print()

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

    # 3. CPU 指標計算（ThreadPoolExecutor: wandb.initと共存可能、forkより安全）
    args_list = [
        (key, gt_index[key], pred_index[key], mask_index[key])
        for key in common_keys
    ]

    n_workers = min(cpu_count(), 16)
    print(f"CPU 指標を {n_workers} スレッドで並列計算中...")
    cpu_results = {}  # key -> (psnr, ssim, m_psnr, m_ssim)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_cpu_worker, item): item[0] for item in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="CPU metrics"):
            ret = future.result()
            if ret is not None:
                key, psnr, s, mp_val, ms = ret
                cpu_results[key] = (psnr, s, mp_val, ms)

    valid_keys = [k for k in common_keys if k in cpu_results]
    print(f"CPU 計算完了: {len(valid_keys)}/{len(common_keys)} 枚")

    # 4. GPU バッチ LPIPS 計算
    BATCH = 256
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    lpips_scores = {}  # key -> float
    print(f"GPU バッチ LPIPS 計算中 (batch={BATCH})...")
    for i in tqdm(range(0, len(valid_keys), BATCH), desc="LPIPS batch"):
        batch_keys = valid_keys[i : i + BATCH]
        batch_gt, batch_pred = [], []
        for key in batch_keys:
            try:
                gt_pil   = Image.open(gt_index[key]).convert('RGB')
                pred_pil = Image.open(pred_index[key]).convert('RGB')
                if pred_pil.size != gt_pil.size:
                    pred_pil = pred_pil.resize(gt_pil.size, Image.BICUBIC)
                batch_gt.append(transform(gt_pil))
                batch_pred.append(transform(pred_pil))
            except Exception:
                batch_gt.append(None)
                batch_pred.append(None)

        # None を除いてテンソル化
        valid_idx = [j for j, t in enumerate(batch_gt) if t is not None]
        if not valid_idx:
            continue
        t_gt   = torch.stack([batch_gt[j]   for j in valid_idx]).to(device)
        t_pred = torch.stack([batch_pred[j] for j in valid_idx]).to(device)

        with torch.no_grad():
            scores = loss_fn(t_gt, t_pred)  # (N, 1, 1, 1)
        scores = scores.squeeze().cpu()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        for j, idx in enumerate(valid_idx):
            lpips_scores[batch_keys[idx]] = float(scores[j])

    # 5. 結果集計
    results = []
    total_lpips = total_psnr = total_masked_psnr = total_ssim = total_masked_ssim = 0.0
    count = 0

    for key in valid_keys:
        if key not in lpips_scores:
            continue
        psnr, s, mp, ms = cpu_results[key]
        lp = lpips_scores[key]
        filename = os.path.basename(pred_index[key])
        results.append([filename, lp, psnr, mp, s, ms])
        total_lpips       += lp
        total_psnr        += psnr
        total_masked_psnr += mp
        total_ssim        += s
        total_masked_ssim += ms
        count += 1

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

    # 5. モデルの計算量を取得
    flops = params = None
    if model is not None:
        print("\nモデルの計算量を計算中...")
        flops, params = calculate_flops(model)
        if flops is not None and params is not None:
            print(f"FLOPs: {flops / 1e9:.2f} G")
            print(f"Parameters: {params / 1e6:.2f} M")

    # 6. CSV保存
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
            
            # 計算量情報
            if flops is not None and params is not None:
                writer.writerow([])
                writer.writerow(['Model Stats', 'FLOPs (G)', 'Parameters (M)'])
                writer.writerow(['', flops / 1e9, params / 1e6])
            
        print(f"結果を {output_csv} に保存しました。")
        
    except Exception as e:
        print(f"CSV書き込みエラー: {e}")

    # 7. wandbログ保存
    if args and args.use_wandb:
        import wandb
        print("wandbに結果を送信中...")
        
        # デバッグ用: wandbが初期化されているか確認
        if wandb.run is None:
            print("Warning: wandb.run is None. wandb.init()が呼ばれていない可能性があります。")
        else:
            print(f"wandb run: {wandb.run.name}, project: {wandb.run.project}")
            
        wandb_results = {
            "eval/LPIPS": avg_lpips,
            "eval/PSNR": avg_psnr,
            "eval/Masked_PSNR": avg_masked_psnr,
            "eval/SSIM": avg_ssim,
            "eval/Masked_SSIM": avg_masked_ssim,
            "eval/Count": count
        }
        
        # 計算量情報を追加
        if flops is not None and params is not None:
            wandb_results["model/FLOPs_G"] = flops / 1e9
            wandb_results["model/Parameters_M"] = params / 1e6
        
        # テーブル作成（詳細データ）
        table = wandb.Table(columns=['Filename', 'LPIPS', 'PSNR', 'Masked_PSNR', 'SSIM', 'Masked_SSIM'])
        for res in results:
            table.add_data(*res)
        
        wandb_results["eval/detailed_results"] = table
        
        # ログをまとめて送信
        wandb.log(wandb_results)

        # サマリーを更新（Runsテーブルに表示）
        for key, value in wandb_results.items():
            if key != "eval/detailed_results":  # テーブルは除外
                wandb.run.summary[key] = value
            
        print("wandb送信完了")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='画像修復の評価指標 (LPIPS, PSNR, SSIM) を計算')
    
    # パス設定
    parser.add_argument('--gt_dir', type=str, 
                        default="/home/imoto/Kuzushiji_Restoration/data/full_padded/gt/test",
                        help='正解画像 (Ground Truth) のフォルダ')
    parser.add_argument('--pred_dir', type=str, 
                        default="outputs/mprnet_gtmask_charbpercep",
                        help='修復画像 (Restored/Output) のフォルダ')
    parser.add_argument('--mask_dir', type=str, 
                        default="/home/imoto/Kuzushiji_Restoration/data/full_padded/gt_mask/test",
                        help='文字領域マスク (Mask) のフォルダ')
    parser.add_argument('--output_csv', type=str, 
                        default="outputs/mprnet_gtmask_charbpercep/evaluation_mprnet_gtmask_charbpercep.csv",
                        help='出力CSVファイルパス')
    
    # wandb設定
    parser.add_argument('--use_wandb', action='store_true', help='wandbに結果を記録する')
    parser.add_argument('--wandb_project', type=str, default='Kuzushiji_Restoration', help='wandbプロジェクト名')
    parser.add_argument('--wandb_name', type=str, default="eval_mprnet_gtmask_charbpercep", help='wandb run名 (未指定なら自動生成)')
    parser.add_argument('--wandb_job_type', type=str, default='evaluation', help='wandbジョブタイプ')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='wandbタグ')

    args = parser.parse_args()

    # ディレクトリ確認を先に行う（wandb.init より前）
    if not (os.path.exists(args.gt_dir) and os.path.exists(args.pred_dir) and os.path.exists(args.mask_dir)):
        print("エラー: 指定されたディレクトリが見つかりません。パスを確認してください。")
        if not os.path.exists(args.gt_dir):   print(f"Missing: {args.gt_dir}")
        if not os.path.exists(args.pred_dir): print(f"Missing: {args.pred_dir}")
        if not os.path.exists(args.mask_dir): print(f"Missing: {args.mask_dir}")
        import sys; sys.exit(1)

    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # wandb初期化（ディレクトリ確認後）
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            job_type=args.wandb_job_type,
            tags=args.wandb_tags,
            config=vars(args).copy(),
            settings=wandb.Settings(start_method="thread"),
        )

    calculate_all_metrics(args.gt_dir, args.pred_dir, args.mask_dir, args.output_csv, args=args, model=None, use_gpu=False)

    if args.use_wandb:
        import wandb
        wandb.finish()