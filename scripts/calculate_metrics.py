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

def calculate_all_metrics(dir_gt, dir_pred, dir_mask, output_csv, use_gpu=True):
    """
    ディレクトリ内の画像を比較し、LPIPS, Masked PSNR, Masked SSIMを計算してCSVに出力
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

    # 2. ファイルリスト取得
    files = [f for f in os.listdir(dir_gt) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    files.sort()

    results = []
    total_lpips = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    print(f"計算開始: {len(files)} 枚の画像を処理します...")

    # 3. ループ処理
    for filename in tqdm(files):
        path_gt = os.path.join(dir_gt, filename)
        path_pred = os.path.join(dir_pred, filename)
        path_mask = os.path.join(dir_mask, filename)

        # 必要なファイルが揃っているか確認
        if not os.path.exists(path_pred):
            print(f"Skip: {filename} (修復画像なし)")
            continue
        if not os.path.exists(path_mask):
            # マスクがない場合、LPIPSだけ計算するかスキップするか選べますが、今回はスキップします
            print(f"Skip: {filename} (マスク画像なし)")
            continue

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

            score_psnr, score_ssim = calculate_masked_metrics(cv_gt, cv_pred, cv_mask)

            # NaN ガード（必要ならスキップまたは 0 代入）
            if np.isnan(score_psnr) or np.isnan(score_ssim):
                print(f"Warning: empty mask for {filename}, skipping masked metrics")
                continue

            # 結果を保存
            results.append([filename, score_lpips, score_psnr, score_ssim])
            
            total_lpips += score_lpips
            total_psnr += score_psnr
            total_ssim += score_ssim
            count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 4. 平均計算と出力
    if count > 0:
        avg_lpips = total_lpips / count
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        
        print(f"\n--- 処理完了 ---")
        print(f"処理枚数: {count}")
        print(f"平均 LPIPS (Lower is better): {avg_lpips:.4f}")
        print(f"平均 Masked PSNR (Higher is better): {avg_psnr:.4f}")
        print(f"平均 Masked SSIM (Higher is better): {avg_ssim:.4f}")
    else:
        print("計算可能な画像がありませんでした。")
        avg_lpips = avg_psnr = avg_ssim = 0

    # 5. CSV保存
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ヘッダー
            writer.writerow(['Filename', 'LPIPS', 'Masked_PSNR', 'Masked_SSIM'])
            
            # データ
            writer.writerows(results)
            
            # 平均行
            writer.writerow([])
            writer.writerow(['Average', avg_lpips, avg_psnr, avg_ssim])
            
        print(f"結果を {output_csv} に保存しました。")
        
    except Exception as e:
        print(f"CSV書き込みエラー: {e}")

if __name__ == "__main__":
    # ==========================================
    # 設定: ここに実際のディレクトリパスを入力してください
    # ==========================================
    
    # 1. 正解画像 (Ground Truth) のフォルダ
    GT_DIR = "dataset_final_hiragana/gt/test"
    
    # 2. 修復画像 (Restored/Output) のフォルダ
    PRED_DIR = "results/restored_net_g_140000/restored"
    
    # 3. 文字領域マスク (Mask) のフォルダ (文字=白(255), 背景=黒(0))
    MASK_DIR = "dataset_final_hiragana/mask_gt/test"
    
    # 4. 出力ファイル名
    OUTPUT_CSV = "evaluation_revaluate_withmask.csv"
    
    # ==========================================
    
    if os.path.exists(GT_DIR) and os.path.exists(PRED_DIR) and os.path.exists(MASK_DIR):
        calculate_all_metrics(GT_DIR, PRED_DIR, MASK_DIR, OUTPUT_CSV)
    else:
        print("エラー: 指定されたディレクトリが見つかりません。パスを確認してください。")