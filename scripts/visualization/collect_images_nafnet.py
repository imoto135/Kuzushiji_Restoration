import os
import shutil
from pathlib import Path
import re

# ==========================================
# PATH SETTINGS (実際の環境に合わせて修正してください)
# ==========================================
SRC_GT      = "datasets/hiragana_fulldataset_5stain/gt"
SRC_INPUT   = "datasets/hiragana_fulldataset_5stain/lq/test"
SRC_NOMASK  = "outputs/nafnet_nomask"
SRC_PRED    = "outputs/nafnet_predmask"
SRC_GTMASK  = "outputs/nafnet_predmask"
SRC_MASK    = "outputs/nafnet_predmask"
SRC_GTM_IMG = "datasets/hiragana_fulldataset_5stain/gt_mask/test"

DEST_ROOT   = "sotsuron/collection_nafnet"

# ==========================================
# TARGET FILES (分析で選定した21枚)
# ==========================================
TARGET_FILES = [
    {"name": "U+3056_200017458_00008_1_X0805_Y1480_Missing.png", "cat": "Missing", "desc": "Best_PSNR"},
    {"name": "U+3070_200021712_00048_2_X1941_Y0960_Missing.png", "cat": "Missing", "desc": "Best_PSNR_2"},
    {"name": "U+3072_200019865_00048_2_X0607_Y1265_Missing.png", "cat": "Missing", "desc": "Best_LPIPS"},
    {"name": "U+3075_brsk005_038_X0320_Y1669_Missing.png",       "cat": "Missing", "desc": "Best_LPIPS_2"},
    {"name": "U+306A_umgy001_007_X0188_Y0782_Scratch.png",       "cat": "Scratch", "desc": "Best_PSNR_Teaser"},
    {"name": "U+307E_umgy010_027_X0187_Y1526_Scratch.png",       "cat": "Scratch", "desc": "Best_PSNR_2"},
    {"name": "U+3072_100241706_00008_2_X1889_Y1360_Scratch.png", "cat": "Scratch", "desc": "Best_LPIPS"},
    {"name": "U+3089_100241706_00005_2_X0940_Y2682_Scratch.png", "cat": "Scratch", "desc": "Best_LPIPS_2"},
    {"name": "U+305B_200014685_00021_1_X1444_Y1264_Stain.png",   "cat": "Stain",   "desc": "Best_PSNR"},
    {"name": "U+307E_200025191_00017_2_X0981_Y1815_Stain.png",   "cat": "Stain",   "desc": "Best_PSNR_2"},
    {"name": "U+304F_200003803_00023_2_X1168_Y1411_Stain.png",   "cat": "Stain",   "desc": "Best_LPIPS"},
    {"name": "U+305F_200021071_00014_2_X0994_Y2191_Stain.png",   "cat": "Stain",   "desc": "Best_LPIPS_2"},
    {"name": "U+307E_200015779_00014_1_X0163_Y1422_Ghosting.png", "cat": "Ghosting", "desc": "Best_PSNR"},
    {"name": "U+3057_100241706_00030_2_X0966_Y2759_Ghosting.png", "cat": "Ghosting", "desc": "Best_PSNR_2"},
    {"name": "U+304F_200020019_00079_2_X1143_Y0554_Ghosting.png", "cat": "Ghosting", "desc": "Best_LPIPS"},
    {"name": "U+304A_200019865_00103_1_X1026_Y1791_Ghosting.png", "cat": "Ghosting", "desc": "Best_LPIPS_2"},
    {"name": "U+306E_200017458_00039_1_X0268_Y1205_Transparent_Stain.png", "cat": "Transparent", "desc": "Success_Case"},
    {"name": "U+306C_200025191_00060_2_X1116_Y1886_Transparent_Stain.png", "cat": "Transparent", "desc": "Success_Case_2"},
    {"name": "U+306C_200003076_00153_1_X0398_Y1285_Transparent_Stain.png", "cat": "Transparent", "desc": "LPIPS_Good"},
    {"name": "U+304B_200018243_00009_1_X0232_Y0825_Transparent_Stain.png", "cat": "Transparent", "desc": "LPIPS_Good_2"},
    {"name": "U+308F_200021712_00068_2_X1937_Y0821_Transparent_Stain.png", "cat": "Transparent", "desc": "Failure_Case_Discussion"},
]

def find_file(directory, filename, label):
    """
    GT系ファイルのマッチングを強化した関数
    """
    if not os.path.exists(directory):
        return None

    # 1. まずはそのままの名前で探す
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return path

    # 2. 座標(_X...) より前の部分を抽出 (例: U+3042_100249376_00034_2)
    stem = os.path.splitext(filename)[0]
    base_id = re.split(r'_X\d+_Y\d+', stem)[0]
    
    # 3. 拡張子のバリエーションを作成
    exts = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    # GT または Mask_GT の場合、サフィックスなしの名前で探索
    if label in ["GT", "Mask_GT"]:
        # (A) base_id そのままの名前で探す
        for ext in exts:
            p = os.path.join(directory, base_id + ext)
            if os.path.exists(p): return p
        
        # (B) ID部分のみで探す (Unicodeタグを除去したパターン)
        # U+3042_100249376_00034_2 -> 100249376_00034_2
        only_id = re.sub(r'^U\+[0-9A-F]{4,6}_', '', base_id)
        if only_id != base_id:
            for ext in exts:
                p = os.path.join(directory, only_id + ext)
                if os.path.exists(p): return p
        
        # (C) フォルダ内を走査して、ファイル名に base_id が含まれているか確認
        try:
            for f in os.listdir(directory):
                f_stem = os.path.splitext(f)[0]
                if f_stem == base_id or f_stem == only_id:
                    return os.path.join(directory, f)
        except OSError:
            pass

    # その他の画像 (Input/Restored等) についても、拡張子が違う可能性を考慮
    else:
        for ext in exts:
            p = os.path.join(directory, stem + ext)
            if os.path.exists(p): return p

    return None

def main():
    if not os.path.exists(DEST_ROOT):
        os.makedirs(DEST_ROOT)

    sources = {
        "Input":    SRC_INPUT,
        "GT":       SRC_GT,
        "Baseline": SRC_NOMASK,
        "Ours":     SRC_PRED,
        "Oracle":   SRC_GTMASK,
        "Mask_P":   SRC_MASK,
        "Mask_GT":  SRC_GTM_IMG
    }

    print(f"Starting collection into: {DEST_ROOT}")
    
    count = 0
    for target in TARGET_FILES:
        fname = target["name"]
        cat   = target["cat"]
        desc  = target["desc"]
        save_dir = os.path.join(DEST_ROOT, cat, desc)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Processing: {fname}")
        
        for label, src_dir in sources.items():
            found_path = find_file(src_dir, fname, label)
            if found_path:
                ext = os.path.splitext(found_path)[1]
                dest_path = os.path.join(save_dir, f"{label}{ext}")
                shutil.copy2(found_path, dest_path)
            else:
                print(f"  [WARN] {label} not found in {src_dir}")
        count += 1

    print(f"\n[DONE] Processed {count} images. Check '{DEST_ROOT}' folder.")

if __name__ == "__main__":
    main()