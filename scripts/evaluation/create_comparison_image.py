import os
import glob
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# 設定パラメータ（環境に合わせて変更してください）
# ==============================================================================
# 8つのディレクトリパス
DIR_INPUT      = "data/hiragana_fulldataset_5stain/lq/test"
DIR_PRED_MASK  = "data/hiragana_fulldataset_5stain/pred_mask/test"
DIR_GT_MASK    = "data/hiragana_fulldataset_5stain/gt_mask/test"
DIR_GT_IMAGE   = "data/hiragana_fulldataset_5stain/gt/test"
DIR_RESTORMER  = "outputs/restormer/restormer_predmask_charbpercep"
DIR_MPRNET     = "outputs/mprnet/mprnet_predmask_charbpercep"
DIR_SWINIR     = "outputs/swinir/swinir_predmask_charbpercep"
DIR_NAFNET     = "outputs/nafnet/nafnet_predmask_charbpercep"

# 出力先ディレクトリ
OUTPUT_DIR = "outputs/comparison_predmask_charbpercep_4models"

# 入力ディレクトリ (DIR_INPUT) 内のすべての画像を対象として処理します。

# ==============================================================================
# 定数定義
# ==============================================================================
IMG_SIZE = 128
MARGIN_TOP = 24
PANEL_WIDTH = IMG_SIZE
PANEL_HEIGHT = IMG_SIZE + MARGIN_TOP

ROWS = 2
COLS = 4

FINAL_WIDTH = PANEL_WIDTH * COLS   # 128 * 4 = 512px
FINAL_HEIGHT = PANEL_HEIGHT * ROWS # 152 * 2 = 304px

# パネルのレイアウト構成
PANELS_CONFIG = [
    {"label": "(a) Input", "dir": DIR_INPUT},
    {"label": "(b) Predicted Mask", "dir": DIR_PRED_MASK},
    {"label": "(c) GT Mask", "dir": DIR_GT_MASK, "is_gt": True},
    {"label": "(d) GT Image", "dir": DIR_GT_IMAGE, "is_gt": True},
    {"label": "(e) Restormer", "dir": DIR_RESTORMER},
    {"label": "(f) MPRNet", "dir": DIR_MPRNET},
    {"label": "(g) SwinIR", "dir": DIR_SWINIR},
    {"label": "(h) NAFNet", "dir": DIR_NAFNET},
]

# 論文用のサンセリフ体フォントの読み込み
# Linux環境等で一般的なフォントをフォールバック付きで指定
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
except IOError:
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 13)
    except IOError:
        try:
            font = ImageFont.truetype("Arial.ttf", 13)
        except IOError:
            # 見つからない場合はPillowのデフォルトを使用
            font = ImageFont.load_default()

def get_core_name(filename):
    """
    ファイル名から劣化表現のサフィックスを除外して返す。
    例: U+3042_..._Transparent_Stain -> U+3042_...
    """
    suffixes = ['_Transparent_Stain', '_Missing', '_Stain', '_Ghosting', '_Scratch']
    for suffix in suffixes:
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
    return filename

def find_image_file(base_name, directory, is_gt=False):
    """
    指定ディレクトリからベース名で始まる画像ファイルを取得する。
    サフィックスや拡張子の違い（.png, .jpgなど）を無視して検索。
    """
    search_name = get_core_name(base_name) if is_gt else base_name
    pattern = os.path.join(directory, f"{search_name}*.*")
    files = glob.glob(pattern)
    valid_exts = {'.jpg', '.jpeg', '.png'}
    
    for f in files:
        if os.path.splitext(f)[1].lower() in valid_exts:
            return f
    return None

def process_single_image(base_name):
    """
    1つのベースファイル名に対する結合画像を生成する
    """
    # 背景白のキャンバスを作成 (512x304)
    final_img = Image.new('RGB', (FINAL_WIDTH, FINAL_HEIGHT), 'white')
    draw = ImageDraw.Draw(final_img)
    
    for idx, config in enumerate(PANELS_CONFIG):
        row = idx // COLS
        col = idx % COLS
        
        # 1. 画像の検索
        is_gt = config.get("is_gt", False)
        img_path = find_image_file(base_name, config["dir"], is_gt=is_gt)
        if img_path is None:
            raise FileNotFoundError(f"Missing image for '{base_name}' in directory: {config['dir']}")
        
        # 2. 画像の読み込みとサイズ補正
        try:
            img = Image.open(img_path).convert('RGB')
            # 仕様通り 128x128 にリサイズ (念のため)
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        except Exception as e:
            raise RuntimeError(f"Failed to load/process image '{img_path}': {e}")
        
        # 3. 配置座標の計算
        x_offset = col * PANEL_WIDTH
        y_offset = row * PANEL_HEIGHT
        
        # 4. ラベル描画（中央揃え）
        label = config["label"]
        
        # Pillowのバージョンによるテキストサイズ取得の違いを吸収
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = draw.textsize(label, font=font)
        
        text_x = x_offset + (PANEL_WIDTH - text_w) / 2
        text_y = y_offset + (MARGIN_TOP - text_h) / 2 - 2  # 少し上に調整
        
        draw.text((text_x, text_y), label, fill="black", font=font)
        
        # 5. 画像の貼り付け
        final_img.paste(img, (x_offset, y_offset + MARGIN_TOP))
        
    return final_img

def get_all_base_filenames(directory):
    """
    指定ディレクトリ内の全画像ファイルの拡張子を除いたベースファイル名を取得する。
    """
    valid_exts = {'.jpg', '.jpeg', '.png'}
    base_names = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_exts:
                base_names.append(name)
    return sorted(list(set(base_names)))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ready: {OUTPUT_DIR}")
    
    base_filenames = get_all_base_filenames(DIR_INPUT)
    if not base_filenames:
        print(f"No images found in directory: {DIR_INPUT}")
        return

    print(f"Found {len(base_filenames)} images. Start processing...")
    print("-" * 50)
    
    success_count = 0
    
    for base_name in base_filenames:
        print(f"Processing: {base_name} ...", end=" ")
        try:
            combined_img = process_single_image(base_name)
            out_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
            combined_img.save(out_path)
            print("Done.")
            success_count += 1
        except Exception as e:
            print(f"\n  -> SKIPPED. Error: {e}")
            
    print("-" * 50)
    print(f"Finished. Successfully processed {success_count} / {len(base_filenames)} images.")

if __name__ == "__main__":
    main()
