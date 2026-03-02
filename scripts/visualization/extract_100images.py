import os
import shutil

# ==========================================
# 1. パスの設定（ご自身の環境に合わせて変更してください）
# ==========================================
# コピー元のディレクトリ
SOURCE_DIRS = {
    "comparison": "/home/imoto/Kuzushiji_Restoration/outputs/comparison_predmask_charbpercep_4models"
}

# 抽出先のベースディレクトリ
OUTPUT_BASE_DIR = "./selected_100_images"

# ==========================================
# 2. 特筆すべき100枚のファイルリスト
# ==========================================
TARGET_FILES = [
    'U+309D_100241706_00011_2_X1208_Y1258_Scratch.jpg', 'U+307F_100241706_00018_2_X1394_Y1144_Scratch.jpg', 
    'U+308A_200021712_00078_2_X0789_Y3051_Missing.jpg', 'U+3057_200021712_00031_1_X1353_Y3383_Ghosting.jpg', 
    'U+306F_200003803_00018_2_X1637_Y2724_Stain.jpg', 'U+3082_hnsd012_009_X0831_Y0900_Scratch.jpg', 
    'U+306E_200015779_00155_2_X0745_Y1339_Scratch.jpg', 'U+3051_200019865_00038_2_X1117_Y2514_Missing.jpg', 
    'U+306F_100241706_00037_2_X1859_Y2270_Stain.jpg', 'U+304B_200021712_00053_1_X0098_Y1918_Stain.jpg', 
    'U+304F_200019865_00095_2_X0060_Y0834_Missing.jpg', 'U+305F_200014685_00020_2_X0305_Y1618_Stain.jpg', 
    'U+308C_200021763_00033_2_X0487_Y2974_Ghosting.jpg', 'U+306E_umgy002_006_X1484_Y0621_Missing.jpg', 
    'U+306A_100241706_00005_1_X1072_Y3050_Ghosting.jpg', 'U+3072_200021763_00023_1_X2087_Y1093_Ghosting.jpg', 
    'U+3084_200019865_00059_1_X0965_Y1315_Transparent_Stain.jpg', 'U+3059_200019865_00118_1_X1874_Y1206_Transparent_Stain.jpg', 
    'U+3057_200022050_00001_X1329_Y0440_Transparent_Stain.jpg', 'U+3067_200019865_00124_2_X0160_Y1681_Transparent_Stain.jpg', 
    'U+309D_100249416_00003_1_X1766_Y0632_Scratch.jpg', 'U+3092_hnsd011_025_X1609_Y1841_Scratch.jpg', 
    'U+305F_hnsd012_019_X1390_Y0311_Scratch.jpg', 'U+3082_hnsd008_018_X0150_Y1411_Scratch.jpg', 
    'U+3046_200015779_00068_2_X1916_Y2116_Stain.jpg', 'U+3046_200015779_00113_1_X1474_Y1375_Missing.jpg', 
    'U+3078_200020019_00004_1_X1509_Y2864_Ghosting.jpg', 'U+3057_umgy006_015_X0947_Y1929_Stain.jpg', 
    'U+306E_200019865_00034_2_X0285_Y2343_Ghosting.jpg', 'U+3066_200019865_00061_1_X0418_Y1087_Ghosting.jpg', 
    'U+3060_200019865_00123_2_X0147_Y2046_Missing.jpg', 'U+306A_200019865_00124_2_X1316_Y1323_Ghosting.jpg', 
    'U+306F_200020019_00043_1_X0659_Y2713_Missing.jpg', 'U+308B_200003076_00030_2_X0820_Y2266_Stain.jpg', 
    'U+3089_200019865_00104_2_X0997_Y2012_Stain.jpg', 'U+308B_200021712_00056_2_X1339_Y1379_Missing.jpg', 
    'U+3092_200019865_00068_1_X1115_Y2911_Transparent_Stain.jpg', 'U+306E_200021763_00013_1_X1529_Y1853_Transparent_Stain.jpg', 
    'U+3057_200003803_00004_2_X1900_Y1424_Transparent_Stain.jpg', 'U+304F_200019865_00060_2_X1145_Y2832_Transparent_Stain.jpg', 
    'U+304D_200021071_00037_1_X1317_Y0742_Stain.jpg', 'U+306A_umgy001_007_X0188_Y0782_Scratch.jpg', 
    'U+306B_200021644_00021_1_X1320_Y2747_Missing.jpg', 'U+3044_200021071_00031_2_X1623_Y1735_Stain.jpg', 
    'U+3078_100249376_00047_2_X0207_Y1538_Missing.jpg', 'U+3066_200020019_00068_2_X1832_Y0712_Missing.jpg', 
    'U+308B_200021071_00007_2_X0827_Y0882_Missing.jpg', 'U+304A_200003803_00024_1_X1537_Y1379_Stain.jpg', 
    'U+3057_200003076_00036_2_X1305_Y1240_Stain.jpg', 'U+307E_umgy010_027_X0187_Y1526_Scratch.jpg', 
    'U+308B_100241706_00029_1_X1318_Y1442_Scratch.jpg', 'U+3046_200022050_00001_X1653_Y0361_Scratch.jpg', 
    'U+305F_200020019_00075_1_X0658_Y0992_Transparent_Stain.jpg', 'U+3057_200019865_00023_2_X0709_Y1348_Transparent_Stain.jpg', 
    'U+3057_200021644_00026_1_X0161_Y0949_Transparent_Stain.jpg', 'U+3053_100241706_00027_2_X1411_Y1704_Transparent_Stain.jpg', 
    'U+3053_200022050_00014_1_X2094_Y0977_Ghosting.jpg', 'U+306E_200020019_00049_2_X1631_Y0373_Ghosting.jpg', 
    'U+3084_200020019_00016_2_X1288_Y1825_Ghosting.jpg', 'U+3092_200020019_00034_1_X0684_Y2098_Ghosting.jpg', 
    'U+3057_200021853_00022_2_X1142_Y0493_Stain.jpg', 'U+3057_200021853_00023_1_X0431_Y2096_Stain.jpg', 
    'U+3046_100249416_00011_1_X0575_Y0741_Transparent_Stain.jpg', 'U+3057_200019865_00049_1_X0611_Y2423_Missing.jpg', 
    'U+3057_200015779_00119_2_X1433_Y1980_Stain.jpg', 'U+3057_200021853_00011_2_X0580_Y1940_Missing.jpg', 
    'U+3057_200021925_00016_2_X2419_Y0669_Ghosting.jpg', 'U+3057_200021853_00019_1_X0212_Y1707_Stain.jpg', 
    'U+3057_100249416_00022_1_X0928_Y1532_Ghosting.jpg', 'U+3057_100249376_00043_1_X1072_Y2243_Ghosting.jpg', 
    'U+3057_200021853_00014_2_X0665_Y2237_Ghosting.jpg', 'U+3057_100249371_00019_2_X0220_Y2188_Transparent_Stain.jpg', 
    'U+3046_200018243_00019_1_X0804_Y1166_Missing.jpg', 'U+3057_200025191_00011_1_X0600_Y1346_Transparent_Stain.jpg', 
    'U+3057_200021853_00019_2_X1054_Y1005_Missing.jpg', 'U+306B_200021644_00012_2_X1021_Y1524_Transparent_Stain.jpg', 
    'U+3057_200021853_00033_1_X0658_Y0764_Scratch.jpg', 'U+3057_100249416_00029_2_X0731_Y1074_Scratch.jpg', 
    'U+3057_200021644_00031_2_X1351_Y1604_Scratch.jpg', 'U+3046_200021853_00006_2_X0427_Y1873_Scratch.jpg', 
    'U+309D_200021712_00039_2_X1752_Y1099_Stain.jpg', 'U+306F_200003803_00028_2_X0322_Y1439_Ghosting.jpg', 
    'U+3055_200010454_00027_2_X0202_Y0707_Missing.jpg', 'U+304D_umgy002_008_X0330_Y0570_Missing.jpg', 
    'U+3075_200019865_00023_2_X1283_Y1057_Ghosting.jpg', 'U+308A_200021660_00009_1_X0215_Y2788_Missing.jpg', 
    'U+304A_200003076_00023_2_X0241_Y2087_Transparent_Stain.jpg', 'U+3066_200003076_00120_2_X1525_Y1786_Ghosting.jpg', 
    'U+3066_200014740_00051_2_X1207_Y0303_Scratch.jpg', 'U+308C_200003076_00154_1_X0586_Y2199_Transparent_Stain.jpg', 
    'U+3069_200003076_00068_2_X1835_Y0590_Ghosting.jpg', 'U+3088_200019865_00058_1_X1103_Y2362_Stain.jpg', 
    'U+309D_200021086_00020_1_X1027_Y2270_Transparent_Stain.jpg', 'U+306B_200025191_00068_1_X0130_Y2194_Missing.jpg', 
    'U+3092_200021086_00010_1_X1266_Y0500_Stain.jpg', 'U+3088_200003803_00021_2_X0538_Y2283_Transparent_Stain.jpg', 
    'U+309D_200019865_00117_2_X1485_Y1349_Scratch.jpg', 'U+3084_200021063_00006_1_X1566_Y1119_Scratch.jpg', 
    'U+3060_hnsd012_019_X0813_Y1536_Stain.jpg', 'U+3093_200021086_00017_2_X1018_Y1666_Scratch.jpg'
]

# ==========================================
# 3. コピー処理の実行
# ==========================================
def extract_images():
    count_success = 0
    count_missing = 0
    
    # 抽出先ディレクトリをソースごとに作成（例： selected_100_images/nafnet/ ）
    for key in SOURCE_DIRS.keys():
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, key), exist_ok=True)

    for base_name in TARGET_FILES:
        # 拡張子を除いたファイル名を取得（念のため）
        name_no_ext = os.path.splitext(base_name)[0]
        
        for key, src_dir in SOURCE_DIRS.items():
            # 考えうる拡張子やサフィックスのパターン
            possible_names = [
                name_no_ext + "_comparison.png",
                name_no_ext + "_comparison.jpg",
                name_no_ext + ".jpg",
                name_no_ext + ".png",
                name_no_ext + "_restored.jpg",
                name_no_ext + "_restored.png"
            ]
            
            found = False
            for p_name in possible_names:
                src_path = os.path.join(src_dir, p_name)
                if os.path.exists(src_path):
                    # コピー先のパス（拡張子は元のまま）
                    dst_path = os.path.join(OUTPUT_BASE_DIR, key, p_name)
                    shutil.copy2(src_path, dst_path)
                    found = True
                    break
            
            if not found:
                print(f"Warning: {name_no_ext} not found in {key} directory.")
                count_missing += 1
        
        count_success += 1

    print("\n--- 完了 ---")
    print(f"処理したファイルセット: {count_success} 件")
    if count_missing > 0:
        print(f"見つからなかった画像: {count_missing} 件（Warningを確認してください）")
    print(f"保存先: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    extract_images()