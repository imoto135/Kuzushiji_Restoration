#!/bin/bash
# Kuzushiji_Restoration ディレクトリ構造整理スクリプト
# 実行前に必ずバックアップを取ること！

set -e  # エラーで停止

echo "=== Kuzushiji_Restoration ディレクトリ構造整理 ==="
echo "⚠️  このスクリプトは実際のファイル移動を行います"
echo "実行前に Ctrl+C で中断し、バックアップを確認してください"
echo ""
read -p "続行しますか？ (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "キャンセルしました"
    exit 0
fi

BASE_DIR="/home/imoto/Kuzushiji_Restoration"
cd "$BASE_DIR"

echo ""
echo "▶ Step 1: 新しいディレクトリを作成"
mkdir -p docs
mkdir -p environments
mkdir -p models
mkdir -p configs/{nafnet,swinir,restormer,mprnet}
mkdir -p scripts/{data_preprocessing,evaluation,visualization}
mkdir -p data
mkdir -p archive
mkdir -p outputs/sample_results

echo ""
echo "▶ Step 2: 環境設定ファイルを移動"
mv environment.yml env_nafnet2.yml env.yml environments/ 2>/dev/null || echo "環境ファイルは既に移動済み"

echo ""
echo "▶ Step 3: モデルディレクトリを整理"
if [ -d "nafnet" ]; then
    mv nafnet models/
fi
if [ -d "swinir" ]; then
    mv swinir models/
fi
if [ -d "restormer" ]; then
    mv restormer models/
fi

echo ""
echo "▶ Step 4: スクリプトを整理"
if [ -f "scripts/add_stain_5types.py" ]; then
    mv scripts/add_stain_5types.py scripts/data_preprocessing/
fi

if [ -f "scripts/calculate_5metrics.py" ]; then
    mv scripts/calculate_5metrics.py scripts/evaluation/
fi
if [ -f "scripts/evaluate_masks.py" ]; then
    mv scripts/evaluate_masks.py scripts/evaluation/
fi
if [ -f "scripts/compare_restormer_nafnet.py" ]; then
    mv scripts/compare_restormer_nafnet.py scripts/evaluation/
fi
if [ -f "scripts/analyse.py" ]; then
    mv scripts/analyse.py scripts/evaluation/
fi

if [ -f "scripts/collect_images_nafnet.py" ]; then
    mv scripts/collect_images_nafnet.py scripts/visualization/
fi
if [ -f "scripts/collect_images.py" ]; then
    mv scripts/collect_images.py scripts/visualization/
fi
if [ -f "scripts/concat_images.py" ]; then
    mv scripts/concat_images.py scripts/visualization/
fi
if [ -f "scripts/concatenate_comparison.py" ]; then
    mv scripts/concatenate_comparison.py scripts/visualization/
fi
if [ -f "scripts/create_zoom.py" ]; then
    mv scripts/create_zoom.py scripts/visualization/
fi

# その他のスクリプトはscripts/直下に残す
echo "その他のスクリプトは scripts/ 直下に保持"

echo ""
echo "▶ Step 5: データセットを data/ に移動"
if [ -d "hiragana_dataset" ]; then
    mv hiragana_dataset data/
fi
if [ -d "hiragana_fulldataset_5stain" ]; then
    mv hiragana_fulldataset_5stain data/
fi

echo ""
echo "▶ Step 6: 古いファイルを archive/ に移動"
mv nohup*.out archive/ 2>/dev/null || echo "nohupファイルなし"
mv test_conda.out archive/ 2>/dev/null || echo "test_conda.outなし"
mv Miniconda3-latest-Linux-x86_64.sh archive/ 2>/dev/null || echo "Minicondaインストーラーなし"
mv hiragana_fulldataset_5stains.zip archive/ 2>/dev/null || echo "zipファイルなし"

if [ -d "co" ]; then
    mv co archive/
fi
if [ -d "sotsuron" ]; then
    mv sotsuron archive/
fi
if [ -d "stage1_segmentaion" ]; then
    mv stage1_segmentaion archive/
fi
if [ -d "stage2_restoration" ]; then
    mv stage2_restoration archive/
fi

echo ""
echo "▶ Step 7: 設定ファイルを configs/ にコピー (元のファイルは保持)"
if [ -d "models/nafnet/options" ]; then
    cp -r models/nafnet/options/* configs/nafnet/
fi
if [ -d "models/swinir/options" ]; then
    cp -r models/swinir/options/* configs/swinir/
fi
if [ -d "models/restormer/configs" ]; then
    cp -r models/restormer/configs/* configs/restormer/
fi

echo ""
echo "▶ Step 8: wandb と tb_logger を archive/ に移動"
if [ -d "wandb" ]; then
    mv wandb archive/
fi
if [ -d "tb_logger" ]; then
    mv tb_logger archive/
fi

echo ""
echo "▶ Step 9: analyze_metrics_custom.py を scripts/evaluation/ に移動"
if [ -f "analyze_metrics_custom.py" ]; then
    mv analyze_metrics_custom.py scripts/evaluation/
fi

echo ""
echo "✅ ディレクトリ構造の整理が完了しました！"
echo ""
echo "次のステップ:"
echo "1. 整理されたディレクトリを確認: ls -la"
echo "2. .gitignore を更新: 提案されたファイルを使用"
echo "3. README.md を作成"
echo "4. Git にコミット: git add . && git commit -m 'Reorganize directory structure'"
