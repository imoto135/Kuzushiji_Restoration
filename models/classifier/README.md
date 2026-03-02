# Kuzushiji Restoration Evaluation Tools

このディレクトリには、くずし字修復モデルの性能を評価するためのスクリプトが含まれています。
特に、修復された画像の文字認識精度（Accuracy）を測定するための分類器を作成・使用するツールです。

## セットアップ手順

### 環境構築 (推奨)
分類器の学習と評価のために、専用の `classifier_env` 環境を使用することを推奨します。

```bash
conda create -n classifier_env python=3.10 -y
conda activate classifier_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install tqdm requests numpy -y
```
※ `Intel MKL` 関連のエラーが出る場合は、`conda install -n classifier_env "intel-openmp<2025" mkl=2024.0.0 -y` 等でバージョンを調整してください。

### 1. データセットの準備

**推奨: ローカルデータセット (`data/hiragana_fulldataset_5stain/gt`) を使用する場合**
プロジェクト内にすでに配置されている `data/hiragana_fulldataset_5stain/gt` を使用します。
このデータセットは `train`, `val`, `test` ディレクトリを含み、ファイル名に正解ラベル（例: `U+3042_...`）が含まれている必要があります。

**(オプション) Kuzushiji-49 データセットを使用する場合**
自動ダウンロードスクリプト（`prepare_data.py`）を使用するか、手動で `.npz` ファイルを `data/kuzushiji/` に配置してください。

### 2. 分類器の学習

#### A. ローカルデータセットを使用する場合 (推奨)
`data/hiragana_fulldataset_5stain/gt` を使用して、プロジェクト固有の分類器を学習します。
Early Stopping（早期終了）や学習率スケジューリング、Weight Decayが導入されています。

学習は時間がかかるため、`screen` セッションでバックグラウンド実行することを推奨します。

```bash
# screen セッションの開始
screen -S classifier_train -d -m bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate classifier_env && python evaluation/train_classifier_local.py --epochs 50 --patience 5 --batch_size 32 > evaluation/training.log 2>&1"

# ログの確認
tail -f evaluation/training.log
```

手動で実行する場合:
```bash
conda activate classifier_env
python evaluation/train_classifier_local.py --epochs 50 --patience 5 --batch_size 32
```

手動で実行する場合:
```bash
conda activate classifier_env
python evaluation/train_classifier_local.py --epochs 50 --patience 5 --batch_size 32
```
- 学習が完了すると、`evaluation/best_classifier_local.pth` が保存されます。
- `--patience`: 検証ロスが改善しなくなってから何エポック待つか（デフォルト: 5）。

#### B. Kuzushiji-49 データセットを使用する場合
Kuzushiji-49 で ResNet18 分類モデルを学習させます。

```bash
python evaluation/train_classifier.py --epochs 10 --batch_size 64
```
- 学習が完了すると、`evaluation/best_classifier_k49.pth` が保存されます。

### 3. 修復画像の評価

#### A. ローカル学習モデルを使用する場合 (推奨)
修復された画像（または評価したい画像群）が保存されているディレクトリを指定して評価を実行します。
ファイル名に正解ラベル（例: `U+3042_...`）が含まれている場合、Accuracy（正解率）も計算されます。

```bash
# 例: data/test_images フォルダ内の画像を評価
conda activate classifier_env
python evaluation/evaluate_restoration_local.py --image_dir data/test_images --model_path evaluation/best_classifier_local.pth
```

#### B. Kuzushiji-49 学習モデルを使用する場合
`evaluate_restoration.py` を使用します。正解ラベルの取得ロジックはデータ構造に合わせて調整が必要な場合があります。

```bash
python evaluation/evaluate_restoration.py --image_dir data/test_images
```

## ファイル構成
- `train_classifier_local.py`: ローカルデータセット用学習スクリプト（Early Stopping対応）
- `evaluate_restoration_local.py`: ローカルモデル用評価スクリプト
- `train_classifier.py`: Kuzushiji-49用学習スクリプト
- `evaluate_restoration.py`: Kuzushiji-49用評価スクリプト
- `prepare_data.py`: Kuzushiji-49データセットダウンロードスクリプト
