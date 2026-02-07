# Kuzushiji Restoration: Stage 1 - Damage Segmentation

本リポジトリは、学士論文「UNet++とRestormerを用いた損傷くずし字画像の二段階修復手法の研究」における**第1段階（損傷領域の特定）**の公式実装です。

深層学習を用いてくずし字画像の損傷箇所（汚れ、欠損、虫食い等）を高精度に検出し、第2段階の修復モデル（Restormer）へ渡すための**ソフトマスク（確率マップ）**を生成します。

##  プロジェクト概要
古文書のデジタル化において、経年劣化による損傷はOCR精度や可読性を著しく低下させます。本研究では、修復プロセスを以下の2段階に分離することで、構造維持とノイズ除去の両立を実現しました。

1.  **Stage 1 (本リポジトリ):** `UNet++` を用いて損傷箇所をピクセルレベルで特定。
2.  **Stage 2:** 特定された損傷マスクをガイドとして、画像修復モデルでテクスチャを復元。

### 主な特徴
* **Model Architecture:** `UNet++` (Backbone: `SE-ResNeXt-50`) を採用し、複雑な損傷形状を捕捉。
* **Robustness:** 学習時に `Input Dropout` (CoarseDropout) を導入することで、未知の損傷パターンに対する頑健性を向上。
* **Output:** 2値マスクではなく、不確実性を考慮したソフトマスク（確率値）を出力し、次段の修復精度を最大化。

##  環境構築

Python 3.8+ 環境での動作を推奨します。

```bash
# リポジトリのクローン
git clone <repository-url>
cd <repository-folder>

# 依存ライブラリのインストール
pip install -r requirements.txt