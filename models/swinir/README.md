# SwinIR-tiny for Kuzushiji Restoration

SwinIR-tinyモデルを使用した崩し字修復の実装です。

## モデルアーキテクチャ

- **embed_dim**: 60
- **depths**: [6, 6, 6, 6]
- **num_heads**: [6, 6, 6, 6]
- **window_size**: 8
- **パラメータ数**: 約8.8M

## セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt
```

## 推論

### マスクなし版

```bash
cd swinir
python infer_nomask.py \
    --checkpoint path/to/checkpoint.pth \
    --input-dir ../hiragana_fulldataset_5stain/lq/test \
    --output-dir ../outputs/swinir_nomask \
    --device cuda
```

### マスクあり版

```bash
cd swinir
python infer_withmask.py \
    --checkpoint path/to/checkpoint.pth \
    --input-dir ../hiragana_fulldataset_5stain/lq/test \
    --mask-dir ../hiragana_fulldataset_5stain/gt_mask/test \
    --output-dir ../outputs/swinir_withmask \
    --device cuda
```

## トレーニング

BasicSRフレームワークを使用してトレーニングを実行できます。

### マスクなし版

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
    basicsr/train.py -opt options/Kuzushiji/nomask.yml --launcher pytorch
```

### マスクあり版

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
    basicsr/train.py -opt options/Kuzushiji/withmask.yml --launcher pytorch
```

## ディレクトリ構造

```
swinir/
├── basicsr/
│   └── models/
│       └── archs/
│           └── SwinIR_arch.py    # SwinIRモデルの実装
├── options/
│   └── Kuzushiji/
│       ├── nomask.yml            # マスクなし学習設定
│       └── withmask.yml          # マスクあり学習設定
├── infer_nomask.py               # マスクなし推論スクリプト
├── infer_withmask.py             # マスクあり推論スクリプト
├── requirements.txt              # 依存関係
└── README.md                     # このファイル
```

## 比較実験

NAFNetやRestormerとの比較実験を行う際は、同じデータセットとメトリクスを使用してください。

```bash
# 推論実行後、メトリクスを計算
python ../scripts/calculate_5metrics.py \
    --gt-dir ../hiragana_fulldataset_5stain/gt/test \
    --restored-dir ../outputs/swinir_nomask \
    --output results.csv
```

## モデルの特徴

- **Swin Transformer**: Window-based self-attentionを使用
- **Residual Swin Transformer Blocks (RSTB)**: 残差学習による学習の安定化
- **軽量設計**: SwinIR-tinyは比較的軽量で効率的
- **画像修復タスク**: デノイジング、超解像、圧縮アーティファクト除去などに対応

## 参考文献

- SwinIR: Image Restoration Using Swin Transformer
- https://github.com/JingyunLiang/SwinIR
