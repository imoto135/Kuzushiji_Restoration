#!/bin/bash
# このスクリプトは、3つの異なるセグメンテーションモデルの学習を連続して実行します。

echo "===== 実験1/3 を開始します: U-Net ====="
python train_segmentation_models.py --model unet
echo "===== U-Net の学習が完了しました ====="


echo "\n===== 実験2/3 を開始します: Unet++ ====="
python train_segmentation_models.py --model unet++
echo "===== Unet++ の学習が完了しました ====="


echo "\n===== 実験3/3 を開始します: Attention U-Net ====="
python train_segmentation_models.py --model attention_unet
echo "===== Attention U-Net の学習が完了しました ====="


echo "\nすべての実験が完了しました。"