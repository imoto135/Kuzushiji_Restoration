import torch
import torch.nn as nn
from basicsr.models.archs.restormer_arch import Restormer

class RestormerWithClassifier(nn.Module):
    def __init__(self, num_classes, **restormer_args):
        super(RestormerWithClassifier, self).__init__()
        
        # 1. 内部にRestormer（修復エンジン）を持つ
        self.restormer = Restormer(**restormer_args)
        
        # 2. 分類ヘッドを定義
        #    入力チャンネル数は、restormerから渡される特徴マップの次元数に合わせる
        feature_dim = restormer_args.get('dim', 48) * 2
        
        self.classifier_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x):
        # 1. Restormerを実行し、修復画像と特徴マップを受け取る
        restored_image, features = self.restormer(x)
        
        # 2. 特徴マップを分類ヘッドに通し、分類結果を得る
        class_logits = self.classifier_head(features)
        
        # 3. 2つの結果を返す
        return restored_image, class_logits