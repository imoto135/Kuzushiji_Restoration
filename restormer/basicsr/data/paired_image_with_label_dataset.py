import os
import pandas as pd
from basicsr.data.paired_image_dataset import Dataset_PairedImage
from basicsr.utils import get_root_logger

class PairedImageWithLabelDataset(Dataset_PairedImage):
    """
    Paired Image Datasetを継承し、アノテーションファイルから文字ラベルを追加するクラス。
    """
    def __init__(self, opt):
        # まず、親クラス(Dataset_PairedImage)の初期化処理を呼び出す
        # これにより、画像のパスなどが準備される
        super(PairedImageWithLabelDataset, self).__init__(opt)
        logger = get_root_logger()
        
        # --- ここからが新しい処理 ---
        # 1. .ymlファイルからclass_map.csvへのパスを取得
        class_map_path = self.opt.get('class_map_path')
        if not class_map_path:
            raise ValueError("設定ファイル(.yml)に class_map_path が指定されていません。")

        try:
            # 2. Pandasを使い、公式の「文字とIDの対応表」を読み込む
            class_map_df = pd.read_csv(class_map_path)
            # 3. 高速に検索できるよう、文字をキー、IDを値とする辞書を作成
            self.char_to_id = pd.Series(class_map_df.class_id.values, index=class_map_df.char_unicode).to_dict()
            logger.info(f"公式のクラスマッピング '{class_map_path}' をロードしました。")
            logger.info(f'クラス総数: {len(self.char_to_id)}')
        except FileNotFoundError:
            raise FileNotFoundError(f"クラスマッピングファイルが見つかりません: {class_map_path}")
        # --- ここまで ---

    def __getitem__(self, index):
        # 1. 親クラスの機能を使って、画像ペア(lq, gt)とパスを取得する
        data = super(PairedImageWithLabelDataset, self).__getitem__(index)
        
        # 2. パスからファイル名を取得し、文字ラベル部分を抽出
        filename = os.path.basename(data['lq_path'])
        char_label = filename.split('_')[0]
        
        # 3. 読み込んでおいた対応辞書を使って、文字ラベルをクラスID（数値）に変換
        class_id = self.char_to_id.get(char_label, -1) # マップにない未知の文字は-1とする
        
        # 4. もし未知の文字があれば警告を出す
        if class_id == -1:
            logger = get_root_logger()
            logger.warning(f"'{char_label}' がクラスマップにありません。ファイル: {filename}")
        
        # 5. 取得したラベルを、返却するデータ辞書に追加する
        data['label'] = class_id
        
        return data

