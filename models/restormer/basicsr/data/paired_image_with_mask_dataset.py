from torch.utils import data as data
import torch
import os
import random
import numpy as np

from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.data.transforms import augment
from basicsr.utils.dist_util import get_dist_info

class PairedImageWithMaskDataset(data.Dataset):
    """
    LQ (損傷画像), GT (正解画像), Mask (予測マスク) の3つのフォルダを扱い、
    事前生成されたファイルリストを読み込むカスタムデータセット。
    """
    def __init__(self, opt):
        super(PairedImageWithMaskDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.mask_folder = opt['dataroot_mask']
        
        # --- ★★★ 変更点: ファイルリストからパスを生成 ★★★ ---
        # 1. .ymlファイルからファイルリストのパスを取得
        filelist_path = opt['filelist_path']
        
        # 2. ファイルリストを読み込み、ファイル名のリストを作成
        with open(filelist_path, 'r') as f:
            self.filenames = [line.strip() for line in f]

        # 3. 拡張子を取得（最初のファイルを代表として全ファイルで同じと仮定）
        if self.filenames:
            sample_filename_with_ext = [f for f in os.listdir(self.lq_folder) if f.startswith(self.filenames[0])][0]
            self.ext = os.path.splitext(sample_filename_with_ext)[1]
        else:
            self.ext = '.png' # デフォルト

        # 4. ファイル名のリストを元に、完全なパスのリストを生成
        self.paths = []
        for filename in self.filenames:
            self.paths.append({
                'lq_path': os.path.join(self.lq_folder, f"{filename}{self.ext}"),
                'gt_path': os.path.join(self.gt_folder, f"{filename}{self.ext}"),
                'mask_path': os.path.join(self.mask_folder, f"{filename}{self.ext}")
            })
        # --- ここまで ---
        
        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt.get('geometric_augs', False)
            self.use_flip = self.opt.get('use_flip', False)
            self.use_rot = self.opt.get('use_rot', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 1)
        
        # gt (正解画像) の読み込み
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # lq (損傷画像) の読み込み
        # LQには劣化タイプのサフィックスがある（例：_Stain, _Ghosting, etc.）
        lq_path = self.paths[index]['lq_path']
        if not os.path.exists(lq_path):
            # ワイルドカードでマッチするファイルを探す
            base_no_ext = os.path.splitext(lq_path)[0]
            lq_dir = os.path.dirname(lq_path)
            lq_basename = os.path.basename(base_no_ext)
            import glob
            candidates = glob.glob(f"{lq_dir}/{lq_basename}_*.jpg")
            if candidates:
                lq_path = candidates[0]
            else:
                raise FileNotFoundError(f"LQ file not found: {lq_path}")
        
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # マスクを読み込む
        # Maskにも劣化タイプのサフィックスがある
        mask_path = self.paths[index]['mask_path']
        if not os.path.exists(mask_path):
            # ワイルドカードでマッチするファイルを探す
            base_no_ext = os.path.splitext(mask_path)[0]
            mask_dir = os.path.dirname(mask_path)
            mask_basename = os.path.basename(base_no_ext)
            import glob
            candidates = glob.glob(f"{mask_dir}/{mask_basename}_*.jpg")
            if candidates:
                mask_path = candidates[0]
            else:
                # Try several likely mask filename variants and extensions.
                # Some masks in this dataset have a "_prediction" suffix (e.g. "<name>_prediction.jpg").
                tried = []
                found = False

                # candidate suffixes to try (empty + common suffixes)
                suffixes = ['', '_prediction', '_mask']
                # candidate extensions to try (order matters: prefer detected ext, then common ones)
                detected_ext = os.path.splitext(mask_path)[1]
                exts = [detected_ext] if detected_ext else []
                for e in ('.jpg', '.png', '.jpeg'):
                    if e not in exts:
                        exts.append(e)

                for suf in suffixes:
                    for ext in exts:
                        candidate = base_no_ext + suf + ext
                        # skip exact duplicate of the primary mask_path when already tried
                        if candidate == mask_path:
                            try:
                                mask_bytes = self.file_client.get(candidate, 'mask')
                                found = True
                                mask_path = candidate
                                break
                            except FileNotFoundError:
                                tried.append(candidate)
                                continue

                        try:
                            mask_bytes = self.file_client.get(candidate, 'mask')
                            mask_path = candidate
                            found = True
                            break
                        except FileNotFoundError:
                            tried.append(candidate)
                    if found:
                        break

                if not found:
                    # Re-raise with helpful message
                    raise FileNotFoundError(f"Mask not found for {mask_path}. Tried: {', '.join(tried)}")
        
        img_bytes = self.file_client.get(mask_path, 'mask')
        img_mask = imfrombytes(img_bytes, flag='grayscale', float32=True)
        img_mask = img_mask[..., None]

        # 学習時のデータ拡張
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq, img_mask = self.custom_paired_random_crop([img_gt, img_lq, img_mask], gt_size, scale, gt_path)
            if self.geometric_augs:
                 [img_gt, img_lq, img_mask] = augment([img_gt, img_lq, img_mask], self.use_flip, self.use_rot)

        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)
        img_lq = torch.cat([img_lq, img_mask], dim=0)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

    def custom_paired_random_crop(self, imgs, gt_size, scale, gt_path):
        """3枚の画像を同時にランダムクロップするカスタム関数"""
        h_gt, w_gt, _ = imgs[0].shape
        
        if h_gt < gt_size or w_gt < gt_size:
            pad_h = max(0, gt_size - h_gt)
            pad_w = max(0, gt_size - w_gt)
            imgs[0] = np.pad(imgs[0], ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')
            imgs[1] = np.pad(imgs[1], ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')
            imgs[2] = np.pad(imgs[2], ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')
            h_gt, w_gt, _ = imgs[0].shape

        top = random.randint(0, h_gt - gt_size)
        left = random.randint(0, w_gt - gt_size)
        
        img_gt_cropped = imgs[0][top : top + gt_size, left : left + gt_size, :]
        
        top_lq, left_lq = top // scale, left // scale
        h_lq_crop, w_lq_crop = gt_size // scale, gt_size // scale
        img_lq_cropped = imgs[1][top_lq : top_lq + h_lq_crop, left_lq : left_lq + w_lq_crop, :]
        img_mask_cropped = imgs[2][top_lq : top_lq + h_lq_crop, left_lq : left_lq + w_lq_crop, :]

        return img_gt_cropped, img_lq_cropped, img_mask_cropped