import os
import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import argparse
import logging
from pathlib import Path

# Restormerモデルの定義（ログから抽出した設定）
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, data_format='channels_first')
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, data_format='channels_first')
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, 
                 inp_channels=4,
                 out_channels=3, 
                 dim=48,
                 num_blocks=[4, 6, 6, 8], 
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img[:, :3, :, :]

        return out_dec_level1


def load_model(weights_path, device):
    """モデルをロード"""
    model = Restormer(
        inp_channels=4,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
    
    model = model.to(device)
    
    # 重みをロード
    checkpoint = torch.load(weights_path, map_location=device)
    
    # state_dictを取得（キーの確認）
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 'params_ema'も試す
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def process_image(model, lq_path, mask_path, device):
    """画像を処理"""
    # 劣化画像を読み込み
    lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
    lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
    
    # マスク画像を読み込み
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    
    # 正規化
    lq = lq.astype(np.float32) / 255.0
    
    # HWC to CHW
    lq = np.transpose(lq, (2, 0, 1))
    mask = np.expand_dims(mask, axis=0)
    
    # マスクとLQ画像を結合
    input_tensor = np.concatenate([lq, mask], axis=0)
    
    # numpy to tensor
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)
    
    # 推論
    with torch.no_grad():
        output = model(input_tensor)
    
    # tensor to numpy
    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Restore images using Restormer')
    parser.add_argument('--lq_dir', type=str, default='data/hiragana_fulldataset_5stain/lq/test',
                        help='劣化画像のディレクトリ')
    parser.add_argument('--mask_dir', type=str, default='data/hiragana_fulldataset_5stain/gt_mask/test',
                        help='予測マスクのディレクトリ')
    parser.add_argument('--output_dir', type=str, default='outputs/restormer_gtmask_charbpercep',
                        help='出力ディレクトリ')
    parser.add_argument('--weights', type=str, 
                        default='models/restormer/experiments/Restormer_GTMask_CharbPercep/models/net_g_200000.pth',
                        help='モデルの重みファイルパス')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス')
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 画像リストを取得
    lq_images = sorted([f for f in os.listdir(args.lq_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    logging.info(f"Found {len(lq_images)} images to process")
    
    # モデルをロード
    logging.info(f"Loading model from {args.weights}")
    model = load_model(args.weights, device)
    
    # 各画像を処理
    for img_name in tqdm(lq_images):
        lq_path = os.path.join(args.lq_dir, img_name)
        # LQ画像名から損傷種別サフィックスを除いてgt_maskファイル名を導出
        # 例: U+3042_xxx_Stain.jpg -> U+3042_xxx.jpg
        stem = Path(img_name).stem
        suffix = Path(img_name).suffix
        damage_types = ['_Transparent_Stain', '_Missing', '_Stain', '_Scratch', '_Ghosting']
        mask_stem = stem
        for dt in damage_types:
            if mask_stem.endswith(dt):
                mask_stem = mask_stem[:-len(dt)]
                break
        mask_filename = mask_stem + suffix
        mask_path = os.path.join(args.mask_dir, mask_filename)
        # .jpg で見つからなければ .png も試す
        if not os.path.exists(mask_path):
            mask_path = os.path.join(args.mask_dir, mask_stem + '.png')

        output_path = os.path.join(args.output_dir, img_name)

        if not os.path.exists(mask_path):
            logging.warning(f"Mask not found for {img_name}, skipping")
            continue
        
        # 画像を処理
        restored = process_image(model, lq_path, mask_path, device)
        
        # 保存
        cv2.imwrite(output_path, restored)
    
    logging.info(f"Processing complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()