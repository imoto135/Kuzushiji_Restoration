"""
MPRNet: Multi-Stage Progressive Image Restoration
Paper: https://arxiv.org/abs/2102.02808
Implementation for image restoration with mask input support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias))
        modules_body.append(act)
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, 3, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv3 = nn.Conv2d(3, n_feat, kernel_size, padding=kernel_size//2, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


##########################################################################
## U-Net Encoder
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


##########################################################################
## U-Net Decoder
class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
## Original Resolution Saliency Module (ORSM)
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats)
        )
        self.up_dec2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats)
        )

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
## Down/Up Sampling
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## MPRNet
class MPRNet(nn.Module):
    """
    MPRNet: Multi-Stage Progressive Image Restoration
    
    Args:
        in_c (int): Number of input channels. Default: 3
        out_c (int): Number of output channels. Default: 3
        n_feat (int): Number of features. Default: 80
        scale_unetfeats (int): Scale factor for U-Net features. Default: 48
        scale_orsnetfeats (int): Scale factor for ORSNet features. Default: 32
        num_cab (int): Number of CAB blocks. Default: 8
        kernel_size (int): Kernel size. Default: 3
        reduction (int): Reduction factor for channel attention. Default: 4
        bias (bool): Whether to use bias. Default: False
    """
    def __init__(self,
                 in_c=3,
                 out_c=3,
                 n_feat=80,
                 scale_unetfeats=48,
                 scale_orsnetfeats=32,
                 num_cab=8,
                 kernel_size=3,
                 reduction=4,
                 bias=False):
        super(MPRNet, self).__init__()

        act = nn.PReLU()
        self.in_c = in_c
        self.out_c = out_c
        
        self.shallow_feat1 = nn.Sequential(
            nn.Conv2d(in_c, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.shallow_feat2 = nn.Sequential(
            nn.Conv2d(in_c, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.shallow_feat3 = nn.Sequential(
            nn.Conv2d(in_c, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        # concat12: n_feat*2 -> n_feat (for stage2_encoder input)
        self.concat12 = nn.Conv2d(n_feat * 2, n_feat, kernel_size, padding=kernel_size//2, bias=bias)
        # concat23: n_feat*2 -> n_feat+scale_orsnetfeats (for ORSNet input)
        self.concat23 = nn.Conv2d(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, padding=kernel_size//2, bias=bias)
        self.tail = nn.Conv2d(n_feat + scale_orsnetfeats, out_c, kernel_size, padding=kernel_size//2, bias=bias)

    def forward(self, x3_img):
        # For mask input (4 channels), we use first 3 for image restoration
        # The mask channel provides additional guidance
        if x3_img.shape[1] == 4:
            # Split into RGB and mask
            x_rgb = x3_img[:, :3, :, :]
            # Mask is used implicitly through concatenated input
            x3_img_input = x3_img
        else:
            x_rgb = x3_img
            x3_img_input = x3_img

        H = x_rgb.size(2)
        W = x_rgb.size(3)

        # Multi-Patch Hierarchy: Split image into four patches
        x2top_img = x_rgb[:, :, 0:int(H/2), :]
        x2bot_img = x_rgb[:, :, int(H/2):H, :]
        
        if x3_img_input.shape[1] == 4:
            x2top_input = x3_img_input[:, :, 0:int(H/2), :]
            x2bot_input = x3_img_input[:, :, int(H/2):H, :]
        else:
            x2top_input = x2top_img
            x2bot_input = x2bot_img

        x1ltop_img = x2top_img[:, :, :, 0:int(W/2)]
        x1rtop_img = x2top_img[:, :, :, int(W/2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W/2)]
        x1rbot_img = x2bot_img[:, :, :, int(W/2):W]

        if x3_img_input.shape[1] == 4:
            x1ltop_input = x2top_input[:, :, :, 0:int(W/2)]
            x1rtop_input = x2top_input[:, :, :, int(W/2):W]
            x1lbot_input = x2bot_input[:, :, :, 0:int(W/2)]
            x1rbot_input = x2bot_input[:, :, :, int(W/2):W]
        else:
            x1ltop_input = x1ltop_img
            x1rtop_input = x1rtop_img
            x1lbot_input = x1lbot_img
            x1rbot_input = x1rbot_img

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        # Four patches for Stage 1
        feat1_ltop = self.shallow_feat1(x1ltop_input)
        feat1_rtop = self.shallow_feat1(x1rtop_input)
        feat1_lbot = self.shallow_feat1(x1lbot_input)
        feat1_rbot = self.shallow_feat1(x1rbot_input)

        # Encoder
        feat1_ltop = self.stage1_encoder(feat1_ltop)
        feat1_rtop = self.stage1_encoder(feat1_rtop)
        feat1_lbot = self.stage1_encoder(feat1_lbot)
        feat1_rbot = self.stage1_encoder(feat1_rbot)

        # Decoder
        res1_ltop = self.stage1_decoder(feat1_ltop)
        res1_rtop = self.stage1_decoder(feat1_rtop)
        res1_lbot = self.stage1_decoder(feat1_lbot)
        res1_rbot = self.stage1_decoder(feat1_rbot)

        # Supervised attention module
        sam_feats_ltop, stage1_img_ltop = self.sam12(res1_ltop[0], x1ltop_img)
        sam_feats_rtop, stage1_img_rtop = self.sam12(res1_rtop[0], x1rtop_img)
        sam_feats_lbot, stage1_img_lbot = self.sam12(res1_lbot[0], x1lbot_img)
        sam_feats_rbot, stage1_img_rbot = self.sam12(res1_rbot[0], x1rbot_img)

        # Concatenate to get stage1_img for the whole image
        stage1_img_top = torch.cat([stage1_img_ltop, stage1_img_rtop], 3)
        stage1_img_bot = torch.cat([stage1_img_lbot, stage1_img_rbot], 3)
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        # Concat top and bottom parts
        sam_feats_top = torch.cat([sam_feats_ltop, sam_feats_rtop], 3)
        sam_feats_bot = torch.cat([sam_feats_lbot, sam_feats_rbot], 3)

        # Combine features from patches of different encoders and decoders
        feat1_top = [torch.cat([feat1_ltop[j], feat1_rtop[j]], 3) for j in range(len(feat1_ltop))]
        feat1_bot = [torch.cat([feat1_lbot[j], feat1_rbot[j]], 3) for j in range(len(feat1_lbot))]
        res1_top = [torch.cat([res1_ltop[j], res1_rtop[j]], 3) for j in range(len(res1_ltop))]
        res1_bot = [torch.cat([res1_lbot[j], res1_rbot[j]], 3) for j in range(len(res1_lbot))]

        # Two patches for Stage 2
        feat2_top = self.shallow_feat2(x2top_input)
        feat2_bot = self.shallow_feat2(x2bot_input)

        # Concatenate SAM features from Stage1
        feat2_top = self.concat12(torch.cat([feat2_top, sam_feats_top], 1))
        feat2_bot = self.concat12(torch.cat([feat2_bot, sam_feats_bot], 1))

        # Encoder
        feat2_top = self.stage2_encoder(feat2_top, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(feat2_bot, feat1_bot, res1_bot)

        # Decoder
        res2_top = self.stage2_decoder(feat2_top)
        res2_bot = self.stage2_decoder(feat2_bot)

        # Supervised attention module
        sam_feats_top, stage2_img_top = self.sam23(res2_top[0], x2top_img)
        sam_feats_bot, stage2_img_bot = self.sam23(res2_bot[0], x2bot_img)

        # Concatenate to get stage2_img for the whole image
        stage2_img = torch.cat([stage2_img_top, stage2_img_bot], 2)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        # Concat top and bottom parts
        sam_feats = torch.cat([sam_feats_top, sam_feats_bot], 2)

        # Combine features from top/bottom encoders and decoders
        feat2 = [torch.cat([feat2_top[j], feat2_bot[j]], 2) for j in range(len(feat2_top))]
        res2 = [torch.cat([res2_top[j], res2_bot[j]], 2) for j in range(len(res2_top))]

        # One patch for Stage 3
        feat3 = self.shallow_feat3(x3_img_input)
        feat3 = self.concat23(torch.cat([feat3, sam_feats], 1))

        feat3 = self.stage3_orsnet(feat3, feat2, res2)
        stage3_img = self.tail(feat3)
        stage3_img = stage3_img + x_rgb

        return [stage3_img, stage2_img, stage1_img]


##########################################################################
## MPRNet-Lite (Smaller version for faster training)
class MPRNetLite(nn.Module):
    """
    MPRNet-Lite: A lighter version of MPRNet
    Single stage encoder-decoder with channel attention
    """
    def __init__(self,
                 in_c=3,
                 out_c=3,
                 n_feat=40,
                 scale_unetfeats=20,
                 num_cab=4,
                 kernel_size=3,
                 reduction=4,
                 bias=False):
        super(MPRNetLite, self).__init__()

        act = nn.PReLU()
        self.in_c = in_c
        self.out_c = out_c

        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_c, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.tail = nn.Conv2d(n_feat, out_c, kernel_size, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        if x.shape[1] == 4:
            x_rgb = x[:, :3, :, :]
        else:
            x_rgb = x

        feat = self.shallow_feat(x)
        enc_outs = self.encoder(feat)
        dec_outs = self.decoder(enc_outs)
        out = self.tail(dec_outs[0])
        out = out + x_rgb

        return out


if __name__ == '__main__':
    # Test the model
    height = 128
    width = 128
    
    # Test MPRNet
    model = MPRNet(in_c=4, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=4)
    print(f'Model: MPRNet')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    x = torch.randn((1, 4, height, width))
    outputs = model(x)
    print(f'Input shape: (1, 4, {height}, {width})')
    print(f'Output shapes: {[o.shape for o in outputs]}')
    
    # Test MPRNetLite
    model_lite = MPRNetLite(in_c=4, out_c=3, n_feat=40, scale_unetfeats=20, num_cab=4)
    print(f'\nModel: MPRNetLite')
    print(f'Number of parameters: {sum(p.numel() for p in model_lite.parameters()) / 1e6:.2f}M')
    
    x = torch.randn((1, 4, height, width))
    output = model_lite(x)
    print(f'Input shape: (1, 4, {height}, {width})')
    print(f'Output shape: {output.shape}')
