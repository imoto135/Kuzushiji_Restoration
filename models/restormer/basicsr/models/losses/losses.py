import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with VGG19 features.
    
    Uses pre-trained VGG19 to extract features and computes L1/L2 loss
    between predicted and target feature maps.
    
    Args:
        layer_weights (dict): Layer name -> weight mapping.
        vgg_type (str): VGG type. Default: 'vgg19'.
        use_input_norm (bool): Normalize input to VGG range. Default: True.
        range_norm (bool): Normalize from [-1,1] to [0,1]. Default: False.
        perceptual_weight (float): Weight for perceptual loss. Default: 1.0.
        style_weight (float): Weight for style loss. Default: 0.
        criterion (str): 'l1' or 'l2'. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        
        # Build VGG feature extractor
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm
        )
        
        self.criterion_type = criterion
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'{criterion} is not supported.')

    def forward(self, x, gt):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).
            gt (Tensor): Ground-truth tensor with shape (N, C, H, W).
            
        Returns:
            Tensor: Perceptual loss.
            Tensor: Style loss (or None).
        """
        # Extract VGG features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        
        # Calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None
            
        # Calculate style loss (Gram matrix based)
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
            
        return percep_loss, style_loss
    
    def _gram_mat(self, x):
        """Calculate Gram matrix for style loss."""
        n, c, h, w = x.size()
        features = x.view(n, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.
    
    Args:
        layer_name_list (list[str]): Layers to extract features from.
        vgg_type (str): VGG type. Options: 'vgg11', 'vgg13', 'vgg16', 'vgg19'.
        use_input_norm (bool): Normalize input to VGG range.
        range_norm (bool): Normalize from [-1,1] to [0,1].
        requires_grad (bool): Whether to require gradients. Default: False.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False):
        super(VGGFeatureExtractor, self).__init__()
        
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        
        # VGG layer name to index mapping
        self.names = {
            'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
            'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
            'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13,
            'conv3_3': 14, 'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
            'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22,
            'conv4_3': 23, 'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
            'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31,
            'conv5_3': 32, 'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35, 'pool5': 36
        }
        
        # Load pretrained VGG
        import torchvision
        if vgg_type == 'vgg19':
            vgg_net = torchvision.models.vgg19(pretrained=True)
        elif vgg_type == 'vgg16':
            vgg_net = torchvision.models.vgg16(pretrained=True)
        else:
            raise NotImplementedError(f'{vgg_type} is not supported.')
        
        # Get max layer index
        max_idx = 0
        for layer_name in layer_name_list:
            idx = self.names[layer_name]
            if idx > max_idx:
                max_idx = idx
        
        # Only keep layers up to max_idx
        self.vgg_net = nn.Sequential(*list(vgg_net.features.children())[:max_idx + 1])
        
        # Freeze parameters
        if not requires_grad:
            self.vgg_net.eval()
            for param in self.vgg_net.parameters():
                param.requires_grad = False
        
        # Register normalization buffers
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Extract VGG features.
        
        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).
            
        Returns:
            dict: Feature maps from specified layers.
        """
        if self.range_norm:
            x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            
        output = {}
        for name, module in self.vgg_net.named_children():
            x = module(x)
            for target_name in self.layer_name_list:
                if int(name) == self.names[target_name]:
                    output[target_name] = x.clone()
                    
        return output
