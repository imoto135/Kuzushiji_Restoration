import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareLoss(nn.Module):
    """Edge-Aware Loss using Sobel operator for gradient computation.
    
    This loss computes the L1 difference between edge gradients of 
    predicted and ground truth images to enhance edge sharpness.
    
    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(EdgeAwareLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)
        
        # Register as buffers (will be moved to GPU automatically)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_gradients(self, img):
        """Compute image gradients using Sobel operator.
        
        Args:
            img (Tensor): Input image tensor (B, C, H, W)
            
        Returns:
            Tensor: Gradient magnitude (B, C, H, W)
        """
        # Convert to grayscale if RGB
        if img.size(1) == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
        
        # Compute gradients
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Gradient magnitude
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        return gradient

    def forward(self, pred, target):
        """Forward function.
        
        Args:
            pred (Tensor): Predicted image (B, C, H, W)
            target (Tensor): Ground truth image (B, C, H, W)
            
        Returns:
            Tensor: Calculated edge-aware loss
        """
        # Compute edge gradients
        pred_edges = self.compute_gradients(pred)
        target_edges = self.compute_gradients(target)
        
        # L1 loss on edge gradients
        loss = F.l1_loss(pred_edges, target_edges, reduction=self.reduction)
        
        return loss * self.loss_weight