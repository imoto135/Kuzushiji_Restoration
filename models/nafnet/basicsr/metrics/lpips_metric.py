# ------------------------------------------------------------------------
# LPIPS metric for image restoration evaluation
# ------------------------------------------------------------------------
import torch
import lpips
import numpy as np


def calculate_lpips(img1, img2, net='alex', crop_border=0, input_order='HWC', **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img1 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        img2 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        net (str): Network type for LPIPS. Default: 'alex'.
            Options: 'alex', 'vgg', 'squeeze'.
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: LPIPS result (lower is better).
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    # Convert to numpy if tensor
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Normalize to [0, 1] if range is [0, 255]
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0

    if input_order == 'HWC':
        img1 = img1.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)

    if crop_border > 0:
        img1 = img1[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]

    # Convert to tensor: LPIPS expects [-1, 1]
    img1_tensor = torch.from_numpy(img1).float().unsqueeze(0) * 2.0 - 1.0
    img2_tensor = torch.from_numpy(img2).float().unsqueeze(0) * 2.0 - 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # Use global model to avoid re-loading every call
    if not hasattr(calculate_lpips, '_model') or calculate_lpips._net != net:
        calculate_lpips._model = lpips.LPIPS(net=net).to(device).eval()
        calculate_lpips._net = net

    with torch.no_grad():
        score = calculate_lpips._model(img1_tensor, img2_tensor)

    return score.item()
