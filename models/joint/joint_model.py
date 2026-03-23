"""
joint_model.py — JointRestorationNet (Adaptive Soft Mask)

End-to-end differentiable pipeline:
  LQ (RGB) → UNet++ → logits → sigmoid(logits / τ) → soft mask [0,1]
           → cat([LQ, mask], dim=1) → NAFNet → restored (RGB)

Key additions over the original:
  - Learnable temperature τ: controls mask sharpness, updated via gradient
  - freeze_unetpp() / unfreeze_unetpp() helpers for progressive unfreezing
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Path setup — allow importing from sibling dirs without installing packages
# --------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent          # models/joint/
_MODELS    = _THIS_DIR.parent                          # models/
_NAFNET    = _MODELS / "nafnet"
_UNETPP    = _MODELS / "unet++"

for _p in [str(_NAFNET), str(_UNETPP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# UNet++ via segmentation_models_pytorch
import segmentation_models_pytorch as smp

# NAFNet arch (BasicSR local install)
from basicsr.models.archs.NAFNet_arch import NAFNet


# --------------------------------------------------------------------------
# ImageNet statistics for UNet++ encoder normalisation
# (smp models expect ImageNet-pre-normalised input)
# --------------------------------------------------------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class JointRestorationNet(nn.Module):
    """
    Combined Stage-1 (UNet++) + Stage-2 (NAFNet) network with
    learnable temperature scaling for the soft mask.

    Forward pass:
        lq   : (B, 3, H, W)  — degraded image, values in [0, 1]
    Returns:
        restored : (B, 3, H, W)  — restored image, values in [0, 1]
        mask     : (B, 1, H, W)  — predicted soft damage mask in [0, 1]
        temp     : float          — current temperature τ value (for logging)
    """

    # Default NAFNet config matching the trained Kuzushiji mask model
    NAFNET_DEFAULTS = dict(
        width=32,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
        img_channel=4,
        out_channel=3,
    )

    def __init__(
        self,
        unetpp_pretrain: Optional[str] = None,
        nafnet_pretrain: Optional[str] = None,
        nafnet_cfg: Optional[dict] = None,
        freeze_unetpp: bool = False,
        freeze_nafnet: bool = False,
        learnable_temp: bool = True,
        temp_init: float = 1.0,
        temp_min: float = 0.1,
    ):
        super().__init__()

        # ---- Stage 1: UNet++ ----
        self.unetpp = smp.UnetPlusPlus(
            encoder_name="se_resnext50_32x4d",
            encoder_weights=None,          # we load our own weights
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )

        # ---- Stage 2: NAFNet ----
        cfg = self.NAFNET_DEFAULTS | (nafnet_cfg or {})
        self.nafnet = NAFNet(**cfg)

        # ---- Learnable temperature τ ----
        # Controls mask sharpness: mask = sigmoid(logits / τ)
        # Small τ → sharp mask (near binary), Large τ → soft/smooth mask
        self.temp_min = temp_min
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(float(temp_init)).log())
        else:
            self.register_buffer("log_temp", torch.tensor(float(temp_init)).log())

        # ---- Load pretrained weights ----
        if unetpp_pretrain:
            state = torch.load(unetpp_pretrain, map_location="cpu")
            self.unetpp.load_state_dict(state, strict=True)
            print(f"[JointModel] Loaded UNet++ weights from {unetpp_pretrain}")

        if nafnet_pretrain:
            state = torch.load(nafnet_pretrain, map_location="cpu")
            # BasicSR saves weights under a 'params' key
            if "params" in state:
                state = state["params"]
            self.nafnet.load_state_dict(state, strict=True)
            print(f"[JointModel] Loaded NAFNet weights from {nafnet_pretrain}")

        # ---- Optional freezing ----
        if freeze_unetpp:
            self.freeze_unetpp()
        if freeze_nafnet:
            self.freeze_nafnet()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (used by progressive unfreezing scheduler)
    # ------------------------------------------------------------------

    def freeze_unetpp(self) -> None:
        """Freeze all UNet++ parameters (Stage 1 warmup phase)."""
        for p in self.unetpp.parameters():
            p.requires_grad = False
        print("[JointModel] UNet++ frozen.")

    def unfreeze_unetpp(self) -> None:
        """Unfreeze all UNet++ parameters."""
        for p in self.unetpp.parameters():
            p.requires_grad = True
        print("[JointModel] UNet++ unfrozen.")

    def freeze_nafnet(self) -> None:
        """Freeze all NAFNet parameters."""
        for p in self.nafnet.parameters():
            p.requires_grad = False
        print("[JointModel] NAFNet frozen.")

    def unfreeze_nafnet(self) -> None:
        """Unfreeze all NAFNet parameters."""
        for p in self.nafnet.parameters():
            p.requires_grad = True
        print("[JointModel] NAFNet unfrozen.")

    # ------------------------------------------------------------------
    # Register ImageNet buffers so they move with .to(device) / .cuda()
    # ------------------------------------------------------------------
    def _norm_for_unetpp(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise [0,1] RGB to ImageNet statistics in-graph (differentiable)."""
        mean = _IMAGENET_MEAN.to(x.device, x.dtype)
        std  = _IMAGENET_STD.to(x.device, x.dtype)
        return (x - mean) / std

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature τ (always > temp_min, in original scale)."""
        return self.log_temp.exp().clamp(min=self.temp_min)

    def forward(self, lq: torch.Tensor):
        """
        Args:
            lq: (B, 3, H, W) in [0, 1]

        Returns:
            restored: (B, 3, H, W)
            mask:     (B, 1, H, W) in [0, 1]  — the soft damage mask
            temp:     float                    — current temperature τ (for logging)
        """
        # Stage 1 — UNet++ expects ImageNet-normalised input
        lq_norm  = self._norm_for_unetpp(lq)
        logits   = self.unetpp(lq_norm)           # (B, 1, H, W)

        # Adaptive mask: sigmoid(logits / τ)
        # τ is learned, controlling mask sharpness throughout training
        tau  = self.temperature                    # scalar tensor
        mask = torch.sigmoid(logits / tau)         # soft mask in [0, 1]

        # Stage 2 — NAFNet takes [lq_rgb, soft_mask] as 4-channel input
        x_in     = torch.cat([lq, mask], dim=1)  # (B, 4, H, W)
        restored = self.nafnet(x_in)              # (B, 3, H, W)

        return restored, mask, tau.item()


# --------------------------------------------------------------------------
# Quick sanity check
# --------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with learnable temp
    print("=== Test: learnable_temp=True ===")
    model = JointRestorationNet(
        unetpp_pretrain=None, nafnet_pretrain=None, learnable_temp=True
    ).to(device)
    model.eval()

    with torch.no_grad():
        dummy = torch.rand(2, 3, 128, 128).to(device)
        restored, mask, temp = model(dummy)

    print(f"restored : {restored.shape}  range [{restored.min():.3f}, {restored.max():.3f}]")
    print(f"mask     : {mask.shape}  range [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"temp (τ) : {temp:.4f}")
    print("Sanity check passed ✓")

    # Test freeze/unfreeze
    print("\n=== Test: freeze / unfreeze ===")
    model.freeze_unetpp()
    assert not any(p.requires_grad for p in model.unetpp.parameters()), "freeze failed"
    model.unfreeze_unetpp()
    assert all(p.requires_grad for p in model.unetpp.parameters()), "unfreeze failed"
    print("Freeze/unfreeze OK ✓")
