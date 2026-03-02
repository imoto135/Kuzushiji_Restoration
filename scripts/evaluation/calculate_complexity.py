"""
calculate_complexity.py

4つの Image Restoration モデル (NAFNet, SwinIR, MPRNet, Restormer) の
以下の指標を計算するスクリプト:
  - パラメータ数 (M)
  - FLOPs (G) [Giga MACs, thop 使用]
  - 推論時間 (ms/image, GPU/CPU)

使い方:
  cd /home/imoto/Kuzushiji_Restoration
  conda activate nafnet2
  python scripts/evaluation/calculate_complexity.py
"""

import sys
import os
import time
import json

import torch
import torch.nn as nn

# ================== sys.path: basicsr パッケージが見えるように ==================
# models/nafnet の中に basicsr/ が存在するので、その親ディレクトリを path に追加
BASE_DIR = "/home/imoto/Kuzushiji_Restoration"
NAFNET_ROOT    = os.path.join(BASE_DIR, "models/nafnet")      # basicsr が入っている
RESTORMER_ROOT = os.path.join(BASE_DIR, "models/restormer")  # 別の basicsr が入っている

# NAFNet の basicsr を最優先で参照
sys.path.insert(0, NAFNET_ROOT)

# ================== NAFNet / SwinIR / MPRNet のインポート (nafnet の basicsr から) ==================
from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.models.archs.SwinIR_arch import SwinIR
from basicsr.models.archs.MPRNet_arch  import MPRNet

# ================== Restormer: 専用の basicsr から ==================
# NAFNet の basicsr と衝突しないよう importlib で直接ロード
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    # restormer の basicsr を sys.path に挟み込んでから exec
    sys.path.insert(0, RESTORMER_ROOT)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)  # 元に戻す
    return mod

restormer_mod = _load_module(
    "restormer_arch",
    os.path.join(RESTORMER_ROOT, "basicsr/models/archs/restormer_arch.py")
)
Restormer = restormer_mod.Restormer

# ---- thop ----
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[WARNING] thop が未インストールです。pip install thop")


# ================== モデル設定 ==================
def build_nafnet():
    """NAFNet_NoMask_charb_percep.yml より"""
    return NAFNet(img_channel=3, width=32,
                  enc_blk_nums=[2,2,4,8], middle_blk_num=12, dec_blk_nums=[2,2,2,2])


def build_swinir():
    """SwinIR_NoMask_charb_percep.yml より"""
    return SwinIR(img_size=128, patch_size=1, in_chans=3,
                  embed_dim=60, depths=[6,6,6,6], num_heads=[6,6,6,6],
                  window_size=8, mlp_ratio=2.0,
                  upscale=1, img_range=1.0, upsampler='')


def build_mprnet():
    """MPRNet_NoMask_charb_percep.yml より"""
    return MPRNet(in_c=3, out_c=3, n_feat=40,
                  scale_unetfeats=20, scale_orsnetfeats=16,
                  num_cab=4, kernel_size=3, reduction=4, bias=False)


def build_restormer():
    """restormer_config.yml (inp_channels=3 for NoMask) より"""
    return Restormer(inp_channels=3, out_channels=3, dim=48,
                     num_blocks=[4,6,6,8], num_refinement_blocks=4,
                     heads=[1,2,4,8], ffn_expansion_factor=2.66,
                     bias=False, LayerNorm_type='WithBias')


# ================== 計算ユーティリティ ==================
def count_params_M(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_time(model, dummy, n_warmup=10, n_repeat=100, device=None):
    model.eval()
    dummy = dummy.to(device)
    if device.type == 'cuda':
        with torch.no_grad():
            for _ in range(n_warmup): model(dummy)
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        times = []
        with torch.no_grad():
            for _ in range(n_repeat):
                start_ev.record(); model(dummy); end_ev.record()
                torch.cuda.synchronize()
                times.append(start_ev.elapsed_time(end_ev))
    else:
        with torch.no_grad():
            for _ in range(n_warmup): model(dummy)
        times = []
        with torch.no_grad():
            for _ in range(n_repeat):
                t0 = time.perf_counter(); model(dummy)
                times.append((time.perf_counter()-t0)*1000)
    return sum(times)/len(times)


def eval_model(name, build_fn, input_size, device):
    print(f"\n{'='*60}")
    print(f"  [{name}]")
    print(f"{'='*60}")
    try:
        model = build_fn().to(device); model.eval()
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"model": name, "error": str(e)}

    dummy = torch.randn(*input_size).to(device)

    params = count_params_M(model)
    print(f"  Params   : {params:.2f} M")

    flops_G = None
    if HAS_THOP:
        try:
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            flops_G = macs / 1e9
            print(f"  FLOPs    : {flops_G:.2f} G")
        except Exception as e:
            print(f"  FLOPs    : 計算失敗 ({e})")
        finally:
            # thop が各レイヤーに追加した フック/属性を削除して inference 計測を正常化
            for m in model.modules():
                if hasattr(m, 'total_ops'):  delattr(m, 'total_ops')
                if hasattr(m, 'total_params'): delattr(m, 'total_params')

    infer_ms = None
    try:
        infer_ms = measure_time(model, dummy, device=device)
        print(f"  Infer    : {infer_ms:.2f} ms/image  (batch=1, {input_size[2]}x{input_size[3]})")
    except Exception as e:
        print(f"  Infer    : 計測失敗 ({e})")

    return {
        "model": name,
        "params_M": round(params, 2),
        "flops_G":  round(flops_G, 2)  if flops_G  is not None else None,
        "infer_ms": round(infer_ms, 2) if infer_ms is not None else None,
        "input_HxW": f"{input_size[2]}x{input_size[3]}",
        "device": str(device),
    }


# ================== メイン ==================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = (1, 3, 256, 256)

    print(f"\n{'='*60}")
    print(f"  モデル計算量測定")
    print(f"  Device    : {device}")
    print(f"  Input     : {INPUT_SIZE}")
    print(f"  FLOPs計算 : thop (MACs = Multiply-Accumulate Operations)")
    print(f"  推論時間  : GPU Event (warmup={10}, repeat={100})")
    print(f"{'='*60}")

    configs = [
        ("NAFNet",    build_nafnet),
        ("SwinIR",    build_swinir),
        ("MPRNet",    build_mprnet),
        ("Restormer", build_restormer),
    ]

    results = [eval_model(n, f, INPUT_SIZE, device) for n, f in configs]

    # サマリー表示
    print(f"\n\n{'='*68}")
    print(f"  {'Model':<14}  {'Params (M)':>12}  {'FLOPs (G)':>12}  {'Infer (ms)':>12}")
    print(f"  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*12}")
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<14}  ERROR: {r['error'][:35]}")
        else:
            print(f"  {r['model']:<14}  "
                  f"{str(r.get('params_M','N/A')):>12}  "
                  f"{str(r.get('flops_G','N/A')):>12}  "
                  f"{str(r.get('infer_ms','N/A')):>12}")
    print(f"{'='*68}")

    # JSON 保存
    out_json = os.path.join(BASE_DIR, "outputs/restults_charbpercep/model_complexity.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {out_json}")
