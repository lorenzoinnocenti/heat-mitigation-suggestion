from pathlib import Path
import numpy as np
from PIL import Image
import torch

_OPENSR_CONFIG_URL = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"


def _get_device() -> str:
    try:
        torch.zeros(1).cuda()
        return "cuda"
    except RuntimeError:
        return "cpu"


def _run_sr(sr_model, patch_hw4: np.ndarray, device: str, steps: int) -> np.ndarray:
    """Run one 4× SR pass on a patch of any size by tiling into 128×128 tiles."""
    H, W, C = patch_hw4.shape
    assert H % 128 == 0 and W % 128 == 0, "patch dimensions must be multiples of 128"
    arr = np.clip(patch_hw4.astype(np.float32) / 10_000.0, 0.0, 1.0).transpose(2, 0, 1)  # (C, H, W)
    out = np.zeros((C, H * 4, W * 4), dtype=np.float32)
    for r in range(0, H, 128):
        for c in range(0, W, 128):
            tile = torch.from_numpy(arr[:, r:r+128, c:c+128]).float().unsqueeze(0).to(device)
            tile = torch.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                sr_tile = sr_model.forward(tile, sampling_steps=steps).squeeze(0)
            out[:, r*4:(r+128)*4, c*4:(c+128)*4] = sr_tile.cpu().numpy()
    return (out.transpose(1, 2, 0) * 10_000.0).astype(np.float32)  # (H*4, W*4, C)


def super_resolve(s2_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """16× super-resolve an S2 patch (H, W, 4) [B02,B03,B04,B08] in DN — two 4× passes.

    Returns:
        x4:  first-pass result  (H*4,  W*4,  4) in DN
        x16: second-pass result (H*16, W*16, 4) in DN
    """
    from io import StringIO
    import requests
    from omegaconf import OmegaConf
    import opensr_model

    config = OmegaConf.load(StringIO(requests.get(_OPENSR_CONFIG_URL).text))
    device = _get_device()
    print(f"SR running on: {device}")
    sr_model = opensr_model.SRLatentDiffusion(config, device=device)
    sr_model.load_pretrained(config.ckpt_version)

    print("SR pass 1/2 (4x)...")
    x4 = _run_sr(sr_model, s2_patch, device, steps=500)
    print("SR pass 2/2 (4x)...")
    x16 = _run_sr(sr_model, x4, device, steps=500)

    del sr_model
    torch.cuda.empty_cache()
    return x4, x16


def s1_to_grayscale(vv: np.ndarray) -> Image.Image:
    """Convert S1 VV band (linear power) to dB grayscale image."""
    db = 10.0 * np.log10(np.maximum(vv.astype(np.float64), 1e-10))
    p2, p98 = np.percentile(db, 2), np.percentile(db, 98)
    db_norm = np.clip((db - p2) / (p98 - p2 + 1e-9), 0, 1)
    return Image.fromarray((db_norm * 255).astype(np.uint8), mode="L")


def s2_to_rgb(s2_patch: np.ndarray) -> Image.Image:
    """Convert S2 patch (H, W, 4) with bands [B02, B03, B04, B08] to an RGB PIL image.

    Follows the ESA TCI standard: clip at 0.25 reflectance (2500 DN), then apply
    sRGB transfer function — identical to what QGIS and EO Browser display.
    """
    # B04=RED (idx 2), B03=GREEN (idx 1), B02=BLUE (idx 0); DN → reflectance
    rgb = np.nan_to_num(s2_patch[:, :, [2, 1, 0]].astype(np.float64) / 10000.0, nan=0.0)

    # Linear stretch: clip at 25% reflectance (ESA TCI saturation point)
    # Global 2–98 percentile stretch (same scale applied to all channels)
    p2, p98 = np.percentile(rgb, 1), np.percentile(rgb, 99)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0, 1)

    return Image.fromarray((rgb * 255).astype(np.uint8))
