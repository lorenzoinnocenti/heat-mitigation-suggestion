from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch
import requests
from omegaconf import OmegaConf

_LDSR_CONFIG_URL = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/tags/v1.1.1/opensr_model/configs/config_10m.yaml"
_LDSR_MODEL_DIR = Path(__file__).parent.parent.parent / "model" / "LDSR_S2_10m"

_MODEL_TILE = 128  # LDSR-S2 expects 128x128 input tiles


def _get_device() -> str:
    try:
        torch.zeros(1).cuda()
        return "cuda"
    except RuntimeError:
        return "cpu"


def _load_ldsr_model(device: str):
    import opensr_model

    _LDSR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config_path = _LDSR_MODEL_DIR / "config_10m.yaml"
    if not config_path.exists():
        print("Downloading LDSR-S2 config...")
        response = requests.get(_LDSR_CONFIG_URL, timeout=30)
        response.raise_for_status()
        config_path.write_text(response.text)

    config = OmegaConf.load(str(config_path))
    model = opensr_model.SRLatentDiffusion(config, device=device)

    # load_pretrained builds the HF URL from the bare filename, so cd to model dir
    # so it downloads/finds the ckpt there
    orig = os.getcwd()
    os.chdir(str(_LDSR_MODEL_DIR))
    try:
        model.load_pretrained(config.ckpt_version)
    finally:
        os.chdir(orig)

    return model


def _run_ldsr(model, patch: np.ndarray, device: str, label: str, sampling_steps: int) -> np.ndarray:
    """Single LDSR-S2 pass on a (H, W, 4) reflectance patch, tiled into _MODEL_TILEx_MODEL_TILE chunks."""
    H, W, C = patch.shape
    arr = patch.transpose(2, 0, 1).astype(np.float32)  # (C, H, W)
    out = np.zeros((C, H * 4, W * 4), dtype=np.float32)
    n_tiles = ((H + _MODEL_TILE - 1) // _MODEL_TILE) * ((W + _MODEL_TILE - 1) // _MODEL_TILE)
    tile_idx = 0
    print(f"SR {label} ({n_tiles} tiles, {sampling_steps} steps each)...")
    for r in range(0, H, _MODEL_TILE):
        for c in range(0, W, _MODEL_TILE):
            raw = arr[:, r:r + _MODEL_TILE, c:c + _MODEL_TILE]
            th, tw = raw.shape[1], raw.shape[2]
            if th < _MODEL_TILE or tw < _MODEL_TILE:
                raw = np.pad(raw, ((0, 0), (0, _MODEL_TILE - th), (0, _MODEL_TILE - tw)), mode="reflect")
            tile = torch.from_numpy(raw).float().unsqueeze(0).to(device)
            tile = torch.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
            sr_tile = model.forward(tile, sampling_steps=sampling_steps).squeeze(0).cpu().numpy()
            out[:, r * 4:r * 4 + th * 4, c * 4:c * 4 + tw * 4] = sr_tile[:, :th * 4, :tw * 4]
            tile_idx += 1
            print(f"  tile {tile_idx}/{n_tiles}", end="\r")
    print()
    return out.transpose(1, 2, 0).astype(np.float32)  # (H*4, W*4, C)


def super_resolve(s2_patch: np.ndarray, sampling_steps: int = 500) -> np.ndarray:
    """Single 4x LDM SR pass on an S2 patch (H, W, 4) [B04,B03,B02,B08] in reflectance [0,1].

    Uses LDSR-S2 (latent diffusion). Input at 10 m/px -> output at 2.5 m/px.

    Returns:
        x4: SR result (H*4, W*4, 4) reflectance
    """
    device = _get_device()
    print(f"SR running on: {device}")
    model = _load_ldsr_model(device)

    x4 = _run_ldsr(model, s2_patch, device, "10m -> 2.5m", sampling_steps)

    del model
    torch.cuda.empty_cache()
    return x4


def s1_to_grayscale(vv: np.ndarray) -> Image.Image:
    """Convert S1 VV band (linear power) to dB grayscale image."""
    db = 10.0 * np.log10(np.maximum(vv.astype(np.float64), 1e-10))
    p2, p98 = np.percentile(db, 2), np.percentile(db, 98)
    db_norm = np.clip((db - p2) / (p98 - p2 + 1e-9), 0, 1)
    return Image.fromarray((db_norm * 255).astype(np.uint8), mode="L")


def s2_to_rgb(s2_patch: np.ndarray) -> Image.Image:
    """Convert S2 patch (H, W, 4) [B04, B03, B02, B08] in reflectance [0,1] to an RGB PIL image."""
    rgb = np.nan_to_num(s2_patch[:, :, :3].astype(np.float64), nan=0.0)  # B04=R, B03=G, B02=B
    p2, p98 = np.percentile(rgb, 1), np.percentile(rgb, 99)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))
