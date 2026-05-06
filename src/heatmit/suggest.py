from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _get_device() -> str:
    try:
        torch.zeros(1).cuda()
        return "cuda"
    except RuntimeError:
        return "cpu"


_OPENSR_CONFIG_URL = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"


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


def _super_resolve(s2_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    print("SR pass 1/2 (4×)...")
    x4 = _run_sr(sr_model, s2_patch, device, steps=500)
    print("SR pass 2/2 (4×)...")
    x16 = _run_sr(sr_model, x4, device, steps=500)

    del sr_model
    torch.cuda.empty_cache()
    return x4, x16


def _s1_to_grayscale(vv: np.ndarray) -> Image.Image:
    """Convert S1 VV band (linear power) to dB grayscale image."""
    db = 10.0 * np.log10(np.maximum(vv.astype(np.float64), 1e-10))
    p2, p98 = np.percentile(db, 2), np.percentile(db, 98)
    db_norm = np.clip((db - p2) / (p98 - p2 + 1e-9), 0, 1)
    return Image.fromarray((db_norm * 255).astype(np.uint8), mode="L")


def _s2_to_rgb(s2_patch: np.ndarray) -> Image.Image:
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

    # # sRGB transfer function (perceptual gamma encoding)
    # rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)

    return Image.fromarray((rgb * 255).astype(np.uint8))


def describe_scene(
    s2_patch: np.ndarray,
    s2_x4: np.ndarray | None = None,
    s2_x16: np.ndarray | None = None,
    prompt: str | None = None,
) -> str:
    """
    Generates a natural language description using Qwen2.5-VL-7B.
    Accepts up to three resolutions of the same patch (original, 4×, 16× SR).

    Args:
        s2_patch: original S2 crop, shape (H, W, 4) — bands [B02, B03, B04, B08]
        s2_x4:   4× SR patch (H*4, W*4, 4), optional
        s2_x16:  16× SR patch (H*16, W*16, 4), optional
        prompt:  Instruction sent to the model. Built automatically if None.

    Returns:
        str: Natural language description of the scene
    """
    n_images = 1 + (s2_x4 is not None) + (s2_x16 is not None)
    if prompt is None:
        if n_images == 3:
            prompt = (
                "You are analyzing satellite imagery of the metropolitan area of Turin, northern Italy "
                "(Piedmont region, Po Plain), covering approximately 2.56 km × 2.56 km. "
                "You are provided with three versions of the same area at increasing resolution: "
                "Image 1: original Sentinel-2 at 20 m/px (128 × 128 px). "
                "Image 2: 4× super-resolved at 5 m/px (512 × 512 px). "
                "Image 3: 16× super-resolved at 1.25 m/px (2048 × 2048 px, shown at 1024 × 1024). "
                "All are true-color RGB composites (bands B04/B03/B02, ESA Copernicus L2A). "
                "Using all three images together, describe in detail the land cover types, urban morphology, "
                "vegetation presence, infrastructure, and any other visible features. "
                "Compute the percentage of occupation of each land cover type and the types of buildings present."
            )
        else:
            prompt = (
                "You are analyzing satellite imagery of the metropolitan area of Turin, northern Italy "
                "(Piedmont region, Po Plain), covering approximately 2.56 km × 2.56 km "
                "(2048 × 2048 pixels at 1.25 m/px after 16× super-resolution from 20 m Sentinel-2). "
                "The image is a true-color RGB composite (bands B04/B03/B02, ESA Copernicus L2A). "
                "Describe in detail the land cover types, urban morphology, vegetation presence, "
                "infrastructure, and any other visible features. "
                "Compute the percentage of occupation of each land cover type and the types of buildings present."
            )

    device = _get_device()
    print(f"VLM running on: {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def _to_image(patch: np.ndarray) -> Image.Image:
        img = _s2_to_rgb(patch)
        if max(img.size) > 1024:
            img = img.resize((1024, 1024), Image.LANCZOS)
        return img

    content = [{"type": "image", "image": _to_image(s2_patch)}]
    if s2_x4 is not None:
        content.append({"type": "image", "image": _to_image(s2_x4)})
    if s2_x16 is not None:
        content.append({"type": "image", "image": _to_image(s2_x16)})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2000)

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    del model, processor, inputs
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    import random
    from pathlib import Path
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    S2_TIF = Path("/nfs/home/innocenti/heat-mitigation-suggestion/resources/sent2.tif")
    OUT = Path("/nfs/home/innocenti/heat-mitigation-suggestion/outputs")
    CROP = 128  # 128 px × 10 m = 1280 m; after 4× SR → 512 px at 2.5 m
    TARGET_CRS = CRS.from_epsg(32632)
    TARGET_RES = 20.0

    def _reproject_tif(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            t, w, h = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height,
                *src.bounds, resolution=TARGET_RES,
            )
            out = np.zeros((src.count, h, w), dtype=np.float32)
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=out[band - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=t,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                )
        return out

    print("Reprojecting resources/sent2.tif...")
    s2_data = _reproject_tif(S2_TIF)
    _, height, width = s2_data.shape

    col = random.randint(0, width - CROP)
    row = random.randint(0, height - CROP)
    s2_patch = np.transpose(s2_data[:, row:row + CROP, col:col + CROP], (1, 2, 0))

    OUT.mkdir(exist_ok=True)

    s2_png = OUT / f"{row}_{col}_s2.png"
    _s2_to_rgb(s2_patch).save(s2_png)
    print(f"Saved: {s2_png}")

    print("Super-resolving S2 patch (16×)...")
    s2_x4, s2_x16 = _super_resolve(s2_patch)

    s2_x4_png = OUT / f"{row}_{col}_s2_x4.png"
    _s2_to_rgb(s2_x4).save(s2_x4_png)
    print(f"Saved: {s2_x4_png}")

    s2_x16_png = OUT / f"{row}_{col}_s2_x16.png"
    _s2_to_rgb(s2_x16).save(s2_x16_png)
    print(f"Saved: {s2_x16_png}")

    description = describe_scene(s2_patch, s2_x4=s2_x4, s2_x16=s2_x16)
    print(f"\n{description}")

    txt_path = OUT / f"{row}_{col}_description.txt"
    txt_path.write_text(description)
    print(f"Saved: {txt_path}")
