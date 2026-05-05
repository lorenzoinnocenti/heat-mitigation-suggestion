import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


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
    prompt: str = (
        "You are analyzing a satellite image captured by the Sentinel-2 MultiSpectral Instrument (MSI), "
        "operated by the European Space Agency (ESA) as part of the Copernicus Earth Observation programme. "
        "The image was acquired at Level-2A (L2A), meaning it has been atmospherically corrected to surface reflectance. "
        "It is rendered as a true-color RGB composite using bands B04 (Red, 665 nm), B03 (Green, 560 nm), and B02 (Blue, 490 nm), "
        "The scene covers an area of approximately 1.12 km × 1.12 km (224 × 224 pixels at 5 m/px). "
        "located in the metropolitan area of Turin, northern Italy (Piedmont region, Po Plain). "
        "Given this context, describe in detail the land cover types, urban morphology, vegetation presence, "
        "infrastructure, and any other visible features in this image."
        "Compute the percentage of occupation of each land cover type and which type of buildings there are."
    ),
) -> str:
    """
    Generates a natural language description of a Sentinel-2 patch using Qwen2.5-VL-3B.

    Args:
        s2_patch (np.ndarray): S2 L2A patch, shape (H, W, 4) — bands [B02, B03, B04, B08]
        prompt (str): Instruction sent to the model

    Returns:
        str: Natural language description of the scene
    """
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    image = _s2_to_rgb(s2_patch)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]


if __name__ == "__main__":
    import random
    from pathlib import Path
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    TIF = Path("/nfs/home/innocenti/heat-mitigation-suggestion/turin.tif")
    OUT = Path("/nfs/home/innocenti/heat-mitigation-suggestion/outputs")
    CROP = 224
    TARGET_CRS = CRS.from_epsg(32632)  # WGS 84 / UTM zone 32N (Turin)
    TARGET_RES = 5.0  # metres

    # Reproject and resample to 5 m/px in memory
    with rasterio.open(TIF) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height,
            *src.bounds, resolution=TARGET_RES,
        )
        profile = src.profile.copy()
        profile.update(crs=TARGET_CRS, transform=transform, width=width, height=height)

        reprojected = np.zeros((src.count, height, width), dtype=np.float32)
        for band in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=reprojected[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
            )

    col = random.randint(0, width - CROP)
    row = random.randint(0, height - CROP)
    patch = reprojected[:, row:row + CROP, col:col + CROP]  # (4, H, W)
    s2_patch = np.transpose(patch, (1, 2, 0))  # (H, W, 4)

    OUT.mkdir(exist_ok=True)

    png_path = OUT / f"crop_{row}_{col}.png"
    _s2_to_rgb(s2_patch).save(png_path)
    print(f"Saved: {png_path}")

    description = describe_scene(s2_patch)
    print(f"\n{description}")

    txt_path = OUT / f"crop_{row}_{col}.txt"
    txt_path.write_text(description)
    print(f"Saved: {txt_path}")
