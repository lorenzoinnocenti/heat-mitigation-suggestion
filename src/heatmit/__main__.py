import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

from heatmit.super_resolution import super_resolve, s2_to_rgb
from heatmit.vl_processing import describe_scene
from heatmit.rag_lm import RAG, suggest_mitigations

S2_TIF = Path("/nfs/home/innocenti/heat-mitigation-suggestion/resources/sent2.tif")
OUT = Path("/nfs/home/innocenti/heat-mitigation-suggestion/outputs")
CROP = 128
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
    return out  # (bands, H, W)


def main() -> None:
    # --- 1. Load and reproject S2 ---
    print("Reprojecting resources/sent2.tif to EPSG:32632 at 20 m/px...")
    s2_data = _reproject_tif(S2_TIF)
    _, height, width = s2_data.shape

    # --- 2. Random 128x128 crop ---
    col = random.randint(0, width - CROP)
    row = random.randint(0, height - CROP)
    s2_patch = np.transpose(s2_data[:, row:row + CROP, col:col + CROP], (1, 2, 0))  # (H, W, 4)

    OUT.mkdir(exist_ok=True)

    # --- 3. Save original PNG ---
    s2_png = OUT / f"{row}_{col}_s2.png"
    s2_to_rgb(s2_patch).save(s2_png)
    print(f"Saved: {s2_png}")

    # --- 4. Super-resolve S2 (16x) and save PNGs ---
    print("Super-resolving S2 patch (16x)...")
    s2_x4, s2_x16 = super_resolve(s2_patch)

    s2_x4_png = OUT / f"{row}_{col}_s2_x4.png"
    s2_to_rgb(s2_x4).save(s2_x4_png)
    print(f"Saved: {s2_x4_png}")

    s2_x16_png = OUT / f"{row}_{col}_s2_x16.png"
    s2_to_rgb(s2_x16).save(s2_x16_png)
    print(f"Saved: {s2_x16_png}")

    # --- 5. Describe scene ---
    print("Describing scene with Qwen2.5-VL-7B...")
    description = describe_scene(s2_patch, s2_x4=s2_x4, s2_x16=s2_x16)
    print(f"\nScene description:\n{description}\n")

    desc_path = OUT / f"{row}_{col}_description.txt"
    desc_path.write_text(description)
    print(f"Saved: {desc_path}")

    # --- 6. Build RAG from web + PDFs and generate mitigations ---
    print("\nBuilding RAG index from EU sources...")
    rag = RAG()

    # # TODO: remove — provisional chunk inspection
    # print("\n=== Top 5 retrieved chunks ===")
    # for i, chunk in enumerate(rag.retrieve(description), 1):
    #     print(f"\n--- Chunk {i} ---\n{chunk[:500]}")
    # print("\n==============================\n")

    print("\nGenerating mitigation suggestions with Qwen2.5-7B...")
    suggestions = suggest_mitigations(description, rag)
    print(f"\nMitigation suggestions:\n{suggestions}")

    mit_path = OUT / f"{row}_{col}_mitigation.txt"
    mit_path.write_text(suggestions)
    print(f"Saved: {mit_path}")


if __name__ == "__main__":
    main()
