import random
from pathlib import Path

import cubo
import numpy as np

from heatmit.super_resolution import super_resolve, s2_to_rgb
from heatmit.vl_processing import describe_scene
from heatmit.rag_lm import RAG, suggest_mitigations

OUT = Path("/nfs/home/innocenti/heat-mitigation-suggestion/outputs")
CROP = 128

# Turin bounding box (matched to resources/sent2.tif extent)
CUBO_LAT = 45.073522
CUBO_LON = 7.675608
CUBO_EDGE = 1539       # pixels at 10 m/px → ~15.4 km x 15.4 km
CUBO_BANDS = ["B04", "B03", "B02", "B08"]
CUBO_START = "2023-07-01"
CUBO_END = "2023-07-31"
CUBO_TIME_IDX = None   # None = auto-select first 0%-NaN index


def _load_s2() -> np.ndarray:
    """Download one S2 L2A acquisition for Turin via cubo."""
    print(f"Downloading S2 L2A ({CUBO_START} – {CUBO_END}) via cubo...")
    da = cubo.create(
        lat=CUBO_LAT,
        lon=CUBO_LON,
        collection="sentinel-2-l2a",
        bands=CUBO_BANDS,
        start_date=CUBO_START,
        end_date=CUBO_END,
        edge_size=CUBO_EDGE,
        resolution=10,
    )
    print(f"  {len(da.time)} acquisitions found, scanning for clean tile...")
    idx = CUBO_TIME_IDX
    if idx is None:
        for i in range(len(da.time)):
            nan_pct = np.isnan(da[i].compute().to_numpy()).mean()
            if nan_pct == 0.0:
                idx = i
                break
        if idx is None:
            idx = 0
    print(f"  using index {idx} ({da.time[idx].values})")
    raw = da[idx].compute().to_numpy()
    arr = (raw / 10_000).astype(np.float32)  # (4, H, W) reflectance [0,1]
    arr = np.nan_to_num(arr, nan=0.0)
    print(f"  reflectance range: min={arr.min():.4f}, max={arr.max():.4f}")
    return arr  # (bands, H, W) in [B04, B03, B02, B08] order


def main() -> None:
    # --- 1. Download S2 ---
    s2_data = _load_s2()
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

    # --- 4. Super-resolve S2 (4x, LDSR-S2) and save PNG ---
    print("Super-resolving S2 patch (4x, LDSR-S2, 500 diffusion steps)...")
    s2_x4 = super_resolve(s2_patch, sampling_steps=500)

    s2_x4_png = OUT / f"{row}_{col}_s2_x4.png"
    s2_to_rgb(s2_x4).save(s2_x4_png)
    print(f"Saved: {s2_x4_png}")

    # --- 5. Describe scene ---
    print("Describing scene with Qwen2.5-VL-7B...")
    description = describe_scene(s2_patch, s2_x4=s2_x4)
    print(f"\nScene description:\n{description}\n")

    desc_path = OUT / f"{row}_{col}_description.txt"
    desc_path.write_text(description)
    print(f"Saved: {desc_path}")

    # --- 6. Build RAG from web + PDFs and generate mitigations ---
    print("\nBuilding RAG index from EU sources...")
    rag = RAG()

    print("\nGenerating mitigation suggestions with Qwen2.5-7B...")
    suggestions = suggest_mitigations(description, rag)
    print(f"\nMitigation suggestions:\n{suggestions}")

    mit_path = OUT / f"{row}_{col}_mitigation.txt"
    mit_path.write_text(suggestions)
    print(f"Saved: {mit_path}")


if __name__ == "__main__":
    main()
