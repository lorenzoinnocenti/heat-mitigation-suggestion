import numpy as np
import torch
import requests
import pdfplumber
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBED_MODEL_ID = "all-MiniLM-L6-v2"
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = Path("/nfs/home/innocenti/heat-mitigation-suggestion/data")

# Multi-source corpus: web pages + PDFs
WEB_SOURCES = [
    "https://climate-adapt.eea.europa.eu/en/eu-adaptation-policy/sector-policies/urban/index_html",
]

PDF_SOURCES = [
    {
        "url": "https://publications.jrc.ec.europa.eu/repository/bitstream/JRC137891/JRC137891_01.pdf",
        "filename": "jrc_eu_cities_heat_extremes.pdf",
    },
    {
        "url": "https://climate-adapt.eea.europa.eu/en/metadata/publications/cooling-the-cities-green-roof-mitigation-technologies-to-fight-heat-island-and-improve-comfort/11238649/@@download/file/11238649.pdf",
        "filename": "eea_cooling_cities_green_roofs.pdf",
    },
    {
        "url": "https://publications.jrc.ec.europa.eu/repository/bitstream/JRC115375/JRC115375_01.pdf",
        "filename": "jrc_green_infrastructure_urban_resilience.pdf",
    },
]


def _fetch_web_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _download_pdf(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {dest.name}...")
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"  Saved: {dest}")


def _extract_pdf_text(path: Path) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)


def _chunk(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        if chunk:
            chunks.append(chunk)
    return chunks


class RAG:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.embedder = SentenceTransformer(EMBED_MODEL_ID)
        DATA_DIR.mkdir(exist_ok=True)

        all_text = []

        print("Loading web sources...")
        for url in WEB_SOURCES:
            try:
                all_text.append(_fetch_web_text(url))
                print(f"  Fetched: {url}")
            except Exception as e:
                print(f"  Failed {url}: {e}")

        print("Loading PDF sources...")
        for source in PDF_SOURCES:
            dest = DATA_DIR / source["filename"]
            try:
                _download_pdf(source["url"], dest)
                all_text.append(_extract_pdf_text(dest))
            except Exception as e:
                print(f"  Failed {source['filename']}: {e}")

        self.chunks = _chunk("\n\n".join(all_text))
        print(f"Built corpus: {len(self.chunks)} chunks from {len(all_text)} sources")

        emb = self.embedder.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        self.embeddings = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    def retrieve(self, query: str) -> list[str]:
        q = self.embedder.encode([query], convert_to_numpy=True)
        q = q / (np.linalg.norm(q, keepdims=True) + 1e-8)
        scores = (self.embeddings @ q.T).squeeze()
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [self.chunks[i] for i in top_idx]


def suggest_mitigations(scene_description: str, rag: RAG) -> str:
    """
    Given a satellite scene description, retrieves relevant EU urban adaptation
    context from web and PDFs, then generates heat mitigation suggestions with Qwen2.5-7B.

    Args:
        scene_description: Text output from describe_scene()
        rag: Initialized RAG instance

    Returns:
        str: Actionable heat mitigation suggestions
    """
    context = "\n\n---\n\n".join(rag.retrieve(scene_description))

    from heatmit.suggest import _get_device
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an urban climate adaptation expert. "
                "Using the provided EU scientific and policy literature and a satellite scene description, "
                "suggest specific, actionable measures to reduce the urban heat island effect "
                "and lower local temperatures in the described zone."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## EU Urban Adaptation Literature (retrieved context)\n\n{context}\n\n"
                f"## Satellite Scene Description\n\n{scene_description}\n\n"
                "Based on the land cover and urban features visible in the satellite image "
                "and the scientific/policy context above, provide specific heat mitigation measures "
                "for this zone, referencing relevant EU frameworks and evidence where appropriate."
            ),
        },
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

    generated = output_ids[0][inputs.input_ids.shape[1] :]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    del model, tokenizer, inputs
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    import random
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from heatmit.suggest import _s2_to_rgb, _super_resolve, describe_scene

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
        return out  # (bands, H, W)

    # --- 1. Load and reproject S2 ---
    print("Reprojecting resources/sent2.tif to EPSG:32632 at 10 m/px...")
    s2_data = _reproject_tif(S2_TIF)
    _, height, width = s2_data.shape

    # --- 2. Random 56×56 crop ---
    col = random.randint(0, width - CROP)
    row = random.randint(0, height - CROP)
    s2_patch = np.transpose(s2_data[:, row:row + CROP, col:col + CROP], (1, 2, 0))  # (H, W, 4)

    OUT.mkdir(exist_ok=True)

    # --- 3. Save original PNG ---
    s2_png = OUT / f"{row}_{col}_s2.png"
    _s2_to_rgb(s2_patch).save(s2_png)
    print(f"Saved: {s2_png}")

    # --- 4. Super-resolve S2 (16×) and save PNGs ---
    print("Super-resolving S2 patch (16×)...")
    s2_x4, s2_x16 = _super_resolve(s2_patch)

    s2_x4_png = OUT / f"{row}_{col}_s2_x4.png"
    _s2_to_rgb(s2_x4).save(s2_x4_png)
    print(f"Saved: {s2_x4_png}")

    s2_x16_png = OUT / f"{row}_{col}_s2_x16.png"
    _s2_to_rgb(s2_x16).save(s2_x16_png)
    print(f"Saved: {s2_x16_png}")

    # --- 5. Describe scene ---
    print("Describing scene with Qwen2.5-VL-3B...")
    description = describe_scene(s2_patch, s2_x4=s2_x4, s2_x16=s2_x16)
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
