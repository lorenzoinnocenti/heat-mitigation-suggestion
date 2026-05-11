import numpy as np
import torch
import requests
import pdfplumber
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from heatmit.super_resolution import _get_device

EMBED_MODEL_ID = "all-MiniLM-L6-v2"
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = Path("./data")

WEB_SOURCES = [
    "https://climate-adapt.eea.europa.eu/en/eu-adaptation-policy/sector-policies/urban/index_html",
]

PDF_SOURCES = [
    # EU sources
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
    {
        "url": "https://eu-mayors.ec.europa.eu/sites/default/files/2022-10/eumayors-Adapting-To-%20Climate-Change-2022.pdf",
        "filename": "covenant_of_mayors_adapting_to_climate_change_2022.pdf",
    },
    # UNEP
    {
        "url": "https://reliefweb.int/attachments/3beb63d7-e232-36f9-8305-ef2e06641e4d/Beating%20the%20heat%20-%20a%20sustainable%20cooling%20handbook%20for%20cities%20%28full%20report%29.pdf",
        "filename": "unep_beating_the_heat_cooling_handbook_2021.pdf",
    },
    # UN-Habitat
    {
        "url": "https://unhabitat.org/sites/default/files/2022/06/wcr_2022.pdf",
        "filename": "unhabitat_world_cities_report_2022.pdf",
    },
    # WHO
    {
        "url": "https://iris.who.int/bitstream/handle/10665/345751/WHO-EURO-2016-3352-43111-60341-eng.pdf?sequence=1&isAllowed=y",
        "filename": "who_urban_green_spaces_health_2016.pdf",
    },
    # IPCC
    {
        "url": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter06.pdf",
        "filename": "ipcc_ar6_wg2_chapter06_cities_2022.pdf",
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
    cache = path.with_suffix(".txt")
    if cache.exists():
        return cache.read_text()
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    result = "\n\n".join(texts)
    cache.write_text(result)
    return result


def _chunk(text: str, max_words: int = 400) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        words = len(para.split())
        if current_len + words > max_words and current:
            chunks.append("\n\n".join(current))
            # carry the last paragraph into the next chunk as overlap
            current = [current[-1]]
            current_len = len(current[0].split())
        current.append(para)
        current_len += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


class RAG:
    def __init__(self, top_k: int = 10):
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
        # embed the query string into a float vector using the same model used to embed the chunks
        q = self.embedder.encode([query], convert_to_numpy=True)
        # L2-normalise so that the dot product below equals cosine similarity (range −1 to 1)
        q = q / (np.linalg.norm(q, keepdims=True) + 1e-8)
        # dot product of every stored chunk embedding against the query vector -> similarity score per chunk
        scores = (self.embeddings @ q.T).squeeze()
        # argsort gives ascending order; [::-1] reverses to descending, then take the top_k indices
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        # return the raw text chunks corresponding to the highest-scoring indices
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
    # top-5 text chunks retrieved from the EU literature corpus

    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # system — sets the model's persona and task. It's processed before anything else and tells the model "you are an expert, here is your job." The model never "responds" to this, it just conditions on it.
    # user — the actual input the model responds to. This is where the retrieved context and the scene description go, because those are dynamic per-call.
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

    # Format the messages list into the model's expected chat string (e.g. "<|system|>...<|user|>...")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Convert the string to token IDs and move them to GPU/CPU
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Run the model: generate up to 1500 new tokens, do_sample=False: no randomness
        output_ids = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

    # Strip the input tokens from the output — keep only the newly generated tokens
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    # Convert token IDs back to a human-readable string, removing special tokens like <|endoftext|>
    result = tokenizer.decode(generated, skip_special_tokens=True)

    del model, tokenizer, inputs
    torch.cuda.empty_cache()
    return result
