import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # HF classes for the VLM model and its tokenizer+image processor
from qwen_vl_utils import process_vision_info  # Qwen helper that extracts image/video tensors from a message dict

from heatmit.super_resolution import _get_device, s2_to_rgb  # reuse device detection and S2→RGB conversion

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # HuggingFace repo ID for the 7B vision-language model


def describe_scene(
    s2_patch: np.ndarray,
    s2_x4: np.ndarray | None = None,
    s2_x16: np.ndarray | None = None,
    prompt: str | None = None,
) -> str:
    """
    Generates a natural language description using Qwen2.5-VL-7B.
    Accepts up to three resolutions of the same patch (original, 4x, 16x SR).

    Args:
        s2_patch: original S2 crop, shape (H, W, 4) — bands [B02, B03, B04, B08]
        s2_x4:   4x SR patch (H*4, W*4, 4), optional
        s2_x16:  16x SR patch (H*16, W*16, 4), optional
        prompt:  Instruction sent to the model. Built automatically if None.

    Returns:
        str: Natural language description of the scene
    """
    n_images = 1 + (s2_x4 is not None) + (s2_x16 is not None)
    if prompt is None:
        if n_images == 3:
            prompt = (
                "You are analyzing Sentinel-2 satellite imagery of the Turin metropolitan area "
                "(Po Plain, northern Italy), covering a 2.56 km x 2.56 km tile. "
                "Three co-registered images are provided at increasing resolution: "
                "Image 1: original at 20 m/px (128 x 128 px). "
                "Image 2: 4x super-resolved at 5 m/px (512 x 512 px). "
                "Image 3: 16x super-resolved at 1.25 m/px (2048 x 2048 px). "
                "All are true-color RGB (B04/B03/B02, ESA Copernicus L2A). "
                "Using all three images, provide: "
                "1. A land cover breakdown with estimated area percentages (e.g. 'Impervious surfaces: 45%'). "
                "2. Urban morphology: building types (residential/commercial/industrial), density, roof types (flat/pitched/green). "
                "3. Vegetation: tree canopy, grass, agriculture — type and distribution. "
                "4. Infrastructure: roads, rail, water bodies. "
                "5. UHI relevance: which features increase or reduce local heat load. "
                "Be specific — avoid qualitative-only statements like 'mostly urban'."
            )
        else:
            raise NotImplementedError("Prompt generation for 1 or 2 images is not implemented yet. Please provide a custom prompt.")

    device = _get_device()  
    print(f"VLM running on: {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  
        MODEL_ID,
        torch_dtype=torch.float16,  
        device_map=device,  
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID) 

    def _to_image(patch: np.ndarray) -> Image.Image:
        return s2_to_rgb(patch)  # convert (H,W,4) S2 DN array to an 8-bit RGB PIL image

    content = [{"type": "image", "image": _to_image(s2_patch)}]
    if s2_x4 is not None:
        content.append({"type": "image", "image": _to_image(s2_x4)})
    if s2_x16 is not None:
        content.append({"type": "image", "image": _to_image(s2_x16)})
    content.append({"type": "text", "text": prompt})  # append the text instruction as the final content item

    messages = [{"role": "user", "content": content}]  # wrap in a single-turn chat format expected by Qwen

    # Two-phase encoding: apply_chat_template handles text only: it produces a string with special tokens
    # and <image> placeholders but never touches the pixel data. process_vision_info then extracts the
    # raw PIL images from the same messages dict and normalises them into tensors. The final processor()
    # call stitches both outputs together, aligning image tensors with the <image> placeholders in the
    # text. The roundtrip through messages is necessary because the chat template and image preprocessor
    # are separate tools with no shared state; messages is the common intermediate format both read from.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(device)  # move all tensors to GPU/CPU

    with torch.no_grad():
        # generated_ids: (batch, prompt_len + reply_len) integer tensor; each value is a vocabulary
        # index (e.g. 15234), not a float. Floats exist only inside the model (embeddings, attention,
        # logits); at each step the final logit vector is collapsed to a single integer via argmax/
        # sampling, and that integer is appended to the sequence. model.generate returns the full
        # sequence (prompt tokens + new tokens) concatenated, not just the new tokens.
        generated_ids = model.generate(**inputs, max_new_tokens=2000)

    # generated_ids_trimmed: strip the prompt prefix from each sequence so only the newly generated
    # token IDs remain. Without this, batch_decode would include the prompt text in the output string.
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    # decode token IDs back to a string, removing special tokens
    result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]  
    
    del model, processor, inputs 
    torch.cuda.empty_cache()
    
    return result
