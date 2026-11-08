"""Web backend for the photo editor: upload + instruction + guidance sliders -> edited PNG.

Runs the model on whatever machine this process is on (your GPU, a Colab T4, a cluster
node, a Mac). No hosted inference API involved.

From the repo root (Fall26ImageEditor):
    pip install -r app/requirements.txt

    python -m app.server --backend diffusers                       # reference pipeline
    python -m app.server --backend diffusers --no_safety_checker   # local testing only
    python -m app.server --backend ours                            # your instructpix2pix/pipeline.py

Then open http://127.0.0.1:8000

Area editing: when the browser sends a mask (white = may change), the diffusers backend pins
every pixel outside the mask to the original photo at every denoising step ("latent blending"),
so the model draws the edit to fit its surroundings and nothing else moves. Afterwards the
original pixels are pasted back outside the (feathered) mask, so untouched areas stay exact.
The `ours` backend gets the paste-back only, until pipeline.py supports a mask itself.
"""
import argparse
import io
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from instructpix2pix.common import IP2P, get_device

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_SIDE = 512            # the model was trained at 256-512px; bigger is slower and not better
MAX_PROMPT_CHARS = 300
SEG_ID = "CIDAS/clipseg-rd64-refined"   # text -> mask model for "auto-select" (~600 MB, CPU is fine)

app = FastAPI(title="InstructPix2Pix editor")
GPU_LOCK = threading.Lock()   # one GPU, one edit at a time; other requests wait their turn
SEG_LOCK = threading.Lock()
RUN = None                    # set in main(): (image, prompt, steps, s_T, s_I, seed, mask) -> PIL image
SEG = None                    # CLIPSeg (processor, model), loaded on first auto-select
INFO = {}


class BlockedOutput(Exception):
    pass


# ---------------------------------------------------------------------------- model

def build_runner(backend: str, model_id: str, unet_path: str | None, device: torch.device,
                 dtype: torch.dtype, safety: bool = True):
    if backend == "ours":
        from instructpix2pix.pipeline import InstructPix2PixPipeline

        pipe = InstructPix2PixPipeline.from_pretrained(model_id, unet_path, device=device, dtype=dtype)

        def run(image, prompt, steps, s_t, s_i, seed, mask=None):
            return pipe(image, prompt, steps, s_t, s_i, seed)[0]   # mask handled by paste-back

        return run

    from diffusers import (EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline,
                           UNet2DConditionModel)

    extra = {}
    if unet_path:
        extra["unet"] = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
    if not safety:
        extra["safety_checker"] = None
        extra["requires_safety_checker"] = False
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=dtype, **extra).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    def latent_blend(image, mask, seed):
        """Keep latents outside the mask equal to the original photo, noised to the current step."""
        with torch.no_grad():
            px = pipe.image_processor.preprocess(image).to(device, dtype)
            z0 = pipe.vae.encode(px).latent_dist.mode() * pipe.vae.config.scaling_factor
        h, w = z0.shape[-2:]
        m = torch.from_numpy(np.asarray(mask.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0)
        m = F.max_pool2d(m[None, None], 3, 1, 1).to(device, dtype)        # grow 1 latent px of room
        noise = torch.randn(z0.shape, generator=torch.Generator("cpu").manual_seed(seed + 1)).to(device, dtype)

        def on_step_end(p, i, t, tensors):
            lat = tensors["latents"]
            sigma = p.scheduler.sigmas[p.scheduler.step_index].to(lat.device, lat.dtype)   # next noise level
            return {"latents": m * lat + (1 - m) * (z0 + sigma * noise)}

        return {"callback_on_step_end": on_step_end, "callback_on_step_end_tensor_inputs": ["latents"]}

    def run(image, prompt, steps, s_t, s_i, seed, mask=None):
        kw = latent_blend(image, mask, seed) if mask is not None else {}
        out = pipe(prompt, image=image, num_inference_steps=steps, guidance_scale=s_t,
                   image_guidance_scale=s_i, generator=torch.Generator("cpu").manual_seed(seed), **kw)
        if getattr(out, "nsfw_content_detected", None) and any(out.nsfw_content_detected):
            raise BlockedOutput
        return out.images[0]

    return run


def segment(img: Image.Image, phrases: list[str]) -> Image.Image:
    """Text -> binary mask with CLIPSeg; several phrases are unioned ("hair, background")."""
    global SEG
    with SEG_LOCK:
        if SEG is None:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
            SEG = (CLIPSegProcessor.from_pretrained(SEG_ID), CLIPSegForImageSegmentation.from_pretrained(SEG_ID).eval())
        proc, model = SEG
        inputs = proc(text=phrases, images=[img] * len(phrases), padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
    prob = torch.sigmoid(logits).reshape(len(phrases), *logits.shape[-2:]).amax(0)
    thr = min(0.35, 0.5 * prob.max().item())          # CLIPSeg is often under-confident on small things
    m = Image.fromarray(((prob > thr).numpy() * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    grow = max(3, round(min(img.size) * 0.03)) | 1     # odd size; a little extra room around the region
    return m.point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(grow))


# ---------------------------------------------------------------------------- images

def read_upload(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)   # phone selfies are often stored sideways + an EXIF flag
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't an image this server can read. Try a JPG or PNG.")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    if min(img.size) < 64:
        raise HTTPException(400, "Image is too small. Use one at least 64 px on each side.")
    w, h = img.size
    return img.resize((w - w % 8, h - h % 8), Image.LANCZOS) if (w % 8 or h % 8) else img   # VAE needs /8


def read_mask(data: bytes, size: tuple[int, int]) -> Image.Image:
    try:
        m = Image.open(io.BytesIO(data)).convert("L").resize(size, Image.BILINEAR)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The selection couldn't be read. Clear it and paint again.")
    m = m.point(lambda v: 255 if v > 127 else 0)
    if m.getbbox() is None:
        raise HTTPException(400, "Nothing is selected. Paint over the part you want to change, or use Auto-select.")
    return m


def paste_back(original: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    """Edited pixels inside the mask, original pixels outside, with a soft seam."""
    edited = edited.resize(original.size, Image.LANCZOS)
    r = max(2, round(min(original.size) * 0.015))
    soft = mask.filter(ImageFilter.MaxFilter(2 * r + 1)).filter(ImageFilter.GaussianBlur(r))
    return Image.composite(edited, original, soft)


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------- routes

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return INFO


@app.post("/api/mask")
def auto_mask(image: UploadFile = File(...), region: str = Form(...)):
    phrases = [p.strip() for p in region.split(",") if p.strip()][:4]
    if not phrases:
        raise HTTPException(400, "Say which part to select, like “hair” or “shirt, background”.")
    img = read_upload(image.file.read())
    mask = segment(img, phrases)
    if mask.getbbox() is None:
        raise HTTPException(422, f"Couldn't find “{region.strip()}” in the photo. Try other words, or paint it.")
    return Response(png(mask), media_type="image/png")


@app.post("/api/edit")
def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    guidance_scale: float = Form(7.5),          # s_T: how strongly to follow the instruction
    image_guidance_scale: float = Form(1.5),    # s_I: how strongly to stay close to the photo
    steps: int = Form(20),
    seed: int = Form(0),
    mask: UploadFile | None = File(None),       # optional: white = allowed to change
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Write an instruction first, like “add a pink bow in the hair”.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Keep the instruction under {MAX_PROMPT_CHARS} characters.")

    img = read_upload(image.file.read())
    m = read_mask(mask.file.read(), img.size) if mask is not None else None
    s_t = min(max(guidance_scale, 1.0), 20.0)
    s_i = min(max(image_guidance_scale, 1.0), 5.0)
    steps = min(max(steps, 5), 60)
    seed = abs(seed) % (2**31 - 1)

    with GPU_LOCK:   # sync endpoint -> FastAPI runs it in a worker thread, so waiting here is fine
        t0 = time.perf_counter()
        try:
            out = RUN(img, prompt, steps, s_t, s_i, seed, m)
        except BlockedOutput:
            raise HTTPException(422, "The safety filter blocked this result. Try a different instruction or variation number.")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(503, "The GPU ran out of memory. Try again, or lower MAX_SIDE on the server.")
        elapsed = time.perf_counter() - t0

    if m is not None:
        out = paste_back(img, out, m)
    return Response(png(out), media_type="image/png",
                    headers={"X-Seed": str(seed), "X-Elapsed": f"{elapsed:.1f}"})


@app.exception_handler(Exception)
def unhandled(_, exc):
    return JSONResponse({"detail": f"Server error: {type(exc).__name__}. Check the server log."}, status_code=500)


def main():
    global RUN, INFO
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ours", "diffusers"], default="diffusers")
    p.add_argument("--model_id", default=IP2P)
    p.add_argument("--unet_path", default=None)
    p.add_argument("--no_safety_checker", action="store_true", help="local development only")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"loading {a.backend} pipeline on {device} ({dtype}) ...")
    RUN = build_runner(a.backend, a.model_id, a.unet_path, device, dtype, safety=not a.no_safety_checker)
    INFO = {"backend": a.backend, "device": device.type, "unet": a.unet_path or a.model_id}
    if device.type == "cpu":
        print("warning: running on CPU, so each edit will take minutes, not seconds")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()