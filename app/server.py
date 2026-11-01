"""Web backend for the photo editor: upload + instruction + guidance sliders -> edited PNG.

Runs the model on whatever machine this process is on (your GPU, a Colab T4, a cluster
node, a Mac). No hosted inference API involved.

From the repo root:
    pip install fastapi "uvicorn[standard]" python-multipart

    # Reference diffusers pipeline: works today, before the pipeline.py TODOs are done
    python -m app.server --backend diffusers

    # Your own sampler from instructpix2pix/pipeline.py (once its checks pass)
    python -m app.server --backend ours

    # Your own trained UNet
    python -m app.server --backend ours --unet_path checkpoints/ip2p/final/unet

Then open http://localhost:8000
"""
import argparse
import io
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from instructpix2pix.common import IP2P, get_device

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_SIDE = 512            # the model was trained at 256-512px; bigger is slower and not better
MAX_PROMPT_CHARS = 300

app = FastAPI(title="InstructPix2Pix editor")
GPU_LOCK = threading.Lock()   # one GPU, one edit at a time; other requests wait their turn
RUN = None                    # set in main(): (image, prompt, steps, s_T, s_I, seed) -> PIL image
INFO = {}


class BlockedOutput(Exception):
    pass


def build_runner(backend: str, model_id: str, unet_path: str | None, device: torch.device, dtype: torch.dtype):
    if backend == "ours":
        from instructpix2pix.pipeline import InstructPix2PixPipeline

        pipe = InstructPix2PixPipeline.from_pretrained(model_id, unet_path, device=device, dtype=dtype)

        def run(image, prompt, steps, s_t, s_i, seed):
            return pipe(image, prompt, steps, s_t, s_i, seed)[0]

        return run

    from diffusers import (EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline,
                           UNet2DConditionModel)

    extra = {}
    if unet_path:
        extra["unet"] = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=dtype, **extra).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    def run(image, prompt, steps, s_t, s_i, seed):
        out = pipe(prompt, image=image, num_inference_steps=steps, guidance_scale=s_t,
                   image_guidance_scale=s_i, generator=torch.Generator("cpu").manual_seed(seed))
        if getattr(out, "nsfw_content_detected", None) and any(out.nsfw_content_detected):
            raise BlockedOutput
        return out.images[0]

    return run


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
    return img


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return INFO


@app.post("/api/edit")
def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    guidance_scale: float = Form(7.5),          # s_T: how strongly to follow the instruction
    image_guidance_scale: float = Form(1.5),    # s_I: how strongly to stay close to the photo
    steps: int = Form(20),
    seed: int = Form(0),
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Write an instruction first, like “add a pink bow to my hair”.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Keep the instruction under {MAX_PROMPT_CHARS} characters.")

    img = read_upload(image.file.read())
    s_t = min(max(guidance_scale, 1.0), 20.0)
    s_i = min(max(image_guidance_scale, 1.0), 5.0)
    steps = min(max(steps, 5), 60)
    seed = abs(seed) % 2**31

    with GPU_LOCK:   # sync endpoint -> FastAPI runs it in a worker thread, so waiting here is fine
        t0 = time.perf_counter()
        try:
            out = RUN(img, prompt, steps, s_t, s_i, seed)
        except BlockedOutput:
            raise HTTPException(422, "The safety filter blocked this result. Try a different instruction.")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(503, "The GPU ran out of memory. Try again, or lower MAX_SIDE on the server.")
        elapsed = time.perf_counter() - t0

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png",
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
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"loading {a.backend} pipeline on {device} ({dtype}) ...")
    RUN = build_runner(a.backend, a.model_id, a.unet_path, device, dtype)
    INFO = {"backend": a.backend, "device": device.type, "unet": a.unet_path or a.model_id}
    if device.type == "cpu":
        print("warning: running on CPU, so each edit will take minutes, not seconds")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()