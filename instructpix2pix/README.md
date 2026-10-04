# Phase 2: InstructPix2Pix from Stable Diffusion 1.5

A minimal, readable re-implementation of
[`StableDiffusionInstructPix2PixPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py)
and [`train_instruct_pix2pix.py`](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py).
We use `diffusers` only for the pretrained building blocks (VAE, UNet, schedulers). The surgery,
data pipeline, training step and guided sampler are ours.

```
common.py      given: loading SD 1.5, tokenization, tiny random models for testing
model.py       TODO: conv_in surgery, text/image encoders        (4 TODOs)
pipeline.py    TODO: dual classifier-free guidance sampler       (4 TODOs)
data.py        TODO: paired augmentation                         (1 TODO)
train.py       TODO: conditioning dropout, training step         (2 TODOs)
solutions/     worked answers (try each TODO for 20 min first)
```

## Setup

```bash
pip install -r instructpix2pix/requirements.txt
```

Run everything from the repo root as a module (`python -m instructpix2pix.<file>`).

## Order of work

Every file has sanity checks that run on **tiny random models on CPU in seconds**, so
you don't need a GPU or the 4 GB of SD weights until the end. A failing check tells you
which function is wrong.

1. **`model.py`**: `python -m instructpix2pix.model`
2. **`pipeline.py`**: `python -m instructpix2pix.pipeline --check`, then the real test with the
   authors' weights, which proves your sampler is correct before you train anything:
   ```bash
   python -m instructpix2pix.pipeline --image photo.jpg --prompt "make it snowy"
   ```
3. **`data.py`**: `python -m instructpix2pix.data`
4. **`train.py`**: `python -m instructpix2pix.train --check`, then overfit ~64 examples
   (loss should drop well below its starting value):
   ```bash
   python -m instructpix2pix.train --max_samples 64 --max_steps 500 --output_dir checkpoints/overfit
   python -m instructpix2pix.pipeline --image photo.jpg --prompt "..." --unet_path checkpoints/overfit/final/unet
   ```
5. Full run (paper settings: 256px, effective batch 16, lr 5e-5, ~15k steps, needs a 24 GB+ GPU):
   ```bash
   python -m instructpix2pix.train --dataset full --streaming
   ```

## The whole model in five facts

1. It is SD 1.5 with `unet.conv_in` widened from 4 to 8 input channels (zero-initialised).
   The extra 4 channels carry the VAE latent of the **original** image.
2. The instruction enters through cross-attention, exactly like a normal SD prompt.
3. Only the UNet trains. The loss is the usual `MSE(predicted noise, true noise)` on the **edited** image's latent.
4. About 5% of the time each condition is dropped (text → `""`, image → zeros), and 5% both,
   so the model also learns the unconditional predictions.
5. Sampling runs the UNet on 3 copies `[text+image, image, nothing]` and mixes them:
   `eps = eps_∅ + s_I (eps_img − eps_∅) + s_T (eps_text − eps_img)` with `s_T=7.5`, `s_I=1.5`.

**Gotcha:** the target latent is `.sample() * 0.18215`; the conditioning latent is `.mode()`
with **no** scaling. The released weights expect exactly this.
