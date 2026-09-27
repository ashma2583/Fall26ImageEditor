# MNIST diffusion demo and solution

Open **[diffusion_mnist.ipynb](diffusion_mnist.ipynb)** and **Run All**. The notebook
is saved with executed outputs and defaults to `RUN_TRAINING = False`. It loads
the trained model automatically, shows real digits and the noise schedule, then
runs DDPM, the denoising trajectory, and optional DDIM sampling.

The demo works offline: the checkpoint includes eight MNIST examples. A GPU is
recommended for the 1,000-step DDPM loop; the saved plots can be presented without
rerunning it. DDIM uses 50 steps. Samples are not perfect, which is useful for discussion.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r mnist-implementation/requirements.txt
python -m ipykernel install --user --name mnist-diffusion --display-name "MNIST diffusion"
python -m jupyterlab mnist-implementation/diffusion_mnist.ipynb
```

Select the **MNIST diffusion** kernel. Launch from the repository root or this folder.

## Files

- `diffusion_mnist.ipynb`: completed lesson, training and sampling loops, saved plots.
- `diffusion.py`: cosine schedule and the forward/reverse equations.
- `unet.py`: explicit 28 → 14 → 7 → 14 → 28 U-Net with timestep conditioning.
- `checkpoints/mnist_demo.pt`: the only retained demo checkpoint (about 3 MB).
- `../mnist-workshop/`: the separate incomplete participant version.

The checkpoint contains CPU model weights, configuration, cosine loss history,
training provenance, and example images. Optimizer state and obsolete experiments
have been removed. The retained model records 25 cosine training epochs; the plotted loss history
covers those epochs. No EMA
is used. Code supports the final architecture and cosine schedule only.

## Optional training

Set `RUN_TRAINING = True`, restart the kernel, and Run All to train a new model.
Defaults are one epoch on 2,048 examples for a short learning exercise; expect rough
samples. For a longer experiment, set `TRAIN_SUBSET = None` and `TRAIN_EPOCHS = 25`.
Training needs MNIST (downloaded on first use) and saves `checkpoints/training.pt`.
It never overwrites the demo checkpoint. Return to `RUN_TRAINING = False` and
restart/run all to restore the prepared demo.

## Sharing for the meeting

Keep `mnist-implementation` and `mnist-workshop` beside each other. Include
`checkpoints/mnist_demo.pt` when distributing the folders. Cached `data/` is optional
for the demo, but useful if the group wants to train without downloading MNIST.
