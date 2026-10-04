# MNIST diffusion demo and solution

Open **[diffusion_mnist.ipynb](diffusion_mnist.ipynb)** and **Run All**. The notebook
is saved with executed outputs and defaults to `RUN_TRAINING = False`. It loads
the trained model automatically, shows real digits and the noise schedule, then
runs DDPM, the denoising trajectory, and optional DDIM sampling.

The demo works offline: the checkpoint includes eight MNIST examples. A GPU is
recommended for the 1,000-step DDPM loop; the saved plots can be presented without
rerunning it. DDIM uses 50 steps. Samples are not perfect, which is useful for discussion.

## Setup

This notebook and its checkpoint run without the workshop folder. From the
repository root, choose either [venv + pip](../requirements-venv.txt) or the
root [Conda environment](../environment-conda.yml):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-venv.txt
python -m jupyterlab mnist-implementation/diffusion_mnist.ipynb
```

Or use Conda:

```bash
conda env create -f environment-conda.yml
conda activate mnist-diffusion
python -m jupyterlab mnist-implementation/diffusion_mnist.ipynb
```

Select the Python kernel from the activated environment. The venv path also
works with another installed Python 3.10+ executable. Both `ipykernel` and
`jupyterlab` are included in the root environment files; starting JupyterLab
from that environment is sufficient for this setup.

## Files

- `diffusion_mnist.ipynb`: completed lesson, training and sampling loops, saved plots.
- `diffusion.py`: cosine schedule and the forward/reverse equations.
- `unet.py`: explicit 28 → 14 → 7 → 14 → 28 U-Net with timestep conditioning.
- `checkpoints/mnist_demo.pt`: the only retained demo checkpoint (about 3 MB).

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

You can distribute this folder with the two root environment files. Include
`checkpoints/mnist_demo.pt`; the notebook needs no workshop files to show the
saved figures or regenerate samples.
Cached `data/` is optional for the demo, but useful if the group wants to train
without downloading MNIST.
