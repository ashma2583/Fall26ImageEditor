# MNIST diffusion workshop

Open **[diffusion_mnist_workshop.ipynb](diffusion_mnist_workshop.ipynb)**. Complete the
numbered TODOs in the notebook and the two exercise modules. It has its own
pretrained checkpoint and example images. The notebook does not use
`mnist-implementation`; both folders use the environment files at the repository
root.

## Before the meeting

From the repository root, choose **one** environment setup. Python 3.11 is a
good choice for the venv path; Conda selects it automatically.

### Python venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-venv.txt
python -m jupyterlab mnist-workshop/diffusion_mnist_workshop.ipynb
```

If `python3.11` is unavailable, use another installed Python 3.10+ executable.
On Windows Command Prompt, activate with `.venv\Scripts\activate.bat` instead of
`source`.

### Conda

```bash
conda env create -f environment-conda.yml
conda activate mnist-diffusion
python -m jupyterlab mnist-workshop/diffusion_mnist_workshop.ipynb
```

The Conda package is named `pytorch`; the pip package is named `torch`. Both
provide the Python import `torch`. Select the Python kernel from the activated
environment in Jupyter. Run through the real-digit plot before the meeting to load the supplied
checkpoint and confirm the setup. The next cell stops at TODO 1 until it is
implemented. `ipykernel` and `jupyterlab` are already included in both root
environment files, so no separate package install or kernel registration is
needed. The [sample slide plan](slideshow-plan.md) includes presenter cues.

The notebook works when launched from its folder or the repository root. The
workshop loads only `diffusion_exercises.py` and `unet_exercises.py`. Its own
`checkpoints/mnist_workshop.pt` supplies the trained weights and eight example
images; optional training data downloads into this folder's `data/`. **Restart
the kernel after editing a module**, then rerun.

An unfinished exercise raises a numbered `NotImplementedError`. That is intentional.
The student notebook has no saved solutions or outputs. Use the plots to reason
about your implementation rather than trying to pass assertion cells.

## Suggested meeting flow (about 90 minutes)

| Time | Activity | Where |
|---|---|---|
| 10 min | Instructor demo: noise → digits | Completed notebook |
| 20 min | TODO 1: alpha and cumulative alpha; TODO 2: broadcasting; TODO 3: forward noising | `diffusion_exercises.py` |
| 20 min | TODO 4: timestep embedding; TODO 5: time conditioning; TODO 6a/6b: skip connections | `unet_exercises.py` |
| 15 min | TODO 7: training step; discuss the MSE target | Notebook |
| 20 min | TODO 8: clean estimate; TODO 9: reverse step; TODO 10: sampling loop | Module + notebook |
| 5 min | Compare DDPM and the provided DDIM sampler | Notebook |

Cosine beta construction, posterior coefficients, attention, layer definitions,
plotting, and data loading are supplied. Each TODO lists input/output shapes or
formula hints. For a shorter session, walk through TODOs 4–6 as a group and focus
independent work on the forward and reverse diffusion equations.

Keep `RUN_TRAINING = False` while implementing the core operations: the supplied
weights let you generate digits as soon as the forward passes and sampler work.
After completing TODO 7, optionally set `RUN_TRAINING = True`, `TRAIN_SUBSET = 256`,
and `TRAIN_EPOCHS = 1`, then restart/run all. This exercises the learning loop; it
will not produce a well-trained generator. Training outputs stay in this folder's
`checkpoints/training.pt`, leaving the prepared workshop checkpoint untouched.

To return to the trained model, set `RUN_TRAINING = False` and restart/run all.
Do not remove or rename model layers: the local checkpoint expects the supplied
architecture. Implement the missing operations inside those layers.
