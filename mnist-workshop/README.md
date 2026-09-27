# MNIST diffusion workshop

Open **[diffusion_mnist_workshop.ipynb](diffusion_mnist_workshop.ipynb)**. Complete the
numbered TODOs in the notebook and the two exercise modules. The completed reference
and instructor demo are in `../mnist-implementation/`.

## Before the meeting

Keep both folders beside each other and install the shared environment:

```bash
python -m pip install -r mnist-workshop/requirements.txt
python -m jupyterlab mnist-workshop/diffusion_mnist_workshop.ipynb
```

Select your Python environment as the kernel. Launch from this folder or the
repository root. The workshop loads only `diffusion_exercises.py` and
`unet_exercises.py`; it never imports the completed implementation. It shares the
trained checkpoint and example images from the demo folder, so those files do not
need to be duplicated. **Restart the kernel after editing a module**, then rerun.

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
`checkpoints/training.pt`, leaving the instructor checkpoint untouched.

To return to the trained model, set `RUN_TRAINING = False` and restart/run all.
Do not remove or rename model layers: the shared checkpoint expects the supplied
architecture. Implement the missing operations inside those layers.
