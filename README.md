# MeshTok

MeshTok is a PyTorch implementation for autoregressive forecasting on 2D structured-grid PDE trajectories. The code uses Hydra configuration files, HDF5 data loaders, and a multi-scale tokenization model that refines selected spatial patches before transformer prediction.

The repository is organized around the current training and evaluation code:

- `src/main.py`: Hydra entry point for training and evaluation.
- `src/configs/`: default data, model, and optimizer configurations.
- `src/data_utils/`: HDF5 dataset readers and batch collation.
- `src/models/`: MeshTok model, embedder, transformer blocks, and KV cache.
- `src/trainer.py`: optimizer, scheduler, checkpointing, and training loop.
- `src/evaluate.py`: validation and test-time rollout/evaluation logic.
- `DATA.md`: expected local data layout and HDF5 tensor contracts.

## Environment

Create the conda environment from the trimmed runtime dependency file:

```bash
conda env create -f environment.yaml
conda activate meshtok
```

The environment file keeps only the direct packages used by the current code. If you manage CUDA or PyTorch separately, install the Python dependencies from `requirements.txt` after your PyTorch installation.

## Data

The data used by this project is prepared from PDEBench, PDENNeval, and The Well. The datasets are not included in this repository. Place the converted HDF5 files under `pde_data/` using the layout described in `DATA.md`.

The default config is `src/configs/data/easy_test.yaml` and expects the following PDE families:

- `gray_scott`
- `allen_cahn`
- `cfddata`
- `com_ns`
- `shallow_water`

Single-file datasets are read from HDF5 tensors shaped as:

```text
(N, T, H, W, C)
```

Each sample yielded to the model has shape:

```text
(T, H, W, C)
```

Folder-based datasets read every `.hdf5` or `.h5` file in the configured split folder.

## Training

Run commands from the repository root so that `pde_data/` and `checkpoint/` resolve to the expected local folders:

```bash
python src/main.py
```

The default command uses:

- data config: `easy_test`
- model config: `meshtok`
- optimizer config: `adamw`
- output root: `checkpoint/meshtok/<exp_id>/`

Useful Hydra overrides:

```bash
python src/main.py use_wandb=0
python src/main.py batch_size=4 max_epoch=10 n_steps_per_epoch=1000
python src/main.py data.types='[gray_scott,allen_cahn]'
python src/main.py dump_path=checkpoint exp_id=my_run
```

## Evaluation

Evaluate a saved experiment by pointing `eval_from_exp` to an experiment directory or checkpoint:

```bash
python src/main.py eval_only=1 eval_from_exp=checkpoint/meshtok/<exp_id> use_wandb=0
```

The evaluation code first looks for `best-<validation_metric>.pth` in the experiment directory, then falls back to `checkpoint.pth`. It reports aggregate metrics and per-equation metrics through the logger.

Additional evaluation options:

```bash
python src/main.py eval_only=1 eval_from_exp=checkpoint/meshtok/<exp_id> rollout=1
python src/main.py eval_only=1 eval_from_exp=checkpoint/meshtok/<exp_id> save_outputs=1
python src/main.py eval_only=1 eval_from_exp=checkpoint/meshtok/<exp_id> print_outputs=1
```

## Configuration

Hydra configs are stored under `src/configs/`.

- `main.yaml` defines training length, batch sizes, logging, normalization, augmentation, checkpointing, and evaluation options.
- `data/easy_test.yaml` defines the default dataset list, local data paths, temporal stride, spatial resolution, channel masks, and sampler weights.
- `model/meshtok.yaml` defines MeshTok architecture parameters, patch sizes, refinement ratio, embedder settings, transformer depth, attention options, and KV cache usage.
- `optim/*.yaml` defines optimizer and scheduler settings for AdamW, Muon, and warmup-stable-decay schedules.

Hydra overrides can be passed directly on the command line. For example:

```bash
python src/main.py model.refine_ratio=0.25 model.topk=4 optim.lr=1e-4
```

## Outputs

Training creates an experiment directory under `checkpoint/` by default. The code writes:

- `configs.yaml`: resolved runtime configuration.
- `train.log`: training and evaluation logs.
- `checkpoint.pth`: latest checkpoint.
- `best-<metric>.pth`: best checkpoint for the configured validation metric.
- `evals_all/`: optional evaluation figures and saved outputs.

Datasets, checkpoints, HDF5 files, NumPy output archives, and WandB logs are ignored by git.

## Citation

If you use this code, cite the paper metadata in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.
