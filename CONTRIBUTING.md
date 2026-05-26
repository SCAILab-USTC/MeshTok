# Contributing

Thanks for your interest in MeshTok.

## Development Setup

Create an environment from `environment.yaml` for the full paper environment, or install the lighter package list:

```bash
pip install -r requirements.txt
```

Before opening a pull request, run the training or evaluation command that matches the part of the code you changed and include the command in the pull request.

## Style

- Keep configuration changes in `src/configs/`.
- Keep generated checkpoints, datasets, and `wandb/` logs out of git.
- Prefer small, reproducible command examples in issues and pull requests.

## Data and Checkpoints

Do not commit benchmark datasets or trained checkpoints. Use paths under `pde_data/` locally and document any new expected HDF5 layout in the README.
