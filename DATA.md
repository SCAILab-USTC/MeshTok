# Data

MeshTok expects structured-grid PDE trajectories in HDF5 files. The data used by this project is prepared from PDEBench, PDENNeval, and The Well. The repository does not include benchmark files; place the converted local datasets under `pde_data/`.

The training code reads samples as tensors with shape:

```text
(T, H, W, C)
```

The collate function pads channel dimensions to `data.max_output_dimension` when necessary, so heterogeneous PDE families can be trained together.

## Default Layout

The default paper-style config is `src/configs/data/easy_test.yaml`. It assumes data under `pde_data/`:

```text
pde_data/
  ShallowWater2D/
    train.hdf5
    test.hdf5
  allen-cahn/
    allen-cahn_train.hdf5
    allen-cahn_test.hdf5
  cfddata_hard/                  # CNS (1.0,0.01)
    train.hdf5
    val.hdf5
    test.hdf5
  cfddata/                       # CNS (0.1,0.01)
    train/*.hdf5
    val/*.hdf5
    test/*.hdf5
  gray_scott_reaction_diffusion/
    train/*.hdf5
    test/*.hdf5
```

In this layout, `cfddata_hard` corresponds to CNS `(1.0,0.01)`, and `cfddata` corresponds to CNS `(0.1,0.01)`.

Most single-file datasets use:

```text
data: (N, T, H, W, C)
grid/t, grid/x, grid/y: optional coordinate arrays
```

Folder-based datasets load all `.hdf5` or `.h5` files in the configured split folder. For `cfddata` (CNS `(0.1,0.01)`), each file should contain:

```text
data: (N_file, T, H, W, C)
```

## Custom Dataset

To add a new PDE family:

1. Add a dataset class in `src/data_utils/all_datasets.py` that yields dictionaries with a `data` tensor shaped `(t_num, x_num, x_num, dim)`.
2. Register it in `src/dataset.py` under `ALL_DATASETS`.
3. Add a config file under `src/configs/data/`.
4. Set `types`, `max_output_dimension`, `t_num`, `x_num`, and per-dataset metadata such as `dim`, `t_step`, `data_path`, or `folder`.

Example:

```yaml
types: [my_pde]
max_output_dimension: 3
t_num: 20
x_num: 128
tie_fields: 1

my_pde:
  data_path:
    train: pde_data/my_pde/train.hdf5
    val: pde_data/my_pde/val.hdf5
    test: pde_data/my_pde/test.hdf5
  t_step: 1
  x_num: 128
  dim: 3
```
