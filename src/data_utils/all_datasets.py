import os
import h5py
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import time
from torch.utils.data import get_worker_info

# import torchdata.datapipes as dp
# from torchdata.datapipes.iter import IterDataPipe

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.datapipe import IterDataPipe

from logging import getLogger

logger = getLogger()

DatasetIdx = {
    "react_diff": 0,
    "shallow_water": 1,
    "incom_ns": 2,
    "com_ns": 3,
    "incom_ns_arena": 4,
    "incom_ns_arena_u": 5,
    "cfddata": 6,
    "allen_cahn": 7,
    "black_scholes_barenblatt": 8,
    "burgers": 9,
    "gray_scott": 10,
    "shear_flow": 11,
    "acoustic_scattering": 12,
    "fpo_geometry_medium_single_obstacle": 13,
}


class myIterDp(IterDataPipe):
    """
    Base class for all iterable datasets, and contains some shared helper methods.
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__()

        # general initialization, should be called by all subclasses

        # self.train = split == "train"
        self.train = train
        self.params = params
        self.symbol_env = symbol_env
        self.split = split

        self.num_workers = params.num_workers if train else params.num_workers_eval
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.t_num = params.data.t_num 
        self.x_num = params.data.x_num 

        if params.overfit_test:
            self.random_start = params.data.random_start["test"]
        else:
            self.random_start = params.data.random_start[split]

        self.rng = None
        self.type_label = ""
        self.fully_shuffled = False
        self.symbol_ids = None

    def init_rng(self):
        """
        Initialize different random generator for each worker.
        """
        if self.rng is not None:
            return

        worker_id = self.get_worker_id()
        self.worker_id = worker_id
        params = self.params
        if self.train:
            # base_seed = params.base_seed
            base_seed = np.random.randint(1_000_000_000)  # ensure seed is different for each epoch

            seed = [worker_id, DatasetIdx[self.type_label], params.global_rank, base_seed]
            self.rng = np.random.default_rng(seed)
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, base_seed)")
        else:
            seed = [worker_id, DatasetIdx[self.type_label], params.global_rank, params.test_seed]
            self.rng = np.random.default_rng(seed) 
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, test_seed)")

    def get_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id

    def augment_data(self, data: np.ndarray):
        """
        data: (t_num, x_num, x_num, data_dim)
        """
        if self.train:
            # self.init_rng()
            if self.params.noise > 0:
                # add noise
                gamma = self.params.noise
                if self.params.noise_type == "multiplicative":
                    cur_noise = self.rng.normal(size=data.shape).astype(np.single)
                    data = data + gamma * np.abs(data) * cur_noise
                elif self.params.noise_type == "additive":
                    cur_noise = self.rng.normal(size=data.shape).astype(np.single)
                    eps = 1e-6
                    sigma = gamma * np.linalg.norm(data) / (np.linalg.norm(cur_noise) + eps)
                    data = data + sigma * cur_noise# data + scaled_noise

            if self.params.flip:
                # flip data
                flip = self.rng.choice(4)
                if flip == 1:
                    data = np.flip(data, axis=1)
                elif flip == 2:
                    data = np.flip(data, axis=2)
                elif flip == 3:
                    data = np.flip(data, axis=(1, 2))

            if self.params.rotate:
                # rotate data
                rot = self.rng.choice(4)
                if rot > 0:
                    data = np.rot90(data, axes=(1, 2), k=rot)

        return np.ascontiguousarray(data)

    def get_iter_range(self, total_len):
        if self.num_workers <= 1:
            return np.arange(total_len)
        else:
            return np.arange(self.worker_id, total_len, self.num_workers)


    def sample_initial_time(self, max_len):
        data_limit = max_len - self.t_num * self.t_step
        start_limit = self.params.data.random_start.start_max
        if start_limit > 0:
            data_limit = min(data_limit, start_limit)
        if data_limit <= 0:
            return 0
        else:
            return self.rng.integers(0, data_limit)


class ReactDiff2D(myIterDp):
    """
    PDEBench 2D reaction_diffusion dataset.
        size:  1000
        t_num: 101           [0, 5] dt=0.05
        x_num: (128, 128)    (-1, 1)
        data_dim: 2
        bc: neumann

    Dataset structure:
    0000 - 0999
        data: (101, 128, 128, 2)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "react_diff"

        self.t_step = params.data.react_diff.t_step
        self.x_step = params.data.react_diff.x_num // params.data.x_num
        self.data_path = params.data.react_diff.data_path[split]
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def get_iter_range(self, total_len):
        # split number of workers (files already split based on train/val/test)

        start = 0
        end = total_len

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def __iter__(self):
        self.init_rng()
        with h5py.File(self.data_path, "r", locking=False) as hf:
            iter_range = self.get_iter_range(len(hf['data']))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                t0 = self.sample_initial_time(hf["data"].shape[1]) if self.random_start else 0

                data = hf["data"][
                     i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                ]  # (t_num, x_num, x_num, 1)
                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                d = {}
                # d["type"] = self.type_label

                if not self.params.data.tie_fields:  
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data


                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                # x_grid = sample["grid"]["x"][::self.x_step]  # (x_num, )
                # y_grid = sample["grid"]["y"][::self.x_step]  # (x_num, )

                yield d


class ShallowWater2D(myIterDp):
    """
    PDEBench 2D shallow_water dataset.
        size:  1000
        t_num: 101            [0, 1] dt=0.01
        x_num: (128, 128)     (-2.5, 2.5)
        data_dim: 1
        bc: neumann

    Dataset structure:
    0000 - 0999
        data: (101, 128, 128, 1)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "shallow_water"

        self.t_step = params.data.shallow_water.t_step
        self.x_step = max(params.data.shallow_water.x_num // params.data.x_num,1)
        self.data_path = params.data.shallow_water.data_path[split]
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()
        with h5py.File(self.data_path, "r", locking=False) as hf:
            iter_range = self.get_iter_range(len(hf['data']))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)
            for i in iter_range:
                t0 = self.sample_initial_time(hf["data"].shape[1]) if self.random_start else 0

                data = hf["data"][
                    i,t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                ]  # (t_num, x_num, x_num, 1)
                d = {}
                # d["type"] = self.type_label

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.use_raw_time:
                    t_grid = hf["grid"]["t"][t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                # x_grid = sample["grid"]["x"][::self.x_step]  # (x_num, )
                # y_grid = sample["grid"]["y"][::self.x_step]  # (x_num, )
                yield d

class AllenCahn2D(myIterDp):
    """
    PDEBench 2D Allen–Cahn dataset.
        size:  1000
        t_num: 101            
        x_num: (128, 128)    
        data_dim: 1

    Dataset structure:
    0000 - 0999
        data: (101, 128, 128, 1)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization
        self.type_label = "allen_cahn"

        self.t_step = params.data.allen_cahn.t_step
        self.x_step = max(params.data.allen_cahn.x_num // params.data.x_num,1)
        self.data_path = params.data.allen_cahn.data_path[split]
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()
        with h5py.File(self.data_path, "r", locking=False) as hf:
            iter_range = self.get_iter_range(len(hf["data"]))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                t0 = self.sample_initial_time(hf["data"].shape[1]) if self.random_start else 0

                data = hf["data"][
                    i,
                    t0 : (t0 + self.t_num * self.t_step) : self.t_step,
                    :: self.x_step,
                    :: self.x_step,
                ]  # (t_num, x_num, x_num, 1)

                d = {}

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(
                        nt,
                        nx,
                        ny,
                        self.params.data.max_output_dimension,
                        dtype=data.dtype,
                    )
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.use_raw_time:
                    t_grid = hf["grid"]["t"][
                        t0 : (t0 + self.t_num * self.t_step) : self.t_step
                    ]  # (t_num,)
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                # x_grid = hf["grid"]["x"][:: self.x_step]  # (x_num,)
                # y_grid = hf["grid"]["y"][:: self.x_step]  # (x_num,)

                yield d

class Burgers2D(myIterDp):
    """
    PDEBench 2D Burgers dataset.
        size:  1000
        t_num: 101           [0, T] dt as given by dataset
        x_num: (128, 128)    (domain as given by dataset)
        data_dim: 2          (e.g., two velocity components)

    Dataset structure:
    0000 - 0999
        data: (101, 128, 128, 2)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization
        self.type_label = "burgers"  

        self.t_step = params.data.burgers.t_step
        self.x_step = params.data.burgers.x_num // params.data.x_num
        self.data_path = params.data.burgers.data_path[split]
        self.fully_shuffled = True  

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def get_iter_range(self, total_len):
        start = 0
        end = total_len

        if self.num_workers <= 1:
            return np.arange(start, end)
        else:
            return np.arange(start + self.worker_id, end, self.num_workers)

    def __iter__(self):
        self.init_rng()
        with h5py.File(self.data_path, "r", locking=False) as hf:
            iter_range = self.get_iter_range(len(hf["data"]))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                t0 = self.sample_initial_time(hf["data"].shape[1]) if self.random_start else 0

                data = hf["data"][
                    i,
                    t0 : (t0 + self.t_num * self.t_step) : self.t_step,
                    :: self.x_step,
                    :: self.x_step,
                ]  # (t_num, x_num, x_num, 2)

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                d = {}
                # d["type"] = self.type_label

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(
                        nt,
                        nx,
                        ny,
                        self.params.data.max_output_dimension,
                        dtype=data.dtype,
                    )
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                yield d


class BlackScholesBarenblatt2D(myIterDp):
    """
    PDEBench 2D Black–Scholes–Barenblatt dataset.
        size:  1000
        t_num: 101           
        x_num: (128, 128)     (domain as given by dataset)
        data_dim: 1


    Dataset structure (per sample index 0000–0999):
        data: (101, 128, 128, 1)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization
        self.type_label = "black_scholes_barenblatt"

        self.t_step = params.data.black_scholes_barenblatt.t_step
        self.x_step = params.data.black_scholes_barenblatt.x_num // params.data.x_num
        self.data_path = params.data.black_scholes_barenblatt.data_path[split]
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()
        with h5py.File(self.data_path, "r", locking=False) as hf:
            iter_range = self.get_iter_range(len(hf["data"]))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                # choose starting time index
                t0 = self.sample_initial_time(hf["data"].shape[1]) if self.random_start else 0

                # data shape: (t_num, x_num, x_num, 1)
                data = hf["data"][
                    i,
                    t0 : (t0 + self.t_num * self.t_step) : self.t_step,
                    :: self.x_step,
                    :: self.x_step,
                ]

                d = {}

                # optional data augmentation
                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(
                        nt,
                        nx,
                        ny,
                        self.params.data.max_output_dimension,
                        dtype=data.dtype,
                    )
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.use_raw_time:
                    t_grid = hf["grid"]["t"][
                        t0 : (t0 + self.t_num * self.t_step) : self.t_step
                    ]  # (t_num,)
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                # x_grid = hf["grid"]["x"][:: self.x_step]  # (x_num,)
                # y_grid = hf["grid"]["y"][:: self.x_step]  # (x_num,)

                yield d



class IncomNS2D(myIterDp):
    """
    PDEBench 2D incompressible navier-stokes dataset. (assumes n_gpu <= 4)
        size:  1096
        t_num: 1000              [0, 5) dt=0.005
        x_num: (512, 512)        [0, 1] ?
        data_dim: 2+1 
        bc: dirichlet

    Dataset structure:
    274 files (missing idx 49). In each file:
        velocity:  (4, 1000, 512, 512, 2)
        particles: (4, 1000, 512, 512, 1)
        force:     (4, 512, 512, 2) 
        t:         (4, 1000)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "incom_ns"

        self.t_step = params.data.incom_ns.t_step
        self.x_step = params.data.incom_ns.x_num // params.data.x_num

        self.folder = params.data.incom_ns.folder
        self.data_files = sorted([f for f in os.listdir(self.folder) if f.endswith(".hdf5")])
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()
        iter_range = self.get_iter_range(len(self.data_files))

        if self.train:
            iter_range = self.rng.permutation(iter_range)

        for file_idx in iter_range:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r", locking=False) as hf:
                file_size = len(hf["velocity"]) 

                for i in range(self.local_rank, file_size, self.n_gpu_per_node):
                    t0 = self.sample_initial_time(hf["velocity"].shape[1]) if self.random_start else 0

                    velocity = hf["velocity"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  
                    particles = hf["particles"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  

                    d = {}
                    # d["type"] = self.type_label

                    data = np.concatenate([velocity, particles], axis=-1)  # (t_num, x_num, x_num, 3)  
                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t_grid = hf["t"][i, t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                        t_grid = torch.from_numpy(t_grid).float()
                        d["t"] = t_grid

                    if self.params.symbol.symbol_input:
                        d["symbol_input"] = self.symbol_ids

                    # force = hf["force"][i, :: self.x_step, :: self.x_step]  # (x_num, x_num, 2)

                    yield d


class ComNS2D(myIterDp):
    """
    PDEBench 2D compressible navier-stokes dataset.
        t_num: 21           [0, 1] dt=0.05
        data_dim: 4
        bc: periodic

        Raw shape: (now all converted to the same space grid (128, 128))

        Random fields - Regular:
            size:  40000
            x_num: (128, 128)

        Random fields - Euler (low shear and bulk viscosity):
            size:  2000
            x_num: (512, 512)

        Turbulence:
            size: 2000
            x_num: (512, 512)

    Raw Dataset structure (now all converted to 128x128):
        Random fields - Regular (4 files). In each file:
            Vx:           (10000, 21, 128, 128)
            Vy:           (10000, 21, 128, 128)
            density:      (10000, 21, 128, 128)
            pressure:     (10000, 21, 128, 128)
            t-coordinate: (22,)                    [0, 1.05]
            x-coordinate: (128,)                   (0, 1)
            y-coordinate: (128,)

        Random fields - Euler (2 files). In each file:
            Vx:           (1000, 21, 512, 512)
            Vy:           (1000, 21, 512, 512)
            density:      (1000, 21, 512, 512)
            pressure:     (1000, 21, 512, 512)
            t-coordinate: (22,)
            x-coordinate: (512,)                   (0, 1)   (average every 4 points gives the previous 128 grid)
            y-coordinate: (512,)

        Turbulence (2 files). In each file::
            Vx:           (1000, 21, 512, 512)
            Vy:           (1000, 21, 512, 512)
            density:      (1000, 21, 512, 512)
            pressure:     (1000, 21, 512, 512)
            t-coordinate: (22,)
            x-coordinate: (512,)                   (0, 1)
            y-coordinate: (512,)
    """
    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "com_ns"

        self.t_step = params.data.com_ns.t_step
        self.x_step = max(params.data.com_ns.x_num // params.data.x_num,1)
        self.data_path = params.data.com_ns.data_path[split]
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()
        with h5py.File(self.data_path, "r", locking=False) as hf:
            iter_range = self.get_iter_range(len(hf['data']))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)
            for i in iter_range:
                t0 = self.sample_initial_time(hf["data"].shape[1]) if self.random_start else 0

                data = hf['data'][
                    i,t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                ]  # (t_num, x_num, x_num, 1)
                d = {}
                # d["type"] = self.type_label

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                    tmp[..., self.c_mask_bool] = data
                    data = tmp
                    

                d["data"] = data

                if self.params.use_raw_time:
                    t_grid = hf["t-coordinate"][t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid

                if self.params.symbol.symbol_input:
                    # tree = self.symbol_env.generator.get_tree(self.type_label, {"eta": 0.1, "zeta": 0.1})
                    # tree_encoded = self.symbol_env.equation_encoder.encode(tree)
                    # symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
                    # d["symbol_input"] = symbol_input

                    d["symbol_input"] = self.symbol_ids

                # x_grid = hf["x-coordinate"][:: self.x_step]  # (x_num, )
                # y_grid = hf["y-coordinate"][:: self.x_step]  # (x_num, )

                yield d



class GrayScott2D(myIterDp):
    """
    Gray-Scott 2D reaction-diffusion dataset.

        boundary_conditions/x_periodic/mask: (128,) bool
        boundary_conditions/y_periodic/mask: (128,) bool
        dimensions/time: (1001,) float32
        dimensions/x:    (128,)  float32
        dimensions/y:    (128,)  float32
        scalars/F:       ()      float32
        scalars/k:       ()      float32
        t0_fields/A:     (160, 1001, 128, 128) float32
        t0_fields/B:     (160, 1001, 128, 128) float32
    """
    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        self.type_label = "gray_scott"
        self.split = split

        self.t_step = params.data.gray_scott.t_step
        self.x_step = max(params.data.gray_scott.x_num // params.data.x_num,1)


        folder_cfg = params.data.gray_scott.folder  

        if split not in folder_cfg:
            raise ValueError(f"Unknown split '{split}' for gray_scott.folder; "
                             f"available keys: {list(folder_cfg.keys())}")

        self.folder = folder_cfg[split]

        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"Gray-Scott folder not found: {self.folder}")


        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if f.endswith(".hdf5") or f.endswith(".h5")]
        )
        if len(self.data_files) == 0:
            raise RuntimeError(f"No .hdf5 files found in {self.folder}")

        self.fully_shuffled = True

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()
    def __iter__(self):
        self.init_rng()

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        all_indices = list(range(len(self.data_files)))

        if self.train:
            all_indices = self.rng.permutation(all_indices)


        file_indices = [
            idx for pos, idx in enumerate(all_indices)
            if pos % num_workers == worker_id
        ]
        for file_idx in file_indices:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r", locking=False) as hf:
                A_ds = hf["t0_fields/A"]   # (160, 1001, 128, 128)
                B_ds = hf["t0_fields/B"]   # (160, 1001, 128, 128)

                file_size = A_ds.shape[0]  


                for i in range(self.local_rank, file_size, self.n_gpu_per_node):
                    t_total = A_ds.shape[1]  # 1001
                    t0 = self.sample_initial_time(t_total) if self.random_start else 0

                    t_end = t0 + self.t_num * self.t_step
                    if t_end > t_total:
                        t0 = max(0, t_total - self.t_num * self.t_step)
                        t_end = t0 + self.t_num * self.t_step


                    A = A_ds[i, t0:t_end:self.t_step, ::self.x_step, ::self.x_step]
                    B = B_ds[i, t0:t_end:self.t_step, ::self.x_step, ::self.x_step]


                    data = np.stack([A, B], axis=-1)

                    d = {}

                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(
                            nt, nx, ny,
                            self.params.data.max_output_dimension,
                            dtype=data.dtype,
                        )
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t_grid_full = hf["dimensions/time"][:]  # (1001,)
                        t_grid = t_grid_full[t0:t_end:self.t_step]  # (t_num,)
                        t_grid = torch.from_numpy(t_grid).float()
                        d["t"] = t_grid

                    if self.params.symbol.symbol_input:
                        d["symbol_input"] = self.symbol_ids


                    # d["F"] = float(hf["scalars/F"][()])
                    # d["k"] = float(hf["scalars/k"][()])
                    yield d


class ShearFlow2D(myIterDp):
    """
    Shear-flow 2D dataset (PDEBench).

        boundary_conditions/x_periodic/mask: (256,) bool
        boundary_conditions/y_periodic/mask: (512,) bool
        dimensions/time: (200,) float64
        dimensions/x:    (256,) float32
        dimensions/y:    (512,) float32
        scalars/Reynolds: ()
        scalars/Schmidt: ()
        t0_fields/pressure: (32, 200, 256, 512) float32
        t0_fields/tracer:   (32, 200, 256, 512) float32
        t1_fields/velocity: (32, 200, 256, 512, 2) float32

    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        self.type_label = "shear_flow"
        self.split = split


        self.t_step = params.data.shear_flow.t_step
        self.x_step = params.data.shear_flow.x_num // params.data.x_num


        folder_cfg = params.data.shear_flow.folder 

        if split not in folder_cfg:
            raise ValueError(
                f"Unknown split '{split}' for shear_flow.folder; available keys: {list(folder_cfg.keys())}"
            )

        self.folder = folder_cfg[split]  
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"ShearFlow folder not found: {self.folder}")

        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if f.endswith(".hdf5") or f.endswith(".h5")]
        )
        if len(self.data_files) == 0:
            raise RuntimeError(f"No .hdf5 files found in {self.folder}")

        self.fully_shuffled = True

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        all_indices = list(range(len(self.data_files)))

        if self.train:
            all_indices = self.rng.permutation(all_indices)


        file_indices = [
            idx for pos, idx in enumerate(all_indices)
            if pos % num_workers == worker_id
        ]
        for file_idx in file_indices:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r", locking=False) as hf:
                P_ds = hf["t0_fields/pressure"]   # (32, 200, 256, 512)
                T_ds = hf["t0_fields/tracer"]     # (32, 200, 256, 512)
                V_ds = hf["t1_fields/velocity"]   # (32, 200, 256, 512, 2)

                file_size = P_ds.shape[0]         # 32 
                t_total = P_ds.shape[1]           # 200

                # 多 GPU：按第 0 维切分样本
                for i in range(self.local_rank, file_size, self.n_gpu_per_node):
                    t0 = self.sample_initial_time(t_total) if self.random_start else 0
                    t_end = t0 + self.t_num * self.t_step
                    if t_end > t_total:
                        t0 = max(0, t_total - self.t_num * self.t_step)
                        t_end = t0 + self.t_num * self.t_step


                    # P, T: (t_num, nx', ny')
                    P = P_ds[i, t0:t_end:self.t_step, ::self.x_step, ::self.x_step*2]
                    Tr = T_ds[i, t0:t_end:self.t_step, ::self.x_step, ::self.x_step*2]
                    # V: (t_num, nx', ny', 2)
                    V = V_ds[i, t0:t_end:self.t_step, ::self.x_step, ::self.x_step*2, :]

                    P_ch = P[..., None]   # (t_num, nx', ny', 1)
                    Tr_ch = Tr[..., None] # (t_num, nx', ny', 1)
                    data = np.concatenate([P_ch, Tr_ch, V], axis=-1)  # (t_num, nx', ny', 4)

                    d = {}

                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(
                            nt, nx, ny,
                            self.params.data.max_output_dimension,
                            dtype=data.dtype,
                        )
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t_grid_full = hf["dimensions/time"][:]   # (200,)
                        t_grid = t_grid_full[t0:t_end:self.t_step]  # (t_num,)
                        t_grid = torch.from_numpy(t_grid).float()
                        d["t"] = t_grid

                    if self.params.symbol.symbol_input:
                        d["symbol_input"] = self.symbol_ids


                    # Re = float(hf["scalars/Reynolds"][()])
                    # Sc = float(hf["scalars/Schmidt"][()])
                    # d["Re"], d["Sc"] = Re, Sc

                    yield d



class AcousticScattering2D(myIterDp):
    """
    Acoustic scattering discontinuous 2D dataset (only time-dependent fields).

        dimensions/time: (102,) float32
        dimensions/x:    (256,) float32
        dimensions/y:    (256,) float32

        t0_fields/pressure: (100, 102, 256, 256)      float32
        t1_fields/velocity: (100, 102, 256, 256, 2)   float32


    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        self.type_label = "acoustic_scattering"   
        self.split = split


        self.t_step = params.data.acoustic_scattering.t_step
        self.x_step = params.data.acoustic_scattering.x_num // params.data.x_num


        folder_cfg = params.data.acoustic_scattering.folder  # DictConfig: {train: ..., val: ..., test: ...}

        if split not in folder_cfg:
            raise ValueError(
                f"Unknown split '{split}' for acoustic_scattering.folder; "
                f"available keys: {list(folder_cfg.keys())}"
            )

        self.folder = folder_cfg[split]  
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"AcousticScattering folder not found: {self.folder}")


        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if f.endswith(".hdf5") or f.endswith(".h5")]
        )
        if len(self.data_files) == 0:
            raise RuntimeError(f"No .hdf5 files found in {self.folder}")

        self.fully_shuffled = True


        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input


        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()


    def __iter__(self):
        self.init_rng()
        iter_range = self.get_iter_range(len(self.data_files))
        if self.train:
            iter_range = self.rng.permutation(iter_range)
        for file_idx in iter_range:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r", locking=False) as hf:
                p_ds = hf["t0_fields/pressure"]      # (100, 102, 256, 256)
                v_ds = hf["t1_fields/velocity"]      # (100, 102, 256, 256, 2)

                file_size = p_ds.shape[0]          
                t_total = p_ds.shape[1]              

                for i in range(self.local_rank, file_size, self.n_gpu_per_node):
                    t0 = self.sample_initial_time(t_total) if self.random_start else 0
                    t_end = t0 + self.t_num * self.t_step
                    if t_end > t_total:
                        t0 = max(0, t_total - self.t_num * self.t_step)
                        t_end = t0 + self.t_num * self.t_step




                    # p: (t_num, nx', ny')
                    p = p_ds[i, t0:t_end:self.t_step, ::self.x_step, ::self.x_step]
                    # v: (t_num, nx', ny', 2)
                    v = v_ds[i, t0:t_end:self.t_step, ::self.x_step, ::self.x_step, :]



                    p_ch = p[..., None]                  # (t_num, nx', ny', 1)
                    data = np.concatenate([p_ch, v], axis=-1)

                    d = {}

                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(
                            nt, nx, ny,
                            self.params.data.max_output_dimension,
                            dtype=data.dtype,
                        )
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t_grid_full = hf["dimensions/time"][:]  # (102,)
                        t_grid = t_grid_full[t0:t_end:self.t_step]  # (t_num,)
                        t_grid = torch.from_numpy(t_grid).float()
                        d["t"] = t_grid

                    if self.params.symbol.symbol_input:
                        d["symbol_input"] = self.symbol_ids


                    yield d


class IncomNS2DArena(myIterDp):
    """
    PDEArena 2D incompressible navier-stokes dataset (conditioned).
        size: 2496/608/608       train/val/test
        t_num: 56                [18, 102] dt=1.5
        x_num: (128, 128)        [0, 32]
        data_dim: 2+1
        bc: dirichlet for velocity, neumann for scalar

    Dataset structure:
    train/val/test: 78/19/19 files. In each file:
        train/valid/test:
            vx:    (32, 56, 128, 128)
            vy:    (32, 56, 128, 128)
            u:     (32, 56, 128, 128)
            buo_y: (32,)
            t:     (32, 56)
            x:     (32, 128)
            y:     (32, 128)
            dt:    (32,)
            dx:    (32,)
            dy:    (32,)
    """

    split_to_name = {"train": "train", "val": "valid", "test": "test"}

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "incom_ns_arena"

        self.t_step = params.data.incom_ns_arena.t_step
        self.x_step = params.data.incom_ns_arena.x_num // params.data.x_num

        self.folder = params.data.incom_ns_arena.folder
        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if self.split_to_name[self.split] in f and f.endswith(".hdf5")]
        )
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        # if self.params.symbol.symbol_input:
        #     tree = self.symbol_env.generator.get_tree(self.type_label)
        #     tree_encoded = self.symbol_env.equation_encoder.encode(tree)
        #     symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
        #     self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def get_iter_range(self, total_len):
        # split number of workers (files already split based on train/val/test)

        start = 0
        end = total_len

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def __iter__(self):
        self.init_rng()
        iter_range = self.get_iter_range(len(self.data_files))

        if self.train:
            iter_range = self.rng.permutation(iter_range)

        for file_idx in iter_range:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r", locking=False) as f:
                hf = f[self.split_to_name[self.split]]
                file_size = len(hf["vx"])

                file_iter_range = np.arange(self.local_rank, file_size, self.n_gpu_per_node)
                if self.train:
                    file_iter_range = self.rng.permutation(file_iter_range)

                for i in file_iter_range:
                    t0 = self.sample_initial_time(hf["vx"].shape[1]) if self.random_start else 0

                    vx = hf["vx"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    vy = hf["vy"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    u = hf["u"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)

                    data = np.stack([vx, vy, u], axis=-1)  # (t_num, x_num, x_num, 3)
                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    d = {}
                    # d["type"] = self.type_label

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t = hf["t"][i, t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                        t = torch.from_numpy(t).float()
                        d["t"] = t

                    if self.params.symbol.symbol_input:
                        buo_y = hf["buo_y"][i]
                        tree = self.symbol_env.generator.get_tree(self.type_label, {"F": buo_y})
                        tree_encoded = self.symbol_env.equation_encoder.encode(tree)
                        symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
                        d["symbol_input"] = symbol_input

                        # d["symbol_input"] = self.symbol_ids

                    # x = hf["x"][i, :: self.x_step]  # (x_num, )
                    # y = hf["y"][i, :: self.x_step]  # (x_num, )

                    yield d


class IncomNS2DArenaU(myIterDp):
    """
    PDEArena 2D incompressible navier-stokes dataset (unconditioned).
        size: 1664/1664/1664     train/val/test
        t_num: 14                [18, 102] dt=6.46
        x_num: (128, 128)        [0, 32]
        data_dim: 2+1
        bc: dirichlet for velocity, neumann for scalar

    Dataset structure:
    train/val/test: 52/52/52 files. In each file:
        train/valid/test:
            vx:    (100, 14, 128, 128)
            vy:    (100, 14, 128, 128)
            u:     (100, 14, 128, 128)
            buo_y: (100,)
            t:     (100, 14)
            x:     (100, 128)
            y:     (100, 128)
            dt:    (100,)
            dx:    (100,)
            dy:    (100,)
    """

    split_to_name = {"train": "train", "val": "valid", "test": "test"}

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "incom_ns_arena_u"

        self.t_step = params.data.incom_ns_arena_u.t_step
        self.x_step = params.data.incom_ns_arena_u.x_num // params.data.x_num

        self.folder = params.data.incom_ns_arena_u.folder
        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if self.split_to_name[self.split] in f and f.endswith(".hdf5")]
        )
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def get_iter_range(self, total_len):
        # split number of workers (files already split based on train/val/test)

        start = 0
        end = total_len

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def __iter__(self):
        self.init_rng()
        iter_range = self.get_iter_range(len(self.data_files))

        if self.train:
            iter_range = self.rng.permutation(iter_range)

        for file_idx in iter_range:

            data_path = os.path.join(self.folder, self.data_files[file_idx])
            with h5py.File(data_path, "r", locking=False) as f:
                hf = f[self.split_to_name[self.split]]
                file_size = len(hf["vx"])
                file_iter_range = np.arange(self.local_rank, file_size, self.n_gpu_per_node)
                if self.train:
                    file_iter_range = self.rng.permutation(file_iter_range)

                for i in file_iter_range:
                    t0 = self.sample_initial_time(hf["vx"].shape[1]) if self.random_start else 0

                    vx = hf["vx"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    vy = hf["vy"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    u = hf["u"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)

                    data = np.stack([vx, vy, u], axis=-1)  # (t_num, x_num, x_num, 3)
                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    d = {}
                    # d["type"] = self.type_label

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t = hf["t"][i, t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                        t = torch.from_numpy(t).float()
                        d["t"] = t

                    if self.params.symbol.symbol_input:
                        # buo_y = hf["buo_y"][i]
                        # tree = self.symbol_env.generator.get_tree(self.type_label, {"F": buo_y})
                        # tree_encoded = self.symbol_env.equation_encoder.encode(tree)
                        # symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
                        # d["symbol_input"] = symbol_input

                        d["symbol_input"] = self.symbol_ids

                    # x = hf["x"][i, :: self.x_step]  # (x_num, )
                    # y = hf["y"][i, :: self.x_step]  # (x_num, )
                    yield d


class CFDdata2D(myIterDp):
    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        self.type_label = "cfddata"
        self.split = split

        # time step and spatial downsampling
        self.t_step = params.data.cfddata.t_step
        self.x_step = max(params.data.cfddata.x_num // params.data.x_num,1)

        # ===== folder-by-split (same pattern as AcousticScattering2D) =====
        folder_cfg = params.data.cfddata.folder  # DictConfig: {train: ..., val: ..., test: ...}
        if split not in folder_cfg:
            raise ValueError(
                f"Unknown split '{split}' for cfddata.folder; "
                f"available keys: {list(folder_cfg.keys())}"
            )
        self.folder = folder_cfg[split]
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"cfddata folder not found: {self.folder}")

        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if f.endswith(".hdf5") or f.endswith(".h5")]
        )
        if len(self.data_files) == 0:
            raise RuntimeError(f"No .hdf5 files found in {self.folder}")
        # ================================================================

        self.fully_shuffled = True

        # symbolic input
        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        # channel mask
        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

        assert not params.use_raw_time

    def __iter__(self):
        self.init_rng()

        iter_range = self.get_iter_range(len(self.data_files))
        if self.train:
            iter_range = self.rng.permutation(iter_range)

        for file_idx in iter_range:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r", locking=False) as f:
                data_ds = f["data"]  # (N_file, T, H, W, C)

                file_size = data_ds.shape[0]
                t_total = data_ds.shape[1]

                # multi-GPU: split along sample dimension
                for i in range(self.local_rank, file_size, self.n_gpu_per_node):
                    t0 = self.sample_initial_time(t_total) if self.random_start else 0
                    t_end = t0 + self.t_num * self.t_step
                    if t_end > t_total:
                        t0 = max(0, t_total - self.t_num * self.t_step)
                        t_end = t0 + self.t_num * self.t_step


                    data = data_ds[
                        i,
                        t0:t_end:self.t_step,
                        ::self.x_step,
                        ::self.x_step,
                        ...,
                    ]  # (t_num, x_num, x_num, C)


                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    d = {}

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(
                            nt, nx, ny,
                            self.params.data.max_output_dimension,
                            dtype=data.dtype,
                        )
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.symbol.symbol_input:
                        d["symbol_input"] = self.symbol_ids

                    yield d

class FPOGeometryMediumSingleObstacle2D(myIterDp):
    """
    PreGen-NavierStokes-2D
    FPO_Geometry_Medium_SingleObstacle dataset.

    Dataset structure:
    0000 - (N-1)
        data: (T, H, W, 6)
        grid
            t: (T,)
            x: (H,) or (H, ?)  (depending on file)
            y: (W,) or (W, ?)

    Channel meaning:
        data[..., 0:3] = physical variables (used)
        data[..., 4]   = datamask (hole=1, data=0)
                         -> when making `hole_mask`, MUST invert:
                            hole_mask = 1 - data[..., 4]
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        self.type_label = "fpo_geometry_medium_single_obstacle"

        # time / space sampling
        self.t_step = params.data.fpo_geometry_medium_single_obstacle.t_step
        self.x_step = max(params.data.fpo_geometry_medium_single_obstacle.x_num // params.data.x_num, 1)

        self.data_path = params.data.fpo_geometry_medium_single_obstacle.data_path[split]
        self.fully_shuffled = True

        # optional symbolic input
        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        # (optional) tie_fields-style padding
        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()

        with h5py.File(self.data_path, "r", locking=False) as hf:
            # hf["data"]: (N, T, H, W, 6)
            N = len(hf["data"])

            iter_range = self.get_iter_range(N)[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                # choose random start time
                t0 = self.sample_initial_time(hf["data"].shape[1]) if self.random_start else 0

                raw = hf["data"][
                    i,
                    t0 : (t0 + self.t_num * self.t_step) : self.t_step,
                    :: self.x_step,
                    :: self.x_step,
                    :
                ]  # (t_num, x_num, x_num, 6)

                d = {}

                # raw[..., 4] is mask where hole=1
                raw_mask = raw[..., 4]                            # (t_num, x, y)
                hole_mask = np.clip(raw_mask, 0.0, 1.0)          # safety
                hole_mask = hole_mask[..., None]            # (t_num, x, y, 1)


                data = raw[..., 0:3]                              # (t_num, x, y, 3)

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()
                hole_mask = torch.from_numpy(hole_mask).float()   # (t_num, x, y)


                d["hole_mask"] = hole_mask
                d["data"] = data


                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(
                        nt, nx, ny,
                        self.params.data.max_output_dimension,
                        dtype=data.dtype,
                    )
                    tmp[..., self.c_mask_bool] = data
                    data = tmp


                if self.params.use_raw_time:
                    t_grid = hf["grid"]["t"][t0 : (t0 + self.t_num * self.t_step) : self.t_step]
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid


                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                yield d
