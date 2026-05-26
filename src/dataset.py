import torchdata.datapipes as dp
from data_utils import all_datasets as ds
from logging import getLogger

logger = getLogger()

ALL_DATASETS = {
    "react_diff": ds.ReactDiff2D,
    "shallow_water": ds.ShallowWater2D,
    "incom_ns": ds.IncomNS2D,
    "com_ns": ds.ComNS2D,
    "incom_ns_arena": ds.IncomNS2DArena,
    "incom_ns_arena_u": ds.IncomNS2DArenaU,
    "cfddata": ds.CFDdata2D,
    "allen_cahn": ds.AllenCahn2D,
    "black_scholes_barenblatt": ds.BlackScholesBarenblatt2D,
    "burgers": ds.Burgers2D,
    "gray_scott": ds.GrayScott2D,
    "shear_flow": ds.ShearFlow2D,
    "acoustic_scattering": ds.AcousticScattering2D,
    "fpo_geometry_medium_single_obstacle": ds.FPOGeometryMediumSingleObstacle2D,
}


def get_dataset(params, symbol_env, split):
    types = params.data.types

    if split == "train":
        datasets = {}
        for t in types:
            ds = ALL_DATASETS[t](params, symbol_env, split, train=True)
            if not ds.fully_shuffled:
                # during training, shuffle iterable datasets that are not fully shuffled 
                ds = ds.shuffle(buffer_size=ds.buffer_size)
            
            # datasets.append(ds.cycle())
            datasets[ds.cycle()] = params.data.sampler[t]
            # datasets[ds] = params.data.sampler[t]
        if params.data.sampler.uniform:
            return dp.iter.Multiplexer(*datasets)
        else:
            return dp.iter.SampleMultiplexer(datasets)
    else:

        datasets = {}
        for t in types:
            use_split = "train" if params.overfit_test else split

            ds = ALL_DATASETS[t](params, symbol_env, split=use_split, train=False)
            datasets[t] = ds

        return datasets
