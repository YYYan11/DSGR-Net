import json
from pathlib import Path

import numpy as np
import torch

from dsgr_net import ProcessConstrainedNN, make_backbone
from experiment_data import derive_soil_water_features


ROOT = Path(__file__).resolve().parents[1]


def load_demo_batch(count=4):
    cache = np.load(ROOT / "data/demo_synthetic.npz", allow_pickle=False)
    sequence = cache["sequence"][:count]
    static = cache["static"][:count]
    metadata = json.loads(str(cache["metadata"]))
    capacity, initial = derive_soil_water_features(static, metadata)
    return (
        torch.as_tensor(sequence),
        torch.as_tensor(static),
        torch.as_tensor(cache["planting_day"][:count]),
        torch.as_tensor(capacity),
        torch.as_tensor(initial),
    )


def make_demo_model(sequence, static):
    sequence_mean = sequence.mean((0, 1))
    sequence_std = sequence.std((0, 1)).clamp_min(1e-3)
    static_mean = static.mean(0)
    static_std = static.std(0).clamp_min(1e-3)
    backbone = make_backbone(
        "dnn", sequence.shape[-1], static.shape[-1], 16,
        sequence_mean, sequence_std, static_mean, static_std,
    )
    return ProcessConstrainedNN(
        backbone, 16, 90, 300,
        use_nutrients=True,
        fixed_yield_coefficient=0.42305775,
        yield_mean=5500.0,
        yield_std=1000.0,
    )
