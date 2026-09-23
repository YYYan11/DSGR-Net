"""Reproducible baselines and a hard-recurrent differentiable cotton model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiment_data import (
    PAPER_FIELD_PROTOCOL,
    SEQUENCE_FEATURES,
    build_2024_field_validation_cache,
    build_cache,
    derive_soil_water_features,
    group_split,
    read_2024_sensor_soil_moisture,
)


PROCESS_MODEL_VERSION = "cotton_process_v13_strong_env_rich_process_multiscale_cnn"
SUPPORTED_PROCESS_MODEL_VERSIONS = {
    PROCESS_MODEL_VERSION,
}
# 初始建模假设：这些量是可审计的单位/范围，不是已完成的品种参数校准。
RAIN_INFILTRATION_EFFICIENCY = 1.0  # 2022 SWXD/ETAA 校准；无量纲
DRAINAGE_SMOOTHING = 10.0  # 1/mm；softplus 平滑排水边界
DRAINAGE_FRACTION = 0.05  # 超田间持水量部分的日排水比例
IRRIGATION_EFFICIENCY = 0.8915783407037171  # 2022 SWXD/ETAA 校准；无量纲
ET_SCALE = 0.9586557067798288  # 2022 SWXD/ETAA 校准；无量纲
WATER_STRESS_THRESHOLD = 0.6579627613886836  # 固定的生长胁迫先验
WATER_STRESS_SLOPE = 2.932634170384998  # 固定的 sigmoid 斜率
PAR_FRACTION = 0.48  # PAR/SRAD，无量纲；待用文献或 DSSAT 输出校准
LIGHT_EXTINCTION_COEFFICIENT = 0.65  # Beer-Lambert 消光系数；当前棉花初始假设
BASE_RUE_G_MJ = 2.634  # g dry matter / MJ APAR；由 2022 DSSAT 教师轨迹拟合，非论文 GLUE 品种系数
RUE_TO_KG_HA = 10.0  # 1 g/m2 = 10 kg/ha
MAX_LAI = 3.787  # m2 leaf / m2 ground；由 2022 DSSAT 教师轨迹拟合，待按论文的 2023 验证轨迹复核
RADIATION_TO_WATER = 0.408  # mm / (MJ m-2)，能量到等效水深换算
NET_RADIATION_FRACTION = 0.65  # 用 SRAD 构造 ET0 代理的初始净辐射比例
ET_CORRECTION_LIMIT = 0.2  # DNN 对 Kc/ET 需求的最大相对修正
RUE_CORRECTION_LIMIT = 0.2  # DNN 对 RUE 的最大相对修正
BASE_TEMPERATURE_C = 10.0
OPTIMUM_TEMPERATURE_C = 28.0
MAXIMUM_TEMPERATURE_C = 40.0
BIOMASS_STATE_SCALE_KG_HA = 15000.0
MATURITY_GATE_WIDTH_GDD = 30.0
GROWTH_PROPOSAL_SCALE_KG_HA_DAY = 100.0
GROWTH_RESIDUAL_SCALE_KG_HA_DAY = 100.0

# V7 strong-environment-DNN + rich multi-scale temporal-process-CNN normalization.
# These are fixed engineering scales (not trainable crop parameters), chosen only
# to keep the four CNN channels on comparable numerical ranges.
PROCESS_ET_SCALE_MM_DAY = 10.0
PROCESS_GROWTH_SCALE_KG_HA_DAY = 200.0
PROCESS_LAI_SCALE = 6.0


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_water_calibration(path: str | Path | None) -> dict:
    """Load frozen water parameters selected on 2022 and checked on 2023."""
    defaults = {
        "rain_efficiency": RAIN_INFILTRATION_EFFICIENCY,
        "irrigation_efficiency": IRRIGATION_EFFICIENCY,
        "et_scale": ET_SCALE,
        "drainage_fraction": DRAINAGE_FRACTION,
        "water_stress_threshold": WATER_STRESS_THRESHOLD,
        "water_stress_slope": WATER_STRESS_SLOPE,
        "source": "built_in_2022_water_calibration",
        "sha256": None,
        "process_parameters": {"et_model": "radiation", "lai_model": "sine"},
    }
    if not path:
        return defaults
    calibration_path = Path(path)
    raw = calibration_path.read_bytes()
    document = json.loads(raw)
    if document.get("method") != "water_only_differentiable_process_calibration":
        raise ValueError("water calibration was not produced by the water-only calibration workflow")
    if not document.get("validation", {}).get("metrics"):
        raise ValueError("water calibration has no independent-year validation metrics")
    params = document["best_params"]
    normalized_params = dict(params)
    normalized_params.setdefault("et_scale", 1.0)
    normalized_params.setdefault("et_model", document.get("et_model", "radiation"))
    normalized_params.setdefault("lai_model", document.get("lai_model", "sine"))
    result = {
        "rain_efficiency": float(params["rain_efficiency"]),
        "irrigation_efficiency": float(params["irrigation_efficiency"]),
        "et_scale": float(normalized_params["et_scale"]),
        "drainage_fraction": float(params["drainage_fraction"]),
        "water_stress_threshold": float(params["stress_threshold"]),
        "water_stress_slope": float(params["stress_slope"]),
        "source": str(calibration_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "process_parameters": normalized_params,
    }
    return result


def _chunk_statistics(values: np.ndarray, indices: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    total = None
    square = None
    count = 0
    for start in range(0, len(indices), chunk):
        block = values[indices[start : start + chunk]].astype(np.float64, copy=False)
        axes = tuple(range(block.ndim - 1))
        block_sum = block.sum(axis=axes)
        block_square = np.square(block).sum(axis=axes)
        total = block_sum if total is None else total + block_sum
        square = block_square if square is None else square + block_square
        count += int(np.prod(block.shape[:-1]))
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(inputs, mean, std):
    """Standardize while neutralizing features constant in the training split.

    A feature with no training variation cannot support an estimated response.
    Dividing an external value by the numerical 1e-6 floor would create a large,
    arbitrary extrapolation, so its standardized value is fixed at zero.
    """
    variable = std > 1e-5
    safe_std = torch.where(variable, std, torch.ones_like(std))
    normalized = (inputs - mean) / safe_std
    return torch.where(variable, normalized, torch.zeros_like(normalized)).clamp(-8.0, 8.0)


class ArrayDataset(Dataset):
    def __init__(
        self,
        sequence,
        static,
        labels,
        planting_day,
        soil_capacity,
        initial_soil_water,
        indices,
    ):
        self.sequence = sequence
        self.static = static
        self.labels = labels
        self.planting_day = planting_day
        self.soil_capacity = soil_capacity
        self.initial_soil_water = initial_soil_water
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return (
            torch.from_numpy(self.sequence[index]),
            torch.from_numpy(self.static[index]),
            torch.tensor(self.labels[index]),
            torch.tensor(self.planting_day[index], dtype=torch.long),
            torch.tensor(self.soil_capacity[index]),
            torch.tensor(self.initial_soil_water[index]),
        )


class TemporalBackbone(nn.Module):
    def __init__(self, sequence_dim, static_dim, hidden, sequence_mean, sequence_std, static_mean, static_std):
        super().__init__()
        self.register_buffer("sequence_mean", torch.as_tensor(sequence_mean))
        self.register_buffer("sequence_std", torch.as_tensor(sequence_std))
        self.register_buffer("static_mean", torch.as_tensor(static_mean))
        self.register_buffer("static_std", torch.as_tensor(static_std))
        self.sequence_net = nn.Sequential(
            nn.Conv1d(sequence_dim, hidden, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.SiLU(),
        )
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, sequence, static):
        sequence = _standardize(sequence, self.sequence_mean, self.sequence_std)
        static = _standardize(static, self.static_mean, self.static_std)
        temporal = self.sequence_net(sequence.transpose(1, 2)).transpose(1, 2)
        context = self.static_net(static)
        return temporal, context


class DNNBackbone(nn.Module):
    """逐日共享的全连接特征提取器，用于 DNN→过程参数/修正量。"""

    def __init__(self, sequence_dim, static_dim, hidden, sequence_mean, sequence_std, static_mean, static_std):
        super().__init__()
        self.register_buffer("sequence_mean", torch.as_tensor(sequence_mean))
        self.register_buffer("sequence_std", torch.as_tensor(sequence_std))
        self.register_buffer("static_mean", torch.as_tensor(static_mean))
        self.register_buffer("static_std", torch.as_tensor(static_std))
        self.sequence_net = nn.Sequential(
            nn.Linear(sequence_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )

    def forward(self, sequence, static):
        sequence = _standardize(sequence, self.sequence_mean, self.sequence_std)
        static = _standardize(static, self.static_mean, self.static_std)
        return self.sequence_net(sequence), self.static_net(static)


def make_backbone(kind, sequence_dim, static_dim, hidden, sequence_mean, sequence_std, static_mean, static_std):
    backbone_class = DNNBackbone if kind == "dnn" else TemporalBackbone
    return backbone_class(
        sequence_dim, static_dim, hidden,
        sequence_mean, sequence_std, static_mean, static_std,
    )


class MultiScaleTemporalProcessEncoder(nn.Module):
    """Multi-scale 1D CNN over differentiable daily process states.

    Three parallel temporal kernels capture short-, medium-, and longer-range
    local response patterns while preserving the full T-step sequence length.
    Each branch emits 32 channels; the concatenated 96-channel tensor is
    refined back to 64 channels. This keeps the process-encoder parameter count
    close to V6 while changing primarily the temporal receptive-field design.
    """

    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.branch_k9 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.SiLU(),
        )
        # Shared refinement after multi-scale concatenation. Padding keeps the
        # temporal length unchanged, so output is always [B, 64, T].
        self.refine = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        multi_scale = torch.cat(
            (self.branch_k3(x), self.branch_k5(x), self.branch_k9(x)),
            dim=1,
        )
        return self.refine(multi_scale)


class PlainNN(nn.Module):
    def __init__(self, backbone, hidden, yield_mean, yield_std, start_day, end_day):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("yield_mean", torch.tensor(float(yield_mean)))
        self.register_buffer("yield_std", torch.tensor(float(yield_std)))
        self.register_buffer("days", torch.arange(start_day, end_day + 1))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, sequence, static, planting_day, soil_capacity=None, initial_soil_water=None):
        temporal, context = self.backbone(sequence, static)
        mask = (self.days[None, :] >= planting_day[:, None]).to(temporal.dtype)
        pooled = (temporal * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        standardized = self.head(torch.cat((pooled, context), dim=-1)).squeeze(-1)
        return {"yield": standardized * self.yield_std + self.yield_mean}


class WLDNNBaseline(nn.Module):
    """Wang's original black-box DNN architecture on its 1997-D feature vector."""

    def __init__(
        self,
        feature_dim,
        feature_mean,
        feature_std,
        yield_mean,
        yield_std,
    ):
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean))
        self.register_buffer("feature_std", torch.as_tensor(feature_std))
        self.register_buffer("yield_mean", torch.tensor(float(yield_mean)))
        self.register_buffer("yield_std", torch.tensor(float(yield_std)))
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, features, static, planting_day, soil_capacity=None, initial_soil_water=None):
        features = _standardize(features, self.feature_mean, self.feature_std)
        standardized = self.net(features).squeeze(-1)
        return {"yield": standardized * self.yield_std + self.yield_mean}


class ProcessConstrainedNN(nn.Module):
    """DSSAT/CROPGRO-inspired differentiable daily crop-water recurrence.

    The DNN environment branch produces bounded ET/RUE corrections and
    environment representations. The explicit recurrence produces daily
    soil-water, ET, water-stress, and biomass-growth states. V4-smoke encodes
    those daily states with a multi-scale 1D CNN before fusion for yield prediction.
    """

    def __init__(
        self,
        backbone,
        hidden,
        start_day,
        end_day,
        maturity_gdd=2600.0,
        rain_efficiency=RAIN_INFILTRATION_EFFICIENCY,
        irrigation_efficiency=IRRIGATION_EFFICIENCY,
        et_scale=ET_SCALE,
        drainage_fraction=DRAINAGE_FRACTION,
        water_stress_threshold=WATER_STRESS_THRESHOLD,
        water_stress_slope=WATER_STRESS_SLOPE,
        process_mode="hard",
        use_nutrients=False,
        fixed_yield_coefficient=None,
        process_parameters=None,
        process_model_version=PROCESS_MODEL_VERSION,
        et_correction_limit=ET_CORRECTION_LIMIT,
        rue_correction_limit=RUE_CORRECTION_LIMIT,
        correction_uses_irrigation=True,
        disable_growth_water_stress=False,
        disable_late_growth_water_stress=False,
        yield_mean=0.0,
        yield_std=1.0,
    ):
        super().__init__()
        if maturity_gdd <= 0:
            raise ValueError("maturity_gdd must be positive")
        if process_mode not in {"hard", "agripinn"}:
            raise ValueError("process_mode must be 'hard' or 'agripinn'")
        self.backbone = backbone
        self.maturity_gdd = float(maturity_gdd)
        params = dict(process_parameters or {})
        self.process_model_version = process_model_version
        self.rain_efficiency = float(params.get("rain_efficiency", rain_efficiency))
        self.irrigation_efficiency = float(params.get("irrigation_efficiency", irrigation_efficiency))
        self.et_scale = float(params.get("et_scale", et_scale))
        self.drainage_fraction = float(params.get("drainage_fraction", drainage_fraction))
        self.water_stress_threshold = float(params.get(
            "stress_threshold", params.get("water_stress_threshold", water_stress_threshold)
        ))
        self.water_stress_slope = float(params.get(
            "stress_slope", params.get("water_stress_slope", water_stress_slope)
        ))
        self.et_model = str(params.get("et_model", "radiation"))
        self.lai_model = str(params.get("lai_model", "sine"))
        self.surface_memory = float(params.get("surface_memory", 0.92))
        self.surface_wetting_scale_mm = float(params.get("surface_wetting_scale_mm", 24.5))
        self.lai_asymptote = float(params.get("lai_asymptote", 5.885967456970629))
        self.lai_growth_midpoint_gdd = float(params.get("lai_growth_midpoint_gdd", 1039.7977587408282))
        self.lai_growth_width_gdd = float(params.get("lai_growth_width_gdd", 167.49811192234083))
        self.lai_decline_midpoint_gdd = float(params.get("lai_decline_midpoint_gdd", 2284.342738225305))
        self.lai_decline_width_gdd = float(params.get("lai_decline_width_gdd", 499.7057243055772))
        self.potential_transpiration_scale = float(params.get("potential_transpiration_scale", 0.8363597558664126))
        self.transpiration_lai_extinction = float(params.get("transpiration_lai_extinction", 0.8941386340468911))
        self.potential_soil_scale = float(params.get("potential_soil_scale", 0.9741626699053171))
        self.soil_lai_extinction = float(params.get("soil_lai_extinction", 0.4096330275229358))
        self.transpiration_stress_threshold = float(params.get("transpiration_stress_threshold", 0.135))
        self.transpiration_stress_slope = float(params.get("transpiration_stress_slope", 11.25))
        self.et_correction_limit = float(et_correction_limit)
        self.rue_correction_limit = float(rue_correction_limit)
        self.correction_uses_irrigation = bool(correction_uses_irrigation)
        self.disable_growth_water_stress = bool(disable_growth_water_stress)
        self.disable_late_growth_water_stress = bool(disable_late_growth_water_stress)
        self.process_parameters = params
        if not 0.0 <= self.rain_efficiency <= 1.0:
            raise ValueError("rain_efficiency must be in [0, 1]")
        if not 0.0 <= self.irrigation_efficiency <= 1.0:
            raise ValueError("irrigation_efficiency must be in [0, 1]")
        if self.et_scale <= 0.0 or not 0.0 <= self.drainage_fraction <= 1.0:
            raise ValueError("et_scale must be positive and drainage_fraction must be in [0, 1]")
        if self.et_model not in {"radiation", "dssat_components_radiation"}:
            raise ValueError(f"unsupported PyTorch ET model: {self.et_model}")
        if not 0.0 <= self.et_correction_limit <= 1.0:
            raise ValueError("et_correction_limit must be in [0, 1]")
        if not 0.0 <= self.rue_correction_limit <= 1.0:
            raise ValueError("rue_correction_limit must be in [0, 1]")
        self.process_mode = process_mode
        self.use_nutrients = bool(use_nutrients)
        self.fixed_yield_coefficient = (
            None if fixed_yield_coefficient is None else float(fixed_yield_coefficient)
        )
        self.nutrient_indices = tuple(
            SEQUENCE_FEATURES.index(name) for name in ("N", "P", "K")
        )
        self.register_buffer("days", torch.arange(start_day, end_day + 1))
        self.process_parameter_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

        # -------------------------------------------------
        # V5 neural-process fusion yield head
        #
        # Relative to V4, ONLY the yield-side environment representation changes:
        #   V4: 4 stage-pooled small-DNN temporal features + static context
        #   V5: strong flattened environment DNN -> h_env
        #
        # The environment branch is unchanged; V6 enriches only the process-state channels supplied to the 1D CNN.
        # The original small DNN backbone is also retained for the bounded
        # daily ET/RUE correction heads inside the crop-water process.
        #
        # Raw irrigation is set to the training mean before the environment DNN,
        # so its standardized value is exactly zero. Irrigation can affect yield
        # only through the explicit differentiable crop-water recurrence.
        # -------------------------------------------------
        self.register_buffer(
            "yield_mean",
            torch.tensor(float(yield_mean))
        )
        self.register_buffer(
            "yield_std",
            torch.tensor(float(yield_std))
        )

        # Strong environment DNN. The input is the standardized fixed-window
        # sequence (raw IRR masked to its train mean) plus standardized static
        # features. The active crop-period gate is applied before flattening.
        sequence_dim = int(self.backbone.sequence_mean.numel())
        static_dim = int(self.backbone.static_mean.numel())
        time_steps = int(end_day - start_day + 1)
        environment_input_dim = time_steps * sequence_dim + static_dim
        self.environment_encoder = nn.Sequential(
            nn.Linear(environment_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, hidden),
            nn.SiLU(),
        )

        # V7 multi-scale temporal encoder over the eight DAILY PROCESS STATE
        # channels. Parallel kernels (3, 5, 9 days) capture short-, medium-,
        # and longer-range local crop-water response patterns. Raw irrigation
        # remains excluded from the CNN input; it can affect yield only via the
        # explicit differentiable process states.
        # Input:  [B, 8, T]
        # Output: [B, 64, T]
        self.process_cnn = MultiScaleTemporalProcessEncoder(in_channels=8)

        # CNN features are pooled AFTER convolution within the same four
        # normalized seasonal-progress bins used by the V3 DNN branch.
        # 4 bins * 64 channels = 256 -> hidden-dimensional process embedding.
        self.process_projection = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, hidden),
            nn.SiLU(),
        )

        fusion_hidden = max(64, hidden * 2)

        # Strong environment embedding: hidden
        # CNN process embedding:       hidden
        # Total fusion dimension:      2 * hidden
        self.yield_head = nn.Sequential(
            nn.Linear(hidden * 2, fusion_hidden),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(fusion_hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.daily_head = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden), nn.SiLU(), nn.Linear(hidden, 3)
        )
        nn.init.zeros_(self.process_parameter_head[-1].weight)
        nn.init.zeros_(self.process_parameter_head[-1].bias)
        nn.init.zeros_(self.daily_head[-1].weight)
        nn.init.zeros_(self.daily_head[-1].bias)
        with torch.no_grad():
            # 仅初始化有效产量转换系数；水分参数由 2022 SWXD/ETAA
            # 校准后冻结，避免季末产量监督重新改变水分状态的含义。
            self.process_parameter_head[-1].bias.fill_(-1.12)
            # 约 47 kg/ha/day 的正初值，避免 softplus 生长头从近零产量起步。
            self.daily_head[-1].bias[2] = -0.5

    def forward(self, sequence, static, planting_day, soil_capacity, initial_soil_water):
        # N/P/K 可通过 DNN 输出的有界 ET/RUE 修正量影响可微过程；
        # 最终产量由环境 DNN 表征与 CNN 过程表征融合预测。
        # 默认关闭养分影响以保留纯水分过程对照，正式路线用 --use-nutrients 打开。
        process_sequence = sequence
        if not self.use_nutrients or not self.correction_uses_irrigation:
            process_sequence = sequence.clone()
        if not self.use_nutrients:
            process_sequence[..., self.nutrient_indices] = self.backbone.sequence_mean[
                list(self.nutrient_indices)
            ]
        if not self.correction_uses_irrigation:
            irrigation_index = SEQUENCE_FEATURES.index("IRR")
            process_sequence[..., irrigation_index] = self.backbone.sequence_mean[irrigation_index]
        temporal, context = self.backbone(process_sequence, static)
        process_parameters = self.process_parameter_head(context)
        capacity = soil_capacity.to(sequence.dtype).clamp_min(1.0)
        initial_water = initial_soil_water.to(sequence.dtype).clamp(0.0).minimum(capacity)
        water_threshold = torch.full_like(initial_water, self.water_stress_threshold)
        irrigation_efficiency = torch.full_like(initial_water, self.irrigation_efficiency)
        if self.fixed_yield_coefficient is None:
            yield_coefficient = 0.3 + 0.5 * torch.sigmoid(process_parameters[:, 0])
        else:
            yield_coefficient = torch.full_like(initial_water, self.fixed_yield_coefficient)

        soil_water = []
        drainage = []
        water_stress = []
        growth_water_stress_daily = []
        actual_et = []
        soil_evaporation = []
        transpiration = []
        transpiration_stress = []
        surface_wetness = []
        lai = []
        stage = []
        growth = []
        growth_process = []
        growth_proposal = []
        growth_residual = []
        et_correction = []
        rue_correction = []
        water_closure = []
        active = []
        current_water = initial_water
        current_biomass = torch.zeros_like(initial_water)
        current_surface_wetness = torch.ones_like(initial_water)
        thermal_time = torch.zeros_like(initial_water)
        radiation_index = SEQUENCE_FEATURES.index("SRAD")
        tmax_index = SEQUENCE_FEATURES.index("TMAX")
        tmin_index = SEQUENCE_FEATURES.index("TMIN")
        rain_index = SEQUENCE_FEATURES.index("RAIN")
        irrigation_index = SEQUENCE_FEATURES.index("IRR")

        for day_index in range(sequence.shape[1]):
            planted = (self.days[day_index] >= planting_day).to(sequence.dtype)
            tmax = sequence[:, day_index, tmax_index]
            tmin = sequence[:, day_index, tmin_index]
            tmean = 0.5 * (tmax + tmin)
            thermal_time = thermal_time + torch.relu(tmean - BASE_TEMPERATURE_C) * planted
            daily_stage = (thermal_time / self.maturity_gdd).clamp(0.0, 1.0)
            maturity_gate = torch.sigmoid(
                (self.maturity_gdd - thermal_time) / MATURITY_GATE_WIDTH_GDD
            ) * planted

            state_features = torch.stack((
                current_water / capacity,
                current_biomass / BIOMASS_STATE_SCALE_KG_HA,
                daily_stage,
            ), dim=1)
            daily_raw = self.daily_head(torch.cat((
                temporal[:, day_index], context, state_features,
            ), dim=1))
            daily_et_correction = self.et_correction_limit * torch.tanh(daily_raw[:, 0])
            daily_rue_correction = self.rue_correction_limit * torch.tanh(daily_raw[:, 1])
            daily_growth_proposal = (
                GROWTH_PROPOSAL_SCALE_KG_HA_DAY
                * torch.nn.functional.softplus(daily_raw[:, 2])
                * maturity_gate
            )

            rain = sequence[:, day_index, rain_index].clamp_min(0.0)
            irrigation = sequence[:, day_index, irrigation_index].clamp_min(0.0)
            water_before_fluxes = (
                current_water
                + self.rain_efficiency * rain
                + irrigation_efficiency * irrigation
            )
            stress = torch.sigmoid(
                self.water_stress_slope
                * (water_before_fluxes / capacity - water_threshold)
            )
            growth_water_stress = (
                torch.ones_like(stress) if self.disable_growth_water_stress else stress
            )
            if self.disable_late_growth_water_stress:
                growth_water_stress = torch.where(
                    daily_stage >= 0.75, torch.ones_like(stress), growth_water_stress
                )

            canopy_shape = torch.sin(torch.pi * daily_stage).clamp_min(0.0)
            if self.lai_model == "double_sigmoid":
                daily_lai = (
                    self.lai_asymptote
                    * torch.sigmoid(
                        (thermal_time - self.lai_growth_midpoint_gdd)
                        / self.lai_growth_width_gdd
                    )
                    * torch.sigmoid(
                        (self.lai_decline_midpoint_gdd - thermal_time)
                        / self.lai_decline_width_gdd
                    )
                    * planted
                )
            else:
                daily_lai = MAX_LAI * canopy_shape.pow(1.2) * planted
            et0_proxy = (
                RADIATION_TO_WATER
                * NET_RADIATION_FRACTION
                * sequence[:, day_index, radiation_index].clamp_min(0.0)
            )
            if self.et_model == "dssat_components_radiation":
                effective_input = (
                    self.rain_efficiency * rain + irrigation_efficiency * irrigation
                )
                event_wetness = 1.0 - torch.exp(
                    -effective_input / self.surface_wetting_scale_mm
                )
                current_surface_wetness = 1.0 - (
                    1.0 - self.surface_memory * current_surface_wetness
                ) * (1.0 - event_wetness)
                potential_soil_evaporation = (
                    et0_proxy * self.potential_soil_scale
                    * torch.exp(-self.soil_lai_extinction * daily_lai)
                )
                potential_transpiration = (
                    et0_proxy * self.potential_transpiration_scale
                    * (1.0 - torch.exp(-self.transpiration_lai_extinction * daily_lai))
                )
                daily_transpiration_stress = torch.sigmoid(
                    self.transpiration_stress_slope
                    * (water_before_fluxes / capacity - self.transpiration_stress_threshold)
                )
                daily_soil_evaporation_demand = (
                    potential_soil_evaporation * current_surface_wetness
                )
                daily_transpiration_demand = (
                    potential_transpiration * daily_transpiration_stress * maturity_gate
                )
                et_multiplier = 1.0 + daily_et_correction
                daily_soil_evaporation_demand = daily_soil_evaporation_demand * et_multiplier
                daily_transpiration_demand = daily_transpiration_demand * et_multiplier
                et_demand = daily_soil_evaporation_demand + daily_transpiration_demand
            else:
                # ETAA includes bare-soil evaporation outside the active crop
                # period, so ET demand is not multiplied by the maturity gate.
                crop_coefficient = 0.3 + 0.9 * canopy_shape * maturity_gate
                et_demand = (
                    et0_proxy
                    * crop_coefficient
                    * self.et_scale
                    * (1.0 + daily_et_correction)
                    * (0.3 + 0.7 * stress)
                )
                daily_transpiration_stress = stress
                daily_soil_evaporation_demand = torch.zeros_like(et_demand)
                daily_transpiration_demand = et_demand
            daily_actual_et = torch.minimum(et_demand, water_before_fluxes.clamp_min(0.0))
            et_flux_fraction = daily_actual_et / et_demand.clamp_min(1e-8)
            daily_soil_evaporation = daily_soil_evaporation_demand * et_flux_fraction
            daily_transpiration = daily_transpiration_demand * et_flux_fraction
            if self.et_model == "dssat_components_radiation":
                current_surface_wetness = current_surface_wetness * torch.exp(
                    -daily_soil_evaporation / self.surface_wetting_scale_mm
                )
            water_after_et = water_before_fluxes - daily_actual_et
            daily_drainage = torch.nn.functional.softplus(
                DRAINAGE_SMOOTHING * (water_after_et - capacity)
            ) / DRAINAGE_SMOOTHING * self.drainage_fraction
            next_water = water_after_et - daily_drainage

            rising_temperature = (
                (tmean - BASE_TEMPERATURE_C)
                / (OPTIMUM_TEMPERATURE_C - BASE_TEMPERATURE_C)
            ).clamp(0.0, 1.0)
            falling_temperature = (
                (MAXIMUM_TEMPERATURE_C - tmean)
                / (MAXIMUM_TEMPERATURE_C - OPTIMUM_TEMPERATURE_C)
            ).clamp(0.0, 1.0)
            temperature_factor = torch.minimum(rising_temperature, falling_temperature)
            par = PAR_FRACTION * sequence[:, day_index, radiation_index].clamp_min(0.0)
            apar = par * (
                1.0 - torch.exp(-LIGHT_EXTINCTION_COEFFICIENT * daily_lai)
            )
            daily_rue = BASE_RUE_G_MJ * (1.0 + daily_rue_correction)
            daily_growth = (
                daily_rue
                * apar
                * growth_water_stress
                * temperature_factor
                * RUE_TO_KG_HA
                * maturity_gate
            )
            if self.process_mode == "agripinn":
                daily_growth_used = daily_growth_proposal
            else:
                daily_growth_used = daily_growth
            next_biomass = current_biomass + daily_growth_used
            closure_error = next_water - (
                current_water
                + self.rain_efficiency * rain
                + irrigation_efficiency * irrigation
                - daily_actual_et
                - daily_drainage
            )

            soil_water.append(next_water)
            drainage.append(daily_drainage)
            water_stress.append(stress)
            growth_water_stress_daily.append(growth_water_stress)
            actual_et.append(daily_actual_et)
            soil_evaporation.append(daily_soil_evaporation)
            transpiration.append(daily_transpiration)
            transpiration_stress.append(daily_transpiration_stress)
            surface_wetness.append(current_surface_wetness)
            lai.append(daily_lai)
            stage.append(daily_stage)
            growth.append(daily_growth_used)
            growth_process.append(daily_growth)
            growth_proposal.append(daily_growth_proposal)
            growth_residual.append(
                (daily_growth_proposal - daily_growth) / GROWTH_RESIDUAL_SCALE_KG_HA_DAY
            )
            et_correction.append(daily_et_correction)
            rue_correction.append(daily_rue_correction)
            water_closure.append(closure_error)
            active.append(maturity_gate)
            current_water = next_water
            current_biomass = next_biomass

        soil_water = torch.stack(soil_water, dim=1)
        drainage = torch.stack(drainage, dim=1)
        water_stress = torch.stack(water_stress, dim=1)
        actual_et = torch.stack(actual_et, dim=1)
        soil_evaporation = torch.stack(soil_evaporation, dim=1)
        transpiration = torch.stack(transpiration, dim=1)
        transpiration_stress = torch.stack(transpiration_stress, dim=1)
        surface_wetness = torch.stack(surface_wetness, dim=1)
        lai = torch.stack(lai, dim=1)
        stage = torch.stack(stage, dim=1)
        growth = torch.stack(growth, dim=1)
        growth_process = torch.stack(growth_process, dim=1)
        growth_proposal = torch.stack(growth_proposal, dim=1)
        growth_residual = torch.stack(growth_residual, dim=1)
        et_correction = torch.stack(et_correction, dim=1)
        rue_correction = torch.stack(rue_correction, dim=1)
        water_closure = torch.stack(water_closure, dim=1)
        active = torch.stack(active, dim=1)
        biomass = torch.cumsum(growth, dim=1)

        # =================================================
        # Differentiable process representation
        # =================================================
        fw_tensor = torch.stack(
            growth_water_stress_daily, dim=1
        )

        active_count = active.sum(dim=1).clamp_min(1.0)

        # DNN representation.
        # With --exclude-irrigation-from-corrections,
        # temporal does NOT contain raw irrigation.
        # Stage-aware neural temporal representation
        neural_stage_features = []

        neural_stage_ranges = (
            (0.00, 0.25),
            (0.25, 0.50),
            (0.50, 0.75),
            (0.75, 1.01),
        )

        for lo, hi in neural_stage_ranges:
            stage_mask = (
                active
                * ((stage >= lo) & (stage < hi)).to(active.dtype)
            )

            count = stage_mask.sum(dim=1).clamp_min(1.0)

            stage_feature = (
                (
                    temporal
                    * stage_mask[:, :, None]
                ).sum(dim=1)
                / count[:, None]
            )

            neural_stage_features.append(stage_feature)

        pooled = torch.cat(
            neural_stage_features,
            dim=1,
        )

        # =================================================
        # V7: richer differentiable DAILY process-state representation
        #
        # z_t = [SW_rel, ET, F_W, ΔB, LAI, transpiration,
        #        soil evaporation, surface wetness]
        # shape before CNN: [B, T, 8]
        # Conv1d input:      [B, 8, T]
        #
        # Important:
        # - raw irrigation is NOT a CNN input channel;
        # - every channel is generated by the explicit differentiable
        #   crop-water recurrence, preserving the modeled pathway
        #   I_t -> process states -> CNN -> yield.
        # - inactive/non-growing-period states are gated before convolution and
        #   excluded again during stage-aware pooling.
        # =================================================
        relative_sw_tensor = (
            soil_water
            / capacity[:, None].clamp_min(1.0e-6)
        )

        normalized_et_tensor = actual_et / PROCESS_ET_SCALE_MM_DAY
        normalized_growth_tensor = growth / PROCESS_GROWTH_SCALE_KG_HA_DAY
        normalized_lai_tensor = lai / PROCESS_LAI_SCALE
        normalized_transpiration_tensor = transpiration / PROCESS_ET_SCALE_MM_DAY
        normalized_soil_evap_tensor = soil_evaporation / PROCESS_ET_SCALE_MM_DAY

        process_state_sequence = torch.stack(
            (
                relative_sw_tensor,
                normalized_et_tensor,
                fw_tensor,
                normalized_growth_tensor,
                normalized_lai_tensor,
                normalized_transpiration_tensor,
                normalized_soil_evap_tensor,
                surface_wetness,
            ),
            dim=-1,
        )  # [B, T, 8]

        # Soft active gate is already part of the differentiable crop model.
        process_state_sequence = process_state_sequence * active[:, :, None]

        # Conv1d operates over time: [B, channels=8, T].
        process_temporal = self.process_cnn(
            process_state_sequence.transpose(1, 2)
        ).transpose(1, 2)  # [B, T, 64]

        process_stage_features = []
        process_stage_ranges = (
            (0.00, 0.25),
            (0.25, 0.50),
            (0.50, 0.75),
            (0.75, 1.01),
        )

        for lo, hi in process_stage_ranges:
            stage_mask = (
                active
                * ((stage >= lo) & (stage < hi)).to(active.dtype)
            )
            count = stage_mask.sum(dim=1).clamp_min(1.0)

            # Pool learned local temporal patterns, NOT raw process variables.
            stage_feature = (
                (process_temporal * stage_mask[:, :, None]).sum(dim=1)
                / count[:, None]
            )
            process_stage_features.append(stage_feature)

        process_pooled = torch.cat(
            process_stage_features,
            dim=1,
        )  # [B, 4 * 64] = [B, 256]

        process_embedding = self.process_projection(
            process_pooled
        )  # [B, hidden]

        # =================================================
        # V7 strong environment DNN + multi-scale differentiable process CNN -> yield
        # =================================================
        # Standardize the same neural-side sequence used by the bounded
        # correction network. With --exclude-irrigation-from-corrections, raw
        # irrigation has already been replaced by the training mean and therefore
        # becomes zero after standardization.
        environment_sequence = _standardize(
            process_sequence,
            self.backbone.sequence_mean,
            self.backbone.sequence_std,
        )
        environment_static = _standardize(
            static,
            self.backbone.static_mean,
            self.backbone.static_std,
        )

        # Restrict the strong DNN to the modelled active crop period so the
        # fixed DOY 90-300 window is not interpreted as a 211-day growing season.
        environment_sequence = environment_sequence * active[:, :, None]
        environment_input = torch.cat(
            (environment_sequence.flatten(start_dim=1), environment_static),
            dim=1,
        )
        environment_embedding = self.environment_encoder(environment_input)

        fusion = torch.cat(
            (
                environment_embedding,
                process_embedding,
            ),
            dim=1,
        )

        standardized_yield = (
            self.yield_head(fusion).squeeze(-1)
        )

        predicted_yield = (
            standardized_yield * self.yield_std
            + self.yield_mean
        )
        return {
            "yield": predicted_yield,
            "biomass": biomass,
            "soil_water": soil_water,
            "drainage": drainage,
            "actual_et": actual_et,
            "soil_evaporation": soil_evaporation,
            "transpiration": transpiration,
            "transpiration_stress": transpiration_stress,
            "surface_wetness": surface_wetness,
            "lai": lai,
            "stage": stage,
            "water_stress": water_stress,
            "growth_water_stress": fw_tensor,
            "process_embedding": process_embedding,
            "process_temporal": process_temporal,
            "process_state_sequence": process_state_sequence,
            "environment_embedding": environment_embedding,
            "neural_pooled": pooled,
            "growth": growth,
            "growth_process": growth_process,
            "growth_proposal": growth_proposal,
            "growth_residual": growth_residual,
            "et_correction": et_correction,
            "rue_correction": rue_correction,
            "water_closure_error": water_closure,
            "active": active,
            "yield_coefficient": yield_coefficient,
            "capacity": capacity,
            "water_threshold": water_threshold,
            "irrigation_efficiency": irrigation_efficiency,
        }


def _metrics(y_true, y_pred):
    error = y_pred - y_true
    mse = np.mean(np.square(error))
    denominator = np.sum(np.square(y_true - np.mean(y_true)))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(np.square(error)) / denominator) if denominator > 0 else None,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truth = []
    prediction = []
    correction_square = 0.0
    correction_count = 0.0
    closure_square = 0.0
    closure_count = 0.0
    for sequence, static, labels, planting_day, soil_capacity, initial_soil_water in loader:
        output = model(
            sequence.to(device),
            static.to(device),
            planting_day.to(device),
            soil_capacity.to(device),
            initial_soil_water.to(device),
        )
        truth.append(labels.numpy())
        prediction.append(output["yield"].cpu().numpy())
        if "et_correction" in output:
            active = output["active"]
            correction_square += float((
                (output["et_correction"].square() + output["rue_correction"].square())
                * active
            ).sum())
            correction_count += float(2.0 * active.sum())
        if "water_closure_error" in output:
            closure_square += float(output["water_closure_error"].square().sum())
            closure_count += float(output["water_closure_error"].numel())
    metrics = _metrics(np.concatenate(truth), np.concatenate(prediction))
    if correction_count:
        metrics["correction_rms"] = float(np.sqrt(correction_square / correction_count))
    if closure_count:
        metrics["water_closure_rmse_mm"] = float(np.sqrt(closure_square / closure_count))
    return metrics


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    predictions = []
    correction_square = 0.0
    correction_count = 0.0
    closure_square = 0.0
    closure_count = 0.0
    soil_water = []
    growth_water_stress = []
    for sequence, static, _, planting_day, soil_capacity, initial_soil_water in loader:
        output = model(
            sequence.to(device),
            static.to(device),
            planting_day.to(device),
            soil_capacity.to(device),
            initial_soil_water.to(device),
        )
        predictions.append(output["yield"].cpu().numpy())
        if "soil_water" in output:
            soil_water.append(output["soil_water"].cpu().numpy())
        if "growth_water_stress" in output:
            growth_water_stress.append(output["growth_water_stress"].cpu().numpy())
        if "et_correction" in output:
            active = output["active"]
            correction_square += float((
                (output["et_correction"].square() + output["rue_correction"].square())
                * active
            ).sum())
            correction_count += float(2.0 * active.sum())
        if "water_closure_error" in output:
            closure_square += float(output["water_closure_error"].square().sum())
            closure_count += float(output["water_closure_error"].numel())
    result = {"prediction": np.concatenate(predictions)}
    if correction_count:
        result["correction_rms"] = float(np.sqrt(correction_square / correction_count))
    if closure_count:
        result["water_closure_rmse_mm"] = float(np.sqrt(closure_square / closure_count))
    if soil_water:
        result["soil_water"] = np.concatenate(soil_water)
    if growth_water_stress:
        result["growth_water_stress"] = np.concatenate(growth_water_stress)
    return result


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _rank_correlation(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    observed_rank = _average_rank(observed)
    predicted_rank = _average_rank(predicted)
    if observed_rank.std() == 0 or predicted_rank.std() == 0:
        return None
    return float(np.corrcoef(observed_rank, predicted_rank)[0, 1])


def _cluster_bootstrap(
    observed: np.ndarray,
    predicted: np.ndarray,
    treatment_ids: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    """按处理而非样方重采样，给小样本外部验证提供不确定性区间。"""
    if repeats <= 0:
        return {}
    treatments = np.unique(treatment_ids)
    rng = np.random.default_rng(seed)
    sample_rmse = []
    treatment_rmse = []
    treatment_spearman = []
    for _ in range(repeats):
        selected = treatments[rng.integers(0, len(treatments), len(treatments))]
        observed_samples = np.concatenate([observed[treatment_ids == item] for item in selected])
        predicted_samples = np.concatenate([predicted[treatment_ids == item] for item in selected])
        observed_means = np.asarray([observed[treatment_ids == item].mean() for item in selected])
        predicted_means = np.asarray([predicted[treatment_ids == item].mean() for item in selected])
        sample_rmse.append(float(np.sqrt(np.mean(np.square(predicted_samples - observed_samples)))))
        treatment_rmse.append(float(np.sqrt(np.mean(np.square(predicted_means - observed_means)))))
        correlation = _rank_correlation(observed_means, predicted_means)
        if correlation is not None:
            treatment_spearman.append(correlation)

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [float(value) for value in np.percentile(values, (2.5, 50.0, 97.5))]

    return {
        "unit": "treatment cluster",
        "treatments": int(len(treatments)),
        "repeats": int(repeats),
        "seed": int(seed),
        "percentiles": [2.5, 50.0, 97.5],
        "sample_rmse_kg_ha": interval(sample_rmse),
        "treatment_mean_rmse_kg_ha": interval(treatment_rmse),
        "treatment_spearman": interval(treatment_spearman),
    }


def _paired_cluster_bootstrap(
    observed: np.ndarray,
    process: np.ndarray,
    plain: np.ndarray,
    treatment_ids: np.ndarray,
    repeats: int,
    seed: int,
    reference_name: str = "plain_nn",
) -> dict:
    """在同一批处理重采样上估计过程模型相对指定基线的指标差。"""
    if repeats <= 0:
        return {}
    treatments = np.unique(treatment_ids)
    groups = [np.flatnonzero(treatment_ids == item) for item in treatments]
    rng = np.random.default_rng(seed)
    rmse_difference = []
    rank_difference = []
    for _ in range(repeats):
        selected = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        process_rmse = np.sqrt(np.mean(np.square(process[indices] - observed[indices])))
        plain_rmse = np.sqrt(np.mean(np.square(plain[indices] - observed[indices])))
        rmse_difference.append(float(process_rmse - plain_rmse))
        observed_means = np.asarray([observed[groups[index]].mean() for index in selected])
        process_means = np.asarray([process[groups[index]].mean() for index in selected])
        plain_means = np.asarray([plain[groups[index]].mean() for index in selected])
        process_rank = _rank_correlation(observed_means, process_means)
        plain_rank = _rank_correlation(observed_means, plain_means)
        if process_rank is not None and plain_rank is not None:
            rank_difference.append(process_rank - plain_rank)
    return {
        "definition": f"process_nn minus {reference_name}; negative RMSE and positive Spearman favor process_nn",
        "unit": "treatment cluster",
        "treatments": int(len(treatments)),
        "repeats": int(repeats),
        "seed": int(seed),
        "percentiles": [2.5, 50.0, 97.5],
        "sample_rmse_difference_kg_ha": np.percentile(rmse_difference, [2.5, 50, 97.5]).tolist(),
        "treatment_spearman_difference": np.percentile(rank_difference, [2.5, 50, 97.5]).tolist() if rank_difference else None,
    }


def _compare_sensor_soil_moisture(
    model_soil_water: np.ndarray,
    soil_capacity: np.ndarray,
    treatment_ids: np.ndarray,
    metadata: dict,
    sensor_data: dict,
) -> dict:
    """比较根区有效水量和传感器湿度的标准化日变化形状。"""
    start_day = int(metadata["start_day"])
    end_day = int(metadata["end_day"])
    rows = []
    pooled_observed = []
    pooled_predicted = []
    for treatment, daily in sensor_data["daily"].items():
        selected = treatment_ids == treatment
        if not np.any(selected):
            continue
        days = np.asarray(sorted(int(day) for day in daily if start_day <= int(day) <= end_day))
        if len(days) < 3:
            continue
        observed = np.asarray([daily[str(day)] for day in days], dtype=np.float64)
        water_mm = model_soil_water[selected].mean(axis=0)[days - start_day]
        capacity_mm = float(soil_capacity[selected].mean())
        relative_water = 100.0 * water_mm / max(capacity_mm, 1e-8)
        observed_std = float(observed.std())
        predicted_std = float(relative_water.std())
        correlation = None
        normalized_rmse = None
        if observed_std > 0 and predicted_std > 0:
            observed_z = (observed - observed.mean()) / observed_std
            predicted_z = (relative_water - relative_water.mean()) / predicted_std
            correlation = float(np.corrcoef(observed_z, predicted_z)[0, 1])
            normalized_rmse = float(np.sqrt(np.mean(np.square(predicted_z - observed_z))))
            pooled_observed.extend(observed_z.tolist())
            pooled_predicted.extend(predicted_z.tolist())
        rows.append({
            "treatment": treatment,
            "matched_days": int(len(days)),
            "first_day_of_year": int(days[0]),
            "last_day_of_year": int(days[-1]),
            "pearson_temporal_correlation": correlation,
            "zscore_rmse": normalized_rmse,
            "observed_humidity_percent_mean": float(observed.mean()),
            "observed_humidity_percent_std": observed_std,
            "model_relative_extractable_water_percent_mean": float(relative_water.mean()),
            "model_relative_extractable_water_percent_std": predicted_std,
            "daily": {
                "day_of_year": days.tolist(),
                "observed_humidity_percent": observed.tolist(),
                "model_relative_extractable_water_percent": relative_water.tolist(),
            },
        })
    pooled_correlation = None
    if len(pooled_observed) >= 3:
        pooled_correlation = float(np.corrcoef(pooled_observed, pooled_predicted)[0, 1])
    return {
        "metric_scope": "within-treatment normalized daily dynamics; absolute units are intentionally not compared",
        "pooled_within_treatment_correlation": pooled_correlation,
        "treatments": rows,
        "sensor_source": {key: value for key, value in sensor_data.items() if key != "daily"},
    }


def _load_trained_model(
    checkpoint_path, model_name, sequence_dim, static_dim, metadata, device,
    wl_feature_dim=None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden = int(checkpoint["arguments"]["hidden"])
    if model_name == "wl_dnn":
        if wl_feature_dim is None:
            raise ValueError("WL-DNN validation cache is missing wl_features")
        model = WLDNNBaseline(
            wl_feature_dim,
            np.zeros(wl_feature_dim, dtype=np.float32),
            np.ones(wl_feature_dim, dtype=np.float32),
            0.0,
            1.0,
        )
    else:
        backbone_kind = checkpoint.get("arguments", {}).get("backbone", "cnn")
        if model_name == "process_nn":
            process_config = checkpoint.get("process_config", {})
            backbone_kind = process_config.get("backbone", backbone_kind)
        backbone = make_backbone(
            backbone_kind,
            sequence_dim,
            static_dim,
            hidden,
            np.zeros(sequence_dim, dtype=np.float32),
            np.ones(sequence_dim, dtype=np.float32),
            np.zeros(static_dim, dtype=np.float32),
            np.ones(static_dim, dtype=np.float32),
        )
    if model_name == "plain_nn":
        model = PlainNN(
            backbone, hidden, 0.0, 1.0, metadata["start_day"], metadata["end_day"]
        )
    elif model_name == "process_nn":
        process_config = checkpoint.get("process_config", {})
        checkpoint_process_version = process_config.get("version")
        if checkpoint_process_version not in SUPPORTED_PROCESS_MODEL_VERSIONS:
            raise ValueError(
                "过程模型 checkpoint 版本不匹配；请使用当前显式 SW→F_W→B→Y 结构重新训练"
            )
        model = ProcessConstrainedNN(
            backbone,
            hidden,
            metadata["start_day"],
            metadata["end_day"],
            maturity_gdd=float(process_config["maturity_gdd"]),
            rain_efficiency=float(process_config["rain_efficiency"]),
            irrigation_efficiency=float(process_config["irrigation_efficiency"]),
            et_scale=float(process_config["et_scale"]),
            drainage_fraction=float(process_config["drainage_fraction"]),
            water_stress_threshold=float(process_config["water_stress_threshold"]),
            water_stress_slope=float(process_config["water_stress_slope"]),
            process_mode=process_config.get("process_mode", "hard"),
            use_nutrients=process_config.get("use_nutrients", False),
            fixed_yield_coefficient=process_config.get(
                "fixed_yield_coefficient", process_config.get("fixed_harvest_index")
            ),
            process_parameters=process_config.get("process_parameters", process_config),
            process_model_version=checkpoint_process_version,
            et_correction_limit=float(process_config.get("et_correction_limit", ET_CORRECTION_LIMIT)),
            rue_correction_limit=float(process_config.get("rue_correction_limit", RUE_CORRECTION_LIMIT)),
            correction_uses_irrigation=bool(process_config.get("correction_uses_irrigation", True)),
            disable_growth_water_stress=bool(process_config.get("disable_growth_water_stress", False)),
            disable_late_growth_water_stress=bool(process_config.get("disable_late_growth_water_stress", False)),
        )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model


def run_external_validation(args):
    cache = np.load(args.cache, allow_pickle=False)
    sequence = cache["sequence"]
    static = cache["static"]
    wl_features = cache["wl_features"] if "wl_features" in cache.files else None
    labels = cache["labels"]
    planting_day = cache["planting_day"]
    sample_ids = cache["sample_ids"]
    treatment_ids = cache["treatment_ids"]
    metadata = json.loads(str(cache["metadata"]))
    soil_capacity, initial_soil_water = derive_soil_water_features(static, metadata)
    sensor_data = (
        read_2024_sensor_soil_moisture(args.sensor_workbook)
        if args.sensor_workbook else None
    )
    indices = np.arange(len(labels))
    device = torch.device(args.device or "cpu")
    loader = DataLoader(
        ArrayDataset(
            sequence, static, labels, planting_day,
            soil_capacity, initial_soil_water, indices,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    checkpoints = {}
    if args.process_checkpoint:
        checkpoints["process_nn"] = args.process_checkpoint
    if args.pure_checkpoint:
        checkpoints["plain_nn"] = args.pure_checkpoint
    if args.wl_checkpoint:
        checkpoints["wl_dnn"] = args.wl_checkpoint
    if not checkpoints:
        raise ValueError("provide at least one process, plain, or WL checkpoint")
    output = {
        "status": "external_field_validation",
        "device": str(device),
        "cache_metadata": metadata,
        "soil_state_summary": {
            "capacity_mm_min": float(soil_capacity.min()),
            "capacity_mm_max": float(soil_capacity.max()),
            "initial_water_mm_min": float(initial_soil_water.min()),
            "initial_water_mm_max": float(initial_soil_water.max()),
            "derivation": "sum(max(SDUL-SLLL,0) * layer_thickness_cm * 10); SH2O linearly interpolated to soil layer bottoms and clipped to [SLLL, SDUL]",
        },
        "models": {},
    }
    predictions = {}
    for model_name, checkpoint_path in checkpoints.items():
        model_loader = loader
        if model_name == "wl_dnn":
            if wl_features is None:
                raise ValueError("rebuild the validation cache with exact WL-DNN features")
            model_loader = DataLoader(
                ArrayDataset(
                    wl_features, np.zeros((len(labels), 1), dtype=np.float32),
                    labels, planting_day, soil_capacity, initial_soil_water, indices,
                ),
                batch_size=args.batch_size, shuffle=False, num_workers=0,
                pin_memory=device.type == "cuda",
            )
        model = _load_trained_model(
            checkpoint_path, model_name, sequence.shape[-1], static.shape[-1], metadata, device,
            None if wl_features is None else wl_features.shape[-1],
        )
        predicted = _predict(model, model_loader, device)
        prediction = predicted["prediction"]
        predictions[model_name] = prediction
        treatment_rows = []
        observed_means = []
        predicted_means = []
        for treatment in metadata["treatments"]:
            selected = treatment_ids == treatment
            observed_mean = float(labels[selected].mean())
            predicted_mean = float(prediction[selected].mean())
            observed_means.append(observed_mean)
            predicted_means.append(predicted_mean)
            treatment_rows.append({
                "treatment": treatment,
                "observed_mean_kg_ha": observed_mean,
                "predicted_mean_kg_ha": predicted_mean,
                "error_kg_ha": predicted_mean - observed_mean,
            })
        observed_means_array = np.asarray(observed_means)
        predicted_means_array = np.asarray(predicted_means)
        result = {
            "sample_metrics": _metrics(labels, prediction),
            "treatment_mean_metrics": _metrics(observed_means_array, predicted_means_array),
            "treatment_spearman": _rank_correlation(observed_means_array, predicted_means_array),
            "cluster_bootstrap_95ci": _cluster_bootstrap(
                labels, prediction, treatment_ids,
                repeats=args.bootstrap_repeats, seed=args.bootstrap_seed,
            ),
            "treatments": treatment_rows,
            "samples": [
                {
                    "sample_id": str(sample_id),
                    "treatment": str(treatment),
                    "observed_kg_ha": float(observed),
                    "predicted_kg_ha": float(estimate),
                    "error_kg_ha": float(estimate - observed),
                }
                for sample_id, treatment, observed, estimate in zip(
                    sample_ids, treatment_ids, labels, prediction
                )
            ],
        }
        if "correction_rms" in predicted:
            result["correction_rms"] = predicted["correction_rms"]
        if "water_closure_rmse_mm" in predicted:
            result["water_closure_rmse_mm"] = predicted["water_closure_rmse_mm"]
        if sensor_data is not None and "soil_water" in predicted:
            result["sensor_soil_moisture_validation"] = _compare_sensor_soil_moisture(
                predicted["soil_water"], soil_capacity, treatment_ids, metadata, sensor_data
            )
        if "growth_water_stress" in predicted:
            growth_stress = predicted["growth_water_stress"]
            treatment_growth_stress = {}
            for treatment in metadata["treatments"]:
                selected = treatment_ids == treatment
                if not np.any(selected):
                    continue
                treatment_daily = growth_stress[selected].mean(axis=0)
                treatment_growth_stress[str(treatment)] = {
                    "mean": float(treatment_daily.mean()),
                    "minimum": float(treatment_daily.min()),
                    "stress_days_below_0.95": int(np.sum(treatment_daily < 0.95)),
                    "daily_mean": treatment_daily.tolist(),
                }
            result["growth_water_stress_summary"] = {
                "stress_day_threshold": 0.95,
                "day_of_year": list(range(int(metadata["start_day"]), int(metadata["end_day"]) + 1)),
                "treatments": treatment_growth_stress,
            }
        output["models"][model_name] = result

    if {"process_nn", "plain_nn"}.issubset(predictions):
        output["paired_process_vs_plain_95ci"] = _paired_cluster_bootstrap(
            labels, predictions["process_nn"], predictions["plain_nn"],
            treatment_ids, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed,
        )
    if {"process_nn", "wl_dnn"}.issubset(predictions):
        output["paired_process_vs_wl_dnn_95ci"] = _paired_cluster_bootstrap(
            labels, predictions["process_nn"], predictions["wl_dnn"],
            treatment_ids, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed,
            reference_name="wl_dnn",
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "result": str(output_path),
        "models": {
            name: {
                "sample_metrics": result["sample_metrics"],
                "treatment_mean_metrics": result["treatment_mean_metrics"],
                "treatment_spearman": result["treatment_spearman"],
            }
            for name, result in output["models"].items()
        },
    }, ensure_ascii=False, indent=2))


def train_model(
    model, loaders, device, yield_std, epochs, learning_rate,
    correction_weight, physics_weight, patience, optimizer_kind="adamw",
    standardize_yield_loss=True,
):
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=learning_rate)
        if optimizer_kind == "adam"
        else torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    )
    best_state = None
    best_rmse = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_data_loss = 0.0
        total_correction_loss = 0.0
        total_physics_loss = 0.0
        total_loss = 0.0
        samples = 0
        for sequence, static, labels, planting_day, soil_capacity, initial_soil_water in loaders["train"]:
            sequence = sequence.to(device)
            static = static.to(device)
            labels = labels.to(device)
            planting_day = planting_day.to(device)
            soil_capacity = soil_capacity.to(device)
            initial_soil_water = initial_soil_water.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence, static, planting_day, soil_capacity, initial_soil_water)
            residual = output["yield"] - labels
            if standardize_yield_loss:
                residual = residual / yield_std
            data_loss = torch.mean(torch.square(residual))
            correction_loss = torch.zeros((), device=device)
            physics_loss = torch.zeros((), device=device)
            if "et_correction" in output:
                active = output["active"]
                correction_loss = (
                    (output["et_correction"].square() + output["rue_correction"].square())
                    * active
                ).sum() / (2.0 * active.sum().clamp_min(1.0))
            if "growth_residual" in output and getattr(model, "process_mode", "hard") == "agripinn":
                active = output["active"]
                physics_loss = (
                    output["growth_residual"].square() * active
                ).sum() / active.sum().clamp_min(1.0)
            loss = (
                data_loss
                + correction_weight * correction_loss
                + physics_weight * physics_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_data_loss += float(data_loss.detach()) * len(labels)
            total_correction_loss += float(correction_loss.detach()) * len(labels)
            total_physics_loss += float(physics_loss.detach()) * len(labels)
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
        validation = evaluate(model, loaders["validation"], device)
        row = {
            "epoch": epoch,
            "train_data_loss": total_data_loss / samples,
            "train_correction_loss": total_correction_loss / samples,
            "train_physics_loss": total_physics_loss / samples,
            "train_total_loss": total_loss / samples,
            **validation,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation["rmse"] < best_rmse:
            best_rmse = validation["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return history, best_epoch


def irrigation_gradient_scan(
    model,
    dataset,
    device,
    max_records=8,
    day_fractions=(0.25, 0.5, 0.75),
    epsilons=(0.01, 0.1, 1.0),
):
    """用可行域内有限差分核对灌溉梯度；过程模型使用 float64 副本。"""
    check_dtype = torch.float32 if isinstance(model, WLDNNBaseline) else torch.float64
    check_model = copy.deepcopy(model).to(device=device, dtype=check_dtype)
    check_model.eval()
    if len(dataset) == 0:
        raise ValueError("梯度扫描数据集为空")
    selected = np.linspace(0, len(dataset) - 1, min(max_records, len(dataset)), dtype=int)
    rows = []
    irrigation_index = SEQUENCE_FEATURES.index("IRR")
    for record_index in selected:
        sequence, static, _, planting_day, soil_capacity, initial_soil_water = dataset[record_index]
        sequence = sequence[None].to(device=device, dtype=check_dtype).requires_grad_(True)
        static = static[None].to(device=device, dtype=check_dtype)
        planting_day = planting_day[None].to(device)
        soil_capacity = soil_capacity[None].to(device=device, dtype=check_dtype)
        initial_soil_water = initial_soil_water[None].to(device=device, dtype=check_dtype)
        output = check_model(sequence, static, planting_day, soil_capacity, initial_soil_water)
        gradient = torch.autograd.grad(output["yield"].sum(), sequence)[0][0, :, irrigation_index]
        finite = bool(torch.isfinite(gradient).all())
        if not finite:
            rows.append({"record_index": int(record_index), "finite": False})
            continue
        for fraction in day_fractions:
            day_index = min(sequence.shape[1] - 1, max(0, round(fraction * (sequence.shape[1] - 1))))
            analytic = float(gradient[day_index].detach().cpu())
            base_irrigation = float(sequence[0, day_index, irrigation_index].detach().cpu())
            for epsilon in epsilons:
                with torch.no_grad():
                    plus = sequence.detach().clone()
                    plus[0, day_index, irrigation_index] += epsilon
                    plus_yield = check_model(
                        plus, static, planting_day, soil_capacity, initial_soil_water
                    )["yield"]
                    if base_irrigation < epsilon:
                        base_yield = check_model(
                            sequence.detach(), static, planting_day,
                            soil_capacity, initial_soil_water,
                        )["yield"]
                        numerical = float(((plus_yield - base_yield) / epsilon).cpu())
                        difference_scheme = "forward_at_lower_bound"
                    else:
                        minus = sequence.detach().clone()
                        minus[0, day_index, irrigation_index] -= epsilon
                        minus_yield = check_model(
                            minus, static, planting_day, soil_capacity, initial_soil_water
                        )["yield"]
                        numerical = float(((plus_yield - minus_yield) / (2 * epsilon)).cpu())
                        difference_scheme = "central"
                relative_error = abs(analytic - numerical) / max(
                    abs(analytic), abs(numerical), 1e-8
                )
                rows.append({
                    "record_index": int(record_index),
                    "day_index": int(day_index),
                    "base_irrigation_mm": base_irrigation,
                    "epsilon": float(epsilon),
                    "difference_scheme": difference_scheme,
                    "finite": True,
                    "analytic": analytic,
                    "finite_difference": numerical,
                    "absolute_error": float(abs(analytic - numerical)),
                    "relative_error": float(relative_error),
                })
    valid = [row for row in rows if row.get("finite") and "relative_error" in row]
    return {
        "records_checked": int(len(selected)),
        "points_checked": int(len(valid)),
        "finite_fraction": float(sum(row.get("finite", False) for row in rows) / max(len(rows), 1)),
        "max_abs_analytic": float(max((abs(row["analytic"]) for row in valid), default=0.0)),
        "mean_absolute_error": float(np.mean([row["absolute_error"] for row in valid])) if valid else None,
        "median_absolute_error": float(np.median([row["absolute_error"] for row in valid])) if valid else None,
        "meaningful_gradient_fraction": float(
            np.mean([abs(row["analytic"]) >= 1e-3 for row in valid])
        ) if valid else None,
        "max_relative_error": float(max((row["relative_error"] for row in valid), default=0.0)),
        "mean_relative_error": float(np.mean([row["relative_error"] for row in valid])) if valid else None,
        "median_relative_error": float(np.median([row["relative_error"] for row in valid])) if valid else None,
        "rows": rows,
    }


def run_training(args):
    set_determinism(args.seed)
    cache = np.load(args.cache, allow_pickle=False)
    sequence = cache["sequence"]
    static = cache["static"]
    wl_features = cache["wl_features"] if "wl_features" in cache.files else None
    labels = cache["labels"]
    planting_day = cache["planting_day"]
    groups = cache["groups"]
    metadata = json.loads(str(cache["metadata"]))
    water_calibration = load_water_calibration(args.water_calibration)
    soil_capacity, initial_soil_water = derive_soil_water_features(static, metadata)
    splits = group_split(groups, args.seed)
    if args.pilot_groups:
        for name in splits:
            group_count = max(1, round(args.pilot_groups * {"train": 0.7, "validation": 0.15, "test": 0.15}[name]))
            selected = np.unique(groups[splits[name]])[:group_count]
            splits[name] = splits[name][np.isin(groups[splits[name]], selected)]

    sequence_mean, sequence_std = _chunk_statistics(sequence, splits["train"])
    static_mean, static_std = _chunk_statistics(static, splits["train"])
    if (args.include_wl_dnn or args.train_model == "wl_dnn") and wl_features is None:
        raise ValueError("rebuild the training cache with exact 1997-D WL-DNN features")
    wl_mean = wl_std = None
    if wl_features is not None:
        wl_mean, wl_std = _chunk_statistics(wl_features, splits["train"])
    yield_mean = float(labels[splits["train"]].mean())
    yield_std = float(labels[splits["train"]].std())
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders = {
        name: DataLoader(
            ArrayDataset(
                sequence, static, labels, planting_day,
                soil_capacity, initial_soil_water, indices,
            ),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name, indices in splits.items()
    }

    results = {
        "status": "pilot" if args.pilot_groups else "formal",
        "process_model_version": PROCESS_MODEL_VERSION,
        "backbone": args.backbone,
        "seed": args.seed,
        "device": str(device),
        "cache_metadata": metadata,
        "soil_state_summary": {
            "capacity_mm_min": float(soil_capacity.min()),
            "capacity_mm_max": float(soil_capacity.max()),
            "initial_water_mm_min": float(initial_soil_water.min()),
            "initial_water_mm_max": float(initial_soil_water.max()),
            "derivation": "sum(max(SDUL-SLLL,0) * layer_thickness_cm * 10); SH2O linearly interpolated to soil layer bottoms and clipped to [SLLL, SDUL]",
        },
        "split_samples": {name: len(indices) for name, indices in splits.items()},
        "split_groups": {name: int(np.unique(groups[indices]).size) for name, indices in splits.items()},
        "models": {},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = ["plain_nn", "process_nn"] if args.train_model == "all" else [args.train_model]
    if args.include_wl_dnn and "wl_dnn" not in model_names:
        model_names.insert(0, "wl_dnn")
    for name in model_names:
        set_determinism(args.seed)
        model_loaders = loaders
        model_epochs = args.epochs
        model_learning_rate = args.learning_rate
        model_patience = args.patience
        optimizer_kind = "adamw"
        standardize_yield_loss = True
        if name == "wl_dnn":
            legacy_protocol = args.wl_training_protocol == "legacy"
            model = WLDNNBaseline(
                wl_features.shape[-1], wl_mean, wl_std,
                0.0 if legacy_protocol else yield_mean,
                1.0 if legacy_protocol else yield_std,
            )
            dummy_static = np.zeros((len(labels), 1), dtype=np.float32)
            requested_wl_batch = args.wl_batch_size or (
                len(splits["train"]) if legacy_protocol else args.batch_size
            )
            model_loaders = {
                split_name: DataLoader(
                    ArrayDataset(
                        wl_features, dummy_static, labels, planting_day,
                        soil_capacity, initial_soil_water, indices,
                    ),
                    batch_size=requested_wl_batch if split_name == "train" else args.batch_size,
                    shuffle=split_name == "train", num_workers=0,
                    pin_memory=device.type == "cuda",
                )
                for split_name, indices in splits.items()
            }
            if legacy_protocol:
                model_epochs = args.wl_epochs
                model_learning_rate = args.wl_learning_rate
                model_patience = args.wl_patience
                optimizer_kind = "adam"
                standardize_yield_loss = False
            correction_weight = 0.0
            physics_weight = 0.0
        else:
            backbone = make_backbone(
                args.backbone,
                sequence.shape[-1], static.shape[-1], args.hidden,
                sequence_mean, sequence_std, static_mean, static_std,
            )
        if name == "plain_nn":
            model = PlainNN(
                backbone, args.hidden, yield_mean, yield_std,
                metadata["start_day"], metadata["end_day"],
            )
            correction_weight = 0.0
            physics_weight = 0.0
        elif name == "process_nn":
            model = ProcessConstrainedNN(
                backbone,
                args.hidden,
                metadata["start_day"],
                metadata["end_day"],
                maturity_gdd=args.maturity_gdd,
                rain_efficiency=water_calibration["rain_efficiency"],
                irrigation_efficiency=water_calibration["irrigation_efficiency"],
                et_scale=water_calibration["et_scale"],
                drainage_fraction=water_calibration["drainage_fraction"],
                water_stress_threshold=water_calibration["water_stress_threshold"],
                water_stress_slope=water_calibration["water_stress_slope"],
                process_mode=args.process_mode,
                use_nutrients=args.use_nutrients,
                fixed_yield_coefficient=args.fixed_yield_coefficient,
                process_parameters=water_calibration["process_parameters"],
                process_model_version=PROCESS_MODEL_VERSION,
                et_correction_limit=args.et_correction_limit,
                rue_correction_limit=args.rue_correction_limit,
                correction_uses_irrigation=not args.exclude_irrigation_from_corrections,
                disable_growth_water_stress=args.disable_growth_water_stress,
                disable_late_growth_water_stress=args.disable_late_growth_water_stress,
                yield_mean=yield_mean,
                yield_std=yield_std,
            )
            correction_weight = args.correction_weight
            physics_weight = args.physics_weight
        else:
            physics_weight = 0.0
        model.to(device)
        started = time.time()
        history, best_epoch = train_model(
            model, model_loaders, device, yield_std, model_epochs,
            model_learning_rate, correction_weight, physics_weight, model_patience,
            optimizer_kind=optimizer_kind,
            standardize_yield_loss=standardize_yield_loss,
        )
        model_result = {
            "best_epoch": best_epoch,
            "seconds": time.time() - started,
            "validation": evaluate(model, model_loaders["validation"], device),
            "test": evaluate(model, model_loaders["test"], device),
            "irrigation_gradient": None if (args.skip_gradient_scan or name == "wl_dnn") else irrigation_gradient_scan(
                model,
                ArrayDataset(
                    sequence, static, labels, planting_day,
                    soil_capacity, initial_soil_water, splits["test"],
                ),
                device,
                max_records=args.gradient_records,
            ),
            "history": history,
        }
        results["models"][name] = model_result
        process_config = None
        if name == "process_nn":
            process_config = {
                "version": PROCESS_MODEL_VERSION,
                "backbone": args.backbone,
                "maturity_gdd": float(model.maturity_gdd),
                "rain_efficiency": float(model.rain_efficiency),
                "irrigation_efficiency": float(model.irrigation_efficiency),
                "et_scale": float(model.et_scale),
                "et_model": model.et_model,
                "lai_model": model.lai_model,
                "drainage_fraction": float(model.drainage_fraction),
                "water_stress_threshold": float(model.water_stress_threshold),
                "water_stress_slope": float(model.water_stress_slope),
                "water_calibration_source": water_calibration["source"],
                "water_calibration_sha256": water_calibration["sha256"],
                "process_parameters": model.process_parameters,
                "process_mode": model.process_mode,
                "constraint_type": "hard_state_recurrence" if model.process_mode == "hard" else "agripinn_soft_growth_residual",
                "learned_daily_corrections": ["et", "rue"],
                "et_correction_limit": float(model.et_correction_limit),
                "rue_correction_limit": float(model.rue_correction_limit),
                "correction_uses_irrigation": bool(model.correction_uses_irrigation),
                "disable_growth_water_stress": bool(model.disable_growth_water_stress),
                "disable_late_growth_water_stress": bool(model.disable_late_growth_water_stress),
                "use_nutrients": bool(model.use_nutrients),
                "fixed_yield_coefficient": model.fixed_yield_coefficient,
                "yield_coefficient_semantics": "effective biomass-to-yield mapping coefficient; not a physiological harvest index",
                "paper_calibration_protocol": PAPER_FIELD_PROTOCOL,
                "physics_weight": float(args.physics_weight),
            }
        torch.save(
            {
                "model": model.state_dict(),
                "result": model_result,
                "arguments": vars(args),
                "process_config": process_config,
            },
            output_dir / f"{name}.pt",
        )
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(result_path), "models": results["models"]}, ensure_ascii=False))


def run_gradient_check(args):
    """Audit one frozen process checkpoint without retraining it."""
    cache = np.load(args.cache, allow_pickle=False)
    sequence = cache["sequence"]
    static = cache["static"]
    labels = cache["labels"]
    planting_day = cache["planting_day"]
    groups = cache["groups"]
    metadata = json.loads(str(cache["metadata"]))
    soil_capacity, initial_soil_water = derive_soil_water_features(static, metadata)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    seed = int(checkpoint.get("arguments", {}).get("seed", args.seed))
    test_indices = group_split(groups, seed)["test"]
    device = torch.device(args.device or "cpu")
    model = _load_trained_model(
        args.checkpoint, "process_nn", sequence.shape[-1], static.shape[-1], metadata, device
    )
    audit = irrigation_gradient_scan(
        model,
        ArrayDataset(
            sequence, static, labels, planting_day,
            soil_capacity, initial_soil_water, test_indices,
        ),
        device,
        max_records=args.records,
    )
    result = {
        "status": "frozen_checkpoint_gradient_audit",
        "checkpoint": str(args.checkpoint),
        "seed": seed,
        "test_samples": int(len(test_indices)),
        "audit": audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(output), "audit": audit}, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--source", default="output_all_12_22.json")
    cache_parser.add_argument("--cache", default="results/training_cache.npz")
    cache_parser.add_argument("--expected-records", type=int, default=51480)
    cache_parser.add_argument("--start-day", type=int, default=90)
    cache_parser.add_argument("--end-day", type=int, default=300)

    validation_cache_parser = subparsers.add_parser("validation-cache")
    validation_cache_parser.add_argument("--workbook", required=True)
    validation_cache_parser.add_argument("--weather-csv", required=True)
    validation_cache_parser.add_argument("--template-json", default="2024data.json")
    validation_cache_parser.add_argument("--cache", default="results/2024_field_validation.npz")
    validation_cache_parser.add_argument("--start-day", type=int, default=90)
    validation_cache_parser.add_argument("--end-day", type=int, default=300)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache", default="results/training_cache.npz")
    train_parser.add_argument("--output-dir", default="results/run")
    train_parser.add_argument("--device", default="")
    train_parser.add_argument("--backbone", choices=("cnn", "dnn"), default="dnn")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--hidden", type=int, default=32)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--correction-weight", type=float, default=1e-3)
    train_parser.add_argument(
        "--et-correction-limit", type=float, default=ET_CORRECTION_LIMIT,
        help="DNN 对逐日 ET 的最大相对修正；设为 0 可冻结已校准 ET",
    )
    train_parser.add_argument(
        "--rue-correction-limit", type=float, default=RUE_CORRECTION_LIMIT,
        help="DNN 对逐日 RUE 的最大相对修正",
    )
    train_parser.add_argument(
        "--disable-growth-water-stress", action="store_true",
        help="将 Growth 侧 F_W 置为 1；保留水量平衡、ET 和蒸腾胁迫计算",
    )
    train_parser.add_argument(
        "--disable-late-growth-water-stress", action="store_true",
        help="仅在 daily_stage >= 0.75 时将 Growth 侧 F_W 置为 1；保留水量平衡、ET 和蒸腾胁迫计算",
    )
    train_parser.add_argument(
        "--exclude-irrigation-from-corrections", action="store_true",
        help="灌溉仅经 SW→F_W 过程链影响产量，不直接进入 ET/RUE 修正网络",
    )
    train_parser.add_argument("--physics-weight", type=float, default=0.1)
    train_parser.add_argument("--process-mode", choices=("hard", "agripinn"), default="hard")
    train_parser.add_argument("--use-nutrients", action="store_true")
    train_parser.add_argument(
        "--fixed-yield-coefficient", "--fixed-hi",
        dest="fixed_yield_coefficient", type=float, default=None,
        help="固定有效生物量-产量转换系数；--fixed-hi 仅为旧命令兼容别名",
    )
    train_parser.add_argument("--maturity-gdd", type=float, default=2600.0)
    train_parser.add_argument(
        "--water-calibration", default="",
        help="2022 SWXD/ETAA water-only calibration JSON with independent 2023 validation",
    )
    train_parser.add_argument("--pilot-groups", type=int, default=0)
    train_parser.add_argument("--include-wl-dnn", action="store_true")
    train_parser.add_argument(
        "--train-model", choices=("all", "plain_nn", "process_nn", "wl_dnn"), default="all",
        help="只训练指定模型；用于在已有过程 checkpoint 的同一划分上补齐基线",
    )
    train_parser.add_argument("--gradient-records", type=int, default=8)
    train_parser.add_argument("--skip-gradient-scan", action="store_true")
    train_parser.add_argument("--wl-epochs", type=int, default=1000)
    train_parser.add_argument("--wl-learning-rate", type=float, default=0.0211)
    train_parser.add_argument("--wl-patience", type=int, default=1000)
    train_parser.add_argument(
        "--wl-batch-size", type=int, default=0,
        help="WL-DNN training batch; 0 reproduces the original full-batch update",
    )
    train_parser.add_argument(
        "--wl-training-protocol", choices=("fair", "legacy"), default="fair",
        help="fair uses the common training budget; legacy uses Wang's 1000-epoch Adam settings",
    )

    validation_parser = subparsers.add_parser("validate")
    validation_parser.add_argument("--cache", default="results/2024_field_validation.npz")
    validation_parser.add_argument("--pure-checkpoint", default="")
    validation_parser.add_argument("--process-checkpoint", default="")
    validation_parser.add_argument("--wl-checkpoint", default="")
    validation_parser.add_argument("--output", default="results/2024_field_validation.json")
    validation_parser.add_argument("--device", default="cpu")
    validation_parser.add_argument("--batch-size", type=int, default=256)
    validation_parser.add_argument("--sensor-workbook", default="")
    validation_parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    validation_parser.add_argument("--bootstrap-seed", type=int, default=20260910)

    gradient_parser = subparsers.add_parser("gradient-check")
    gradient_parser.add_argument("--cache", default="results/training_cache.npz")
    gradient_parser.add_argument("--checkpoint", required=True)
    gradient_parser.add_argument("--output", required=True)
    gradient_parser.add_argument("--device", default="cpu")
    gradient_parser.add_argument("--records", type=int, default=8)
    gradient_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "cache":
        print(json.dumps(build_cache(
            arguments.source, arguments.cache, arguments.expected_records,
            arguments.start_day, arguments.end_day,
        ), ensure_ascii=False, indent=2))
    elif arguments.command == "validation-cache":
        print(json.dumps(build_2024_field_validation_cache(
            arguments.workbook, arguments.weather_csv, arguments.template_json,
            arguments.cache, arguments.start_day, arguments.end_day,
        ), ensure_ascii=False, indent=2))
    elif arguments.command == "train":
        run_training(arguments)
    elif arguments.command == "validate":
        run_external_validation(arguments)
    else:
        run_gradient_check(arguments)
