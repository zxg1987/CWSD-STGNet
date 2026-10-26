from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _logit_scalar(value: float) -> float:
    value = min(max(float(value), 1e-7), 1.0 - 1e-7)
    return math.log(value) - math.log1p(-value)


class RobustMedianMADNorm(nn.Module):







    def __init__(
        self,
        mad_scale: float = 1.4826,
        eps: float = 1e-6,
        clip: float = 8.0,
        *,
        reference_scale: Optional[float] = None,
        ratio_min: float = 0.25,
        ratio_max: float = 4.0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if not (
            mad_scale > 0.0
            and eps > 0.0
            and clip > 0.0
            and 0.0 < ratio_min < 1.0 < ratio_max
            and 0.0 <= alpha <= 1.0
        ):
            raise ValueError("Invalid bounded-reference MAD parameters")
        self.mad_scale = float(mad_scale)
        self.eps = float(eps)
        self.clip = float(clip)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.alpha = float(alpha)
        initial = float("nan") if reference_scale is None else float(reference_scale)
        if not math.isnan(initial) and initial <= 0.0:
            raise ValueError("reference_scale must be positive")
        self.register_buffer(
            "reference_scale", torch.tensor(initial, dtype=torch.float32), persistent=True
        )
        self.last_diagnostics: Dict[str, Tensor] = {}
        self.last_per_sample_diagnostics: Dict[str, Tensor] = {}

    @staticmethod
    def _linear_median(x: Tensor) -> Tensor:
        return torch.quantile(
            x, 0.5, dim=-1, keepdim=True, interpolation="linear"
        )

    @classmethod
    def sample_robust_scales(cls, x: Tensor, mad_scale: float = 1.4826) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x=(B,C,T), got {tuple(x.shape)}")
        median = cls._linear_median(x)
        mad = cls._linear_median((x - median).abs())
        return float(mad_scale) * mad.squeeze(-1)

    @torch.no_grad()
    def set_reference_scale(self, value: float | Tensor) -> None:
        scalar = torch.as_tensor(
            value, dtype=self.reference_scale.dtype, device=self.reference_scale.device
        )
        if scalar.numel() != 1 or not torch.isfinite(scalar) or scalar <= 0:
            raise ValueError("reference_scale must be one finite positive scalar")
        self.reference_scale.copy_(scalar.reshape(()))

    @property
    def is_calibrated(self) -> bool:
        return bool(torch.isfinite(self.reference_scale).item()) and bool(
            (self.reference_scale > 0).item()
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x=(B,C,T), got {tuple(x.shape)}")
        if not self.is_calibrated:
            raise RuntimeError(
                "Bounded-reference MAD is not calibrated; compute s_ref from clean "
                "training samples and call set_reference_scale() first"
            )
        median = self._linear_median(x)
        sample_scale = self.mad_scale * self._linear_median((x - median).abs())
        reference = self.reference_scale.to(dtype=x.dtype, device=x.device)
        raw_ratio = sample_scale / reference.clamp_min(self.eps)
        ratio = raw_ratio.clamp(self.ratio_min, self.ratio_max)
        effective_scale = reference * ratio.pow(self.alpha)
        pre_clip = (x - median) / (effective_scale + self.eps)
        normalized = pre_clip.clamp(-self.clip, self.clip)
        lower_hit = raw_ratio.le(self.ratio_min).float().flatten(1).mean(dim=1)
        upper_hit = raw_ratio.ge(self.ratio_max).float().flatten(1).mean(dim=1)
        value_clip = pre_clip.abs().ge(self.clip).float().flatten(1).mean(dim=1)
        self.last_per_sample_diagnostics = {
            "ratio_lower_hit_rate": lower_hit.detach(),
            "ratio_upper_hit_rate": upper_hit.detach(),
            "value_clip_rate": value_clip.detach(),
        }
        self.last_diagnostics = {
            "sample_scale": sample_scale.detach(),
            "reference_scale": reference.detach().reshape(1),
            "ratio_lower_hit_rate": lower_hit.mean().detach(),
            "ratio_upper_hit_rate": upper_hit.mean().detach(),
            "value_clip_rate": value_clip.mean().detach(),
        }
        if not torch.isfinite(normalized).all():
            raise FloatingPointError("Bounded-reference MAD produced non-finite values")
        return normalized


class PerSampleZScoreNorm(nn.Module):


    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError("z-score epsilon must be positive")
        self.eps = float(eps)
        self.last_diagnostics: Dict[str, Tensor] = {}
        self.last_per_sample_diagnostics: Dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x=(B,C,T), got {tuple(x.shape)}")
        mean = x.mean(dim=-1, keepdim=True)
        scale = x.var(dim=-1, keepdim=True, unbiased=False).sqrt()
        normalized = (x - mean) / scale.clamp_min(self.eps)
        if not torch.isfinite(normalized).all():
            raise FloatingPointError("Per-sample z-score produced non-finite values")
        self.last_per_sample_diagnostics = {
            "sample_mean": mean.detach().flatten(1).mean(dim=1),
            "sample_std": scale.detach().flatten(1).mean(dim=1),
        }
        self.last_diagnostics = {
            "sample_mean": mean.detach().mean(),
            "sample_std": scale.detach().mean(),
        }
        return normalized


class LearnableWaveletBranchV2(nn.Module):


    def __init__(
        self,
        wavelet_type: str,
        frequency_anchors_hz: Sequence[float],
        *,
        sampling_rate: int = 8000,
        kernel_size: int = 63,
        frequency_offset_fraction: float = 0.04,
        cycles_min: float = 1.5,
        cycles_max: float = 2.25,
        cycles_init: float = 1.875,
    ) -> None:
        super().__init__()
        if wavelet_type not in {"laplace", "mexh", "morlet"}:
            raise ValueError(f"Unsupported wavelet_type={wavelet_type!r}")
        if kernel_size % 2 != 1:
            raise ValueError("Wavelet kernel size must be odd")
        anchors = torch.as_tensor(tuple(frequency_anchors_hz), dtype=torch.float32)
        if anchors.ndim != 1 or anchors.numel() == 0:
            raise ValueError("frequency_anchors_hz must be a non-empty vector")
        if not bool((anchors[1:] > anchors[:-1]).all()):
            raise ValueError("frequency anchors must be strictly ordered")
        self.wavelet_type = str(wavelet_type)
        self.out_channels = int(anchors.numel())
        self.sampling_rate = int(sampling_rate)
        self.kernel_size = int(kernel_size)
        self.frequency_offset_fraction = float(frequency_offset_fraction)
        self.cycles_min = float(cycles_min)
        self.cycles_max = float(cycles_max)
        self.register_buffer("frequency_anchors_hz", anchors, persistent=True)
        self.frequency_offset_raw = nn.Parameter(torch.zeros(self.out_channels))
        if self.wavelet_type in {"laplace", "morlet"}:
            position = (cycles_init - cycles_min) / (cycles_max - cycles_min)
            self.cycles_raw = nn.Parameter(
                torch.full((self.out_channels,), _logit_scalar(position))
            )
        else:
            self.register_parameter("cycles_raw", None)

    def physical_parameters(self) -> Dict[str, Tensor]:
        offset = self.frequency_offset_fraction * torch.tanh(
            self.frequency_offset_raw
        )
        frequency = self.frequency_anchors_hz * (1.0 + offset)
        values: Dict[str, Tensor] = {
            "frequency_hz": frequency,
            "frequency_offset_fraction": offset,
            "shift_seconds": torch.zeros_like(frequency),
        }
        if self.cycles_raw is not None:
            values["cycles"] = self.cycles_min + (
                self.cycles_max - self.cycles_min
            ) * torch.sigmoid(self.cycles_raw)
        else:
            values["sigma_seconds"] = math.sqrt(2.0) / (
                2.0 * math.pi * frequency
            )
        return values

    def build_kernel(self) -> Tensor:
        params = self.physical_parameters()
        half = self.kernel_size // 2
        time = (
            torch.arange(
                -half,
                half + 1,
                device=self.frequency_offset_raw.device,
                dtype=self.frequency_offset_raw.dtype,
            )
            / float(self.sampling_rate)
        ).view(1, -1)
        frequency = params["frequency_hz"].unsqueeze(1)
        if self.wavelet_type == "laplace":
            tau = (params["cycles"] / frequency.squeeze(1)).unsqueeze(1)
            kernel = torch.exp(-time.abs() / tau) * torch.sin(
                2.0 * math.pi * frequency * time
            )
        elif self.wavelet_type == "morlet":
            sigma = (
                params["cycles"] / (2.0 * math.pi * frequency.squeeze(1))
            ).unsqueeze(1)
            kernel = torch.exp(-0.5 * (time / sigma).square()) * torch.cos(
                2.0 * math.pi * frequency * time
            )
        else:
            sigma = params["sigma_seconds"].unsqueeze(1)
            radius = time / sigma
            kernel = (1.0 - radius.square()) * torch.exp(-0.5 * radius.square())
        kernel = kernel - kernel.mean(dim=-1, keepdim=True)
        kernel = kernel / kernel.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(
            1e-6
        )
        return kernel.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(
            x, self.build_kernel(), bias=None, padding=self.kernel_size // 2
        )


class EquivariantWaveletFrontEnd(nn.Module):


    def __init__(
        self,
        channels_per_branch: int = 16,
        kernel_size: int = 63,
        pool_kernel: int = 4,
        pool_stride: int = 4,
        *,
        sampling_rate: int = 8000,
        frequency_anchors_hz: Optional[Sequence[float]] = None,
        frequency_offset_fraction: float = 0.04,
        cycles_min: float = 1.5,
        cycles_max: float = 2.25,
        cycles_init: float = 1.875,
        node_gain_min: float = 0.25,
        node_gain_max: float = 4.0,
        node_gain_init: float = 1.0,
        node_scale_mode: str = "n1_gain_only",
        n2_ema_decay: float = 0.99,
        branch_types: Sequence[str] = ("laplace", "mexh", "morlet"),
    ) -> None:
        super().__init__()
        if frequency_anchors_hz is None:
            frequency_anchors_hz = tuple(
                torch.logspace(math.log10(300.0), math.log10(3000.0), channels_per_branch)
                .tolist()
            )
        if len(tuple(frequency_anchors_hz)) != channels_per_branch:
            raise ValueError("Frequency anchor count must equal channels_per_branch")
        if not node_gain_min < node_gain_init < node_gain_max:
            raise ValueError("node_gain_init must lie strictly inside bounds")
        if node_scale_mode != "n1_gain_only":
            raise ValueError("the final model supports only N1 gain-only scaling")
        self.channels_per_branch = int(channels_per_branch)
        self.num_nodes = self.channels_per_branch * 3
        self.pool_kernel = int(pool_kernel)
        self.pool_stride = int(pool_stride)
        self.node_gain_min = float(node_gain_min)
        self.node_gain_max = float(node_gain_max)
        self.node_scale_mode = str(node_scale_mode)
        self.n2_ema_decay = float(n2_ema_decay)
        branch_types = tuple(str(value) for value in branch_types)
        if len(branch_types) != 3 or any(
            value not in {"laplace", "mexh", "morlet"} for value in branch_types
        ):
            raise ValueError("branch_types must contain exactly three supported wavelet names")
        self.branch_types = branch_types
        branch_kwargs = dict(
            frequency_anchors_hz=tuple(frequency_anchors_hz),
            sampling_rate=sampling_rate,
            kernel_size=kernel_size,
            frequency_offset_fraction=frequency_offset_fraction,
            cycles_min=cycles_min,
            cycles_max=cycles_max,
            cycles_init=cycles_init,
        )


        self.laplace = LearnableWaveletBranchV2(branch_types[0], **branch_kwargs)
        self.mexh = LearnableWaveletBranchV2(branch_types[1], **branch_kwargs)
        self.morlet = LearnableWaveletBranchV2(branch_types[2], **branch_kwargs)
        position = (node_gain_init - node_gain_min) / (
            node_gain_max - node_gain_min
        )
        self.node_gain_raw = nn.Parameter(
            torch.full((self.num_nodes,), _logit_scalar(position))
        )
        self.register_buffer(
            "n2_reference_power", torch.ones(self.num_nodes), persistent=True
        )
        self.register_buffer(
            "n2_initialized", torch.tensor(False), persistent=True
        )
        self.register_buffer(
            "n2_update_count", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def node_gain(self) -> Tensor:
        return self.node_gain_min + (
            self.node_gain_max - self.node_gain_min
        ) * torch.sigmoid(self.node_gain_raw)

    def raw_response(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected x=(B,1,T), got {tuple(x.shape)}")
        return torch.cat((self.laplace(x), self.mexh(x), self.morlet(x)), dim=1)

    def signed_response(self, x: Tensor) -> Tensor:
        return self.raw_response(x) * self.node_gain().view(1, -1, 1)

    def pooled_on_off(self, response: Tensor) -> Tensor:
        on_energy = torch.log1p(F.relu(response).square())
        off_energy = torch.log1p(F.relu(-response).square())
        on_down = F.avg_pool1d(
            on_energy, kernel_size=self.pool_kernel, stride=self.pool_stride
        )
        off_down = F.avg_pool1d(
            off_energy, kernel_size=self.pool_kernel, stride=self.pool_stride
        )
        return torch.stack((on_down, off_down), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pooled_on_off(self.signed_response(x))

    def all_kernels(self) -> Tensor:
        return torch.cat(
            (
                self.laplace.build_kernel(),
                self.mexh.build_kernel(),
                self.morlet.build_kernel(),
            ),
            dim=0,
        ).squeeze(1)
