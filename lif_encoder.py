from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _inverse_sigmoid_scalar(value: float) -> float:
    value = min(max(float(value), 1e-7), 1.0 - 1e-7)
    return math.log(value) - math.log1p(-value)


def spacing_normalized_st_spike(
    distance: Tensor,
    spacing: Tensor,
    kappa: Tensor,
    *,
    spacing_eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor]:

    safe_spacing = spacing.detach() + float(spacing_eps)
    normalized_distance = distance / safe_spacing
    soft = torch.sigmoid(kappa * normalized_distance)
    hard = (distance >= 0).to(distance.dtype)
    spikes_st = soft + (hard - soft).detach()
    return spikes_st, hard, soft


@dataclass(slots=True)
class SpikeState:
    spikes_st: Tensor
    spikes_hard: Tensor
    proxy_activation: Optional[Tensor] = None


@dataclass(slots=True)
class ReliableSpikeState:

    spikes_st: Tensor
    spikes_hard: Tensor
    segment_starts_hard: Tensor


class OrderedONOFFLIFEncoder(nn.Module):

    def __init__(
        self,
        num_nodes: int = 48,
        num_thresholds: int = 3,
        beta_min: float = 0.5,
        beta_max: float = 0.99,
        beta_init: float = 0.9,
        spacing_min: float = 0.05,
        spacing_max: float = 1.0,
        theta_init: Sequence[float] = (0.25, 0.50, 0.75),
        kappa_min: float = 1.0,
        kappa_max: float = 8.0,
        kappa_init: Sequence[float] = (4.0, 4.0, 4.0),
        spacing_eps: float = 1e-6,
        initial_membrane: float = 0.0,
        effective_gradient_eps: float = 1e-6,
        threshold_mode: str = "ordered",
    ) -> None:
        super().__init__()
        if num_thresholds not in {1, 2, 3}:
            raise ValueError("Only K=1/K2/K3 encoders are supported")
        if threshold_mode not in {"ordered", "independent"}:
            raise ValueError("threshold_mode must be ordered or independent")
        if not beta_min < beta_init < beta_max:
            raise ValueError("beta_init must lie strictly inside beta bounds")
        if not 0.0 < spacing_min < spacing_max:
            raise ValueError("Invalid threshold-spacing bounds")
        self.num_nodes = int(num_nodes)
        self.num_thresholds = int(num_thresholds)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.spacing_min = float(spacing_min)
        self.spacing_max = float(spacing_max)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.spacing_eps = float(spacing_eps)
        self.initial_membrane = float(initial_membrane)
        self.effective_gradient_eps = float(effective_gradient_eps)
        self.threshold_mode = str(threshold_mode)

        beta_position = (beta_init - beta_min) / (beta_max - beta_min)
        self.beta_raw = nn.Parameter(
            torch.full((self.num_nodes,), _inverse_sigmoid_scalar(beta_position))
        )

        theta = tuple(float(value) for value in theta_init[: self.num_thresholds])
        if len(theta) != self.num_thresholds:
            raise ValueError("theta_init length does not match num_thresholds")
        increments = (
            (theta[0],) + tuple(theta[i] - theta[i - 1] for i in range(1, len(theta)))
            if self.threshold_mode == "ordered" else theta
        )
        if any(not spacing_min < inc < spacing_max for inc in increments):
            raise ValueError("Every initial threshold parameter must lie inside bounds")
        increment_positions = [
            (inc - spacing_min) / (spacing_max - spacing_min) for inc in increments
        ]
        spacing_raw = torch.tensor(
            [_inverse_sigmoid_scalar(value) for value in increment_positions]
        ).expand(self.num_nodes, -1)
        self.threshold_spacing_raw = nn.Parameter(spacing_raw.clone())

        kappa_values = tuple(float(value) for value in kappa_init[: self.num_thresholds])
        if len(kappa_values) != self.num_thresholds:
            raise ValueError("kappa_init length does not match num_thresholds")
        kappa_positions = [
            (value - kappa_min) / (kappa_max - kappa_min) for value in kappa_values
        ]
        self.kappa_raw = nn.Parameter(
            torch.tensor([_inverse_sigmoid_scalar(value) for value in kappa_positions])
        )

    def beta(self) -> Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(
            self.beta_raw
        )

    def fixed_input_gain(self) -> Tensor:

        return self.beta_raw.new_ones((self.num_nodes,))

    def threshold_spacings(self) -> Tensor:
        return self.spacing_min + (self.spacing_max - self.spacing_min) * torch.sigmoid(
            self.threshold_spacing_raw
        )

    def thresholds(self) -> Tensor:
        values = self.threshold_spacings()
        return values.cumsum(dim=-1) if self.threshold_mode == "ordered" else values

    def kappas(self) -> Tensor:
        return self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(
            self.kappa_raw
        )

    def _spike(
        self,
        distance: Tensor,
        spacings: Tensor,
        kappas: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return spacing_normalized_st_spike(
            distance,
            spacings,
            kappas,
            spacing_eps=self.spacing_eps,
        )

    def lif_step(
        self,
        input_t: Tensor,
        membrane: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if input_t.shape != membrane.shape or input_t.ndim != 3:
            raise ValueError("input_t and membrane must both be (B,N,2)")
        beta = self.beta().view(1, self.num_nodes, 1)
        spacings = self.threshold_spacings().view(
            1, self.num_nodes, 1, self.num_thresholds
        )
        thresholds = self.thresholds().view(
            1, self.num_nodes, 1, self.num_thresholds
        )
        kappas = self.kappas().view(1, 1, 1, self.num_thresholds)
        return self._lif_step_with_parameters(
            input_t,
            membrane,
            beta=beta,
            spacings=spacings,
            thresholds=thresholds,
            kappas=kappas,
        )

    def _lif_step_with_parameters(
        self,
        input_t: Tensor,
        membrane: Tensor,
        *,
        beta: Tensor,
        spacings: Tensor,
        thresholds: Tensor,
        kappas: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if input_t.shape != membrane.shape or input_t.ndim != 3:
            raise ValueError("input_t and membrane must both be (B,N,2)")
        expected_threshold_shape = (
            1,
            self.num_nodes,
            1,
            self.num_thresholds,
        )
        if spacings.shape != expected_threshold_shape:
            raise ValueError("Cached threshold spacings have incompatible shape")
        if thresholds.shape != expected_threshold_shape:
            raise ValueError("Cached thresholds have incompatible shape")
        if beta.shape != (1, self.num_nodes, 1):
            raise ValueError("Cached beta has incompatible shape")
        if kappas.shape != (1, 1, 1, self.num_thresholds):
            raise ValueError("Cached kappas have incompatible shape")
        membrane_pre = beta * membrane + input_t
        distance = membrane_pre.unsqueeze(-1) - thresholds
        spikes_st, hard, soft = self._spike(distance, spacings, kappas)
        reset = (
            (hard * spacings).sum(dim=-1)
            if self.threshold_mode == "ordered"
            else (hard * thresholds).amax(dim=-1)
        ).detach()
        membrane_next = F.relu(membrane_pre - reset)
        return membrane_next, membrane_pre, spikes_st, hard, soft

    def forward(
        self, r_down: Tensor
    ) -> Tuple[SpikeState, Dict[str, Tensor], Dict[str, Tensor]]:
        if r_down.ndim != 4 or r_down.shape[1] != self.num_nodes or r_down.shape[-1] != 2:
            raise ValueError(f"Expected r_down=(B,{self.num_nodes},T,2), got {tuple(r_down.shape)}")
        batch_size, _, time_steps, _ = r_down.shape
        membrane = r_down.new_full(
            (batch_size, self.num_nodes, 2), self.initial_membrane
        )
        beta = self.beta().view(1, self.num_nodes, 1)
        spacings_2d = self.threshold_spacings()
        thresholds_2d = self.thresholds()
        kappas_1d = self.kappas()
        spacings = spacings_2d.view(
            1, self.num_nodes, 1, self.num_thresholds
        )
        thresholds = thresholds_2d.view(
            1, self.num_nodes, 1, self.num_thresholds
        )
        kappas = kappas_1d.view(1, 1, 1, self.num_thresholds)
        spike_steps: List[Tensor] = []
        hard_steps: List[Tensor] = []
        soft_steps: List[Tensor] = []
        for time_index in range(time_steps):
            membrane, _, spikes_st, hard, soft = self._lif_step_with_parameters(
                r_down[:, :, time_index, :],
                membrane,
                beta=beta,
                spacings=spacings,
                thresholds=thresholds,
                kappas=kappas,
            )
            spike_steps.append(spikes_st)
            hard_steps.append(hard)
            soft_steps.append(soft)
        spikes_st_all = torch.stack(spike_steps, dim=2)
        spikes_hard = torch.stack(hard_steps, dim=2).detach()
        soft_all = torch.stack(soft_steps, dim=2)
        state = SpikeState(
            spikes_st=spikes_st_all,
            spikes_hard=spikes_hard,
            proxy_activation=soft_all,
        )
        train_aux, diagnostics = self.summarize_state(
            state, final_membrane=membrane
        )
        return state, train_aux, diagnostics

    def summarize_state(
        self,
        state: SpikeState,
        *,
        final_membrane: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        if state.proxy_activation is None:
            raise ValueError("proxy_activation is required for SNN diagnostics")
        spikes_st_all = state.spikes_st
        spikes_hard = state.spikes_hard
        soft_all = state.proxy_activation
        rate_st = spikes_st_all.mean(dim=(0, 1, 2))
        rate_hard = spikes_hard.mean(dim=(0, 1, 2))
        train_aux = {"rate_st": rate_st}
        spacings_2d = self.threshold_spacings()
        thresholds_2d = self.thresholds()
        kappas_1d = self.kappas()
        beta_1d = self.beta()
        local_derivative = (
            kappas_1d.view(1, 1, 1, 1, -1)
            / (
                spacings_2d.detach().view(1, self.num_nodes, 1, 1, -1)
                + self.spacing_eps
            )
            * soft_all
            * (1.0 - soft_all)
        )
        diagnostics = {
            "rate_hard": rate_hard.detach(),
            "node_rate_hard": spikes_hard.mean(dim=(0, 2)).detach(),
            "final_membrane": (
                final_membrane.detach()
                if final_membrane is not None
                else spikes_st_all.new_empty((0,))
            ),
            "threshold_spacings": spacings_2d.detach(),
            "thresholds": thresholds_2d.detach(),
            "kappas": kappas_1d.detach(),
            "beta": beta_1d.detach(),
            "fixed_input_gain": self.fixed_input_gain().detach(),
            "effective_gradient_ratio": (
                local_derivative > self.effective_gradient_eps
            ).float().mean(dim=(0, 1, 2)).detach(),
            "thermometer_violations": (
                (spikes_hard[..., 1:] > spikes_hard[..., :-1]).sum().detach()
                if self.num_thresholds > 1
                else spikes_hard.new_zeros((), dtype=torch.long)
            ),
        }
        return train_aux, diagnostics
