from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if __package__:
    from .spike_graph import GraphState
else:
    from spike_graph import GraphState


def _inverse_softplus_scalar(value: float) -> float:
    return math.log(math.expm1(float(value)))


class StrictSpatioTemporalGraphLayer(nn.Module):

    def __init__(self, hidden_dim: int = 48, relation_lambda_init: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.self_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.spatial_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temporal_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(6)]
        )
        raw_init = _inverse_softplus_scalar(relation_lambda_init)
        self.spatial_lambda_raw = nn.Parameter(torch.tensor(raw_init))
        self.temporal_lambda_raw = nn.Parameter(torch.full((6,), raw_init))
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.last_diagnostics: Dict[str, Tensor] = {}

    def relation_lambdas(self) -> Dict[str, Tensor]:
        return {
            "spatial": F.softplus(self.spatial_lambda_raw),
            "temporal": F.softplus(self.temporal_lambda_raw),
        }

    def forward(self, state: GraphState) -> GraphState:
        if not isinstance(state, GraphState):
            raise TypeError("StrictSpatioTemporalGraphLayer accepts only GraphState")
        h = state.h
        if h.ndim != 4 or h.shape[-1] != self.hidden_dim:
            raise ValueError("GraphState.h must be (B,W,N,D)")
        batch, windows, nodes, hidden = h.shape
        lambdas = self.relation_lambdas()
        self_message = self.self_linear(h)
        spatial_message = torch.zeros_like(h)
        temporal_message = torch.zeros_like(h)

        if state.spatial_edge_index.numel():
            b, w, source, target = state.spatial_edge_index
            source_values = self.spatial_linear(h[b, w, source])
            weighted = source_values * state.spatial_edge_weight.unsqueeze(-1)
            flat_target = ((b * windows + w) * nodes + target).long()
            spatial_flat = spatial_message.view(batch * windows * nodes, hidden)
            spatial_flat.index_add_(0, flat_target, weighted)

        temporal_components = [torch.zeros_like(h) for _ in range(6)]
        if state.temporal_edge_index.numel():
            if state.temporal_edge_weight is None or state.temporal_edge_type is None:
                raise ValueError("Temporal edges require weights and relation types")
            b, source_w, source_node, target_w, target_node = state.temporal_edge_index
            source_h = h[b, source_w, source_node]
            flat_target = ((b * windows + target_w) * nodes + target_node).long()
            for relation in range(6):
                selected = state.temporal_edge_type == relation
                if not bool(selected.any()):
                    continue
                values = self.temporal_linears[relation](source_h[selected])
                values = values * state.temporal_edge_weight[selected].unsqueeze(-1)
                temporal_components[relation].view(-1, hidden).index_add_(
                    0, flat_target[selected], values
                )
        temporal_components = [
            lambdas["temporal"][index] * value
            for index, value in enumerate(temporal_components)
        ]
        temporal_message = torch.stack(temporal_components).sum(dim=0)
        spatial_component = lambdas["spatial"] * spatial_message
        pre_norm = self_message + spatial_component + temporal_message
        h_next = self.node_norm(F.gelu(pre_norm)) * state.mask.unsqueeze(-1)

        def rms(value: Tensor) -> Tensor:
            return value.square().mean().sqrt().detach()

        self.last_diagnostics = {
            "lambda_spatial": lambdas["spatial"].detach(),
            "lambda_temporal": lambdas["temporal"].detach(),
            "weight_norm_spatial": self.spatial_linear.weight.norm().detach(),
            "message_rms_spatial": rms(spatial_component),
            "message_rms_temporal": torch.stack(
                [rms(value) for value in temporal_components]
            ),
        }
        self.last_diagnostics["weight_norm_temporal"] = torch.stack(
            [layer.weight.norm().detach() for layer in self.temporal_linears]
        )
        return GraphState(
            h=h_next,
            mask=state.mask,
            spatial_edge_index=state.spatial_edge_index,
            spatial_edge_weight=state.spatial_edge_weight,
            temporal_edge_index=state.temporal_edge_index,
            temporal_edge_weight=state.temporal_edge_weight,
            temporal_edge_type=state.temporal_edge_type,
        )


class HiddenStateMixedReadout(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 48,
        *,
        mean_fraction: float = 0.70,
        attention_logit_scale: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mean_fraction = float(mean_fraction)
        self.attention_logit_scale = float(attention_logit_scale)
        self.eps = float(eps)
        if not 0.0 <= self.mean_fraction <= 1.0:
            raise ValueError("mean_fraction must be in [0,1]")
        if self.attention_logit_scale <= 0.0:
            raise ValueError("attention_logit_scale must be positive")
        self.query = nn.Parameter(F.normalize(torch.randn(hidden_dim), dim=0))
        self.last_diagnostics: Dict[str, Tensor] = {}

    def forward(self, state: GraphState) -> Tuple[Tensor, Tensor]:
        if not isinstance(state, GraphState):
            raise TypeError("HiddenStateMixedReadout accepts only GraphState")
        mask = state.mask.to(state.h.dtype)
        numerator = (state.h * mask.unsqueeze(-1)).sum(dim=(1, 2))
        count = mask.sum(dim=(1, 2))
        z_mean = numerator / count.clamp_min(self.eps).unsqueeze(-1)
        all_silent = count.eq(0)
        normalized_h = F.layer_norm(
            state.h, (self.hidden_dim,), weight=None, bias=None
        )
        normalized_h = F.normalize(normalized_h, dim=-1, eps=self.eps)
        normalized_query = F.normalize(self.query, dim=0, eps=self.eps)
        scores = self.attention_logit_scale * torch.einsum(
            "bwnd,d->bwn", normalized_h, normalized_query
        )
        scores = scores.masked_fill(~state.mask, -torch.inf)

        scores = torch.where(
            all_silent.view(-1, 1, 1), torch.zeros_like(scores), scores
        )
        attention = torch.softmax(scores.flatten(1), dim=-1).view_as(scores)
        attention = attention * mask
        z_attention = (
            state.h * attention.unsqueeze(-1)
        ).sum(dim=(1, 2))
        representation = (
            self.mean_fraction * z_mean
            + (1.0 - self.mean_fraction) * z_attention
        )
        representation = representation.masked_fill(all_silent.unsqueeze(-1), 0.0)
        flat_attention = attention.flatten(1)
        entropy = -(
            flat_attention
            * flat_attention.clamp_min(self.eps).log()
        ).sum(dim=-1)
        attention_peak = flat_attention.max(dim=-1).values
        effective_count = entropy.exp()
        attention_peak = attention_peak.masked_fill(all_silent, 0.0)
        effective_count = effective_count.masked_fill(all_silent, 0.0)
        self.last_diagnostics = {
            "readout_valid_state_count": count.detach(),
            "readout_attention_peak": attention_peak.detach(),
            "readout_attention_effective_count": effective_count.detach(),
        }
        return representation, all_silent
