from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if __package__:
    from .lif_encoder import ReliableSpikeState
else:
    from lif_encoder import ReliableSpikeState


def segment_match_counts_by_offset(
    left_hard: Tensor,
    right_hard: Tensor,
    offsets: Sequence[int],
    *,
    left_soft: Optional[Tensor] = None,
    right_soft: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    if left_hard.shape != right_hard.shape:
        raise ValueError("Hard segment tensors must share shape")
    if left_hard.ndim < 1:
        raise ValueError("Segment tensors require a time dimension")
    if (left_soft is None) != (right_soft is None):
        raise ValueError("Soft segment tensors must be supplied together")
    if left_soft is not None and (
        left_soft.shape != left_hard.shape or right_soft.shape != right_hard.shape
    ):
        raise ValueError("Hard/soft segment shapes differ")
    length = left_hard.shape[-1]
    hard_counts: List[Tensor] = []
    soft_counts: List[Tensor] = []
    for delta in offsets:
        if delta < 0:
            left_h = left_hard[..., -delta:]
            right_h = right_hard[..., : length + delta]
            left_s = left_soft[..., -delta:] if left_soft is not None else None
            right_s = right_soft[..., : length + delta] if right_soft is not None else None
        elif delta > 0:
            left_h = left_hard[..., : length - delta]
            right_h = right_hard[..., delta:]
            left_s = left_soft[..., : length - delta] if left_soft is not None else None
            right_s = right_soft[..., delta:] if right_soft is not None else None
        else:
            left_h, right_h = left_hard, right_hard
            left_s, right_s = left_soft, right_soft
        hard_counts.append((left_h & right_h).sum(dim=-1))
        if left_s is not None and right_s is not None:
            soft_counts.append((left_s * right_s).sum(dim=-1))
    return (
        torch.stack(hard_counts, dim=-1),
        torch.stack(soft_counts, dim=-1) if soft_counts else None,
    )


@dataclass(slots=True)
class GraphState:
    h: Tensor
    mask: Tensor
    spatial_edge_index: Tensor
    spatial_edge_weight: Tensor
    temporal_edge_index: Tensor
    temporal_edge_weight: Optional[Tensor] = None
    temporal_edge_type: Optional[Tensor] = None


def build_fixed_node_codes(
    channels_per_branch: int = 16,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:

    rows: List[List[float]] = []
    uids: List[int] = []
    for branch in range(3):
        for rank in range(channels_per_branch):
            u = rank / float(channels_per_branch - 1)
            one_hot = [1.0 if branch == index else 0.0 for index in range(3)]
            rows.append(
                one_hot
                + [
                    u,
                    math.sin(2.0 * math.pi * u),
                    math.cos(2.0 * math.pi * u),
                    math.sin(4.0 * math.pi * u),
                    math.cos(4.0 * math.pi * u),
                ]
            )
            uids.append(branch * channels_per_branch + rank)
    return (
        torch.tensor(rows, device=device, dtype=dtype),
        torch.tensor(uids, device=device, dtype=torch.long),
    )


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def symmetric_pair_hash(uid_a: int, uid_b: int, seed: int) -> int:
    low, high = sorted((int(uid_a), int(uid_b)))
    packed = ((low & 0xFFFFFFFF) << 32) | (high & 0xFFFFFFFF)
    return _splitmix64(packed ^ (int(seed) & ((1 << 64) - 1)))


def pair_time_support(q: Tensor) -> Tensor:

    if q.ndim != 2:
        raise ValueError("q must be (N,L)")

    q_count = q.to(torch.float32)
    return (q_count @ q_count.transpose(0, 1)).to(torch.int64)


def build_node_mask(spikes_hard: Tensor, window_length: int = 16) -> Tensor:

    if spikes_hard.ndim != 5:
        raise ValueError("spikes_hard must be (B,N,T,P,K)")
    batch, nodes, time_steps, polarities, thresholds = spikes_hard.shape
    if time_steps % window_length:
        raise ValueError("Spike time length must be divisible by window_length")
    windows = time_steps // window_length
    blocks = (
        spikes_hard.detach()
        .view(batch, nodes, windows, window_length, polarities * thresholds)
        .permute(0, 2, 1, 3, 4)
    )
    return blocks.sum(dim=(-1, -2)).gt(0).detach()


def lagged_pair_support(q: Tensor, radius: int = 2) -> Tensor:
    if q.ndim != 2:
        raise ValueError("q must be (N,L)")
    q_count = q.to(torch.float32)
    nodes, length = q.shape
    result = torch.zeros((nodes, nodes), dtype=torch.int64, device=q.device)
    for delta in range(-int(radius), int(radius) + 1):
        if delta < 0:
            score = q_count[:, -delta:] @ q_count[:, : length + delta].transpose(0, 1)
        elif delta > 0:
            score = q_count[:, : length - delta] @ q_count[:, delta:].transpose(0, 1)
        else:
            score = q_count @ q_count.transpose(0, 1)
        result = torch.maximum(result, score.to(torch.int64))
    return result


class SpikeGraphBuilder(nn.Module):

    def __init__(
        self,
        num_nodes: int = 48,
        num_thresholds: int = 3,
        window_length: int = 16,
        hidden_dim: int = 48,
        topk: int = 6,
        min_time_support: int = 2,
        lag_radius: int = 2,
        pair_hash_seed: int = 20240725,
        use_node_identity: bool = True,
        temporal_direction: str = "bidirectional_offline",
        temporal_lags: Sequence[int] = tuple(range(1, 9)),
        temporal_offsets: Sequence[int] = (-2, -1, 0, 1, 2),
        temporal_min_segments: int = 1,
        temporal_min_ranks: int = 2,
        channels_per_branch: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_thresholds = int(num_thresholds)
        self.window_length = int(window_length)
        self.spike_channels = 2 * self.num_thresholds
        self.hidden_dim = int(hidden_dim)
        self.topk = int(topk)
        self.min_time_support = int(min_time_support)
        self.lag_radius = int(lag_radius)
        self.pair_hash_seed = int(pair_hash_seed)
        self.use_node_identity = bool(use_node_identity)
        self.temporal_direction = str(temporal_direction)
        self.temporal_lags = tuple(int(value) for value in temporal_lags)
        self.temporal_offsets = tuple(int(value) for value in temporal_offsets)
        self.temporal_min_segments = int(temporal_min_segments)
        self.temporal_min_ranks = int(temporal_min_ranks)
        self.channels_per_branch = int(channels_per_branch)
        self.eps = float(eps)
        self.spike_projection = nn.Linear(
            self.window_length * self.spike_channels, self.hidden_dim, bias=True
        )
        self.structure_projection = nn.Linear(8, self.hidden_dim, bias=False)
        self.node_norm = nn.LayerNorm(self.hidden_dim)
        codes, uids = build_fixed_node_codes(self.num_nodes // 3)
        self.register_buffer("node_structure_codes", codes, persistent=True)
        self.register_buffer("stable_node_uids", uids, persistent=True)
        hash_values = [
            [
                symmetric_pair_hash(uid_a, uid_b, self.pair_hash_seed)
                for uid_b in range(self.num_nodes)
            ]
            for uid_a in range(self.num_nodes)
        ]
        ordered_hashes = sorted({value for row in hash_values for value in row})
        hash_to_rank = {value: rank for rank, value in enumerate(ordered_hashes)}
        pair_hash_rank = torch.tensor(
            [
                [hash_to_rank[value] for value in row]
                for row in hash_values
            ],
            dtype=torch.long,
        )

        self.register_buffer(
            "pair_hash_rank_by_uid", pair_hash_rank, persistent=False
        )

    @staticmethod
    def _empty_edges(device: torch.device) -> Tensor:
        return torch.empty((4, 0), dtype=torch.long, device=device)

    @staticmethod
    def _empty_temporal_edges(device: torch.device) -> Tensor:
        return torch.empty((5, 0), dtype=torch.long, device=device)

    def project_spike_windows(
        self, spikes_st: Tensor, spikes_hard: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if spikes_st.shape != spikes_hard.shape or spikes_st.ndim != 5:
            raise ValueError("Spike tensors must share shape (B,N,T,2,K)")
        batch, nodes, time_steps, polarities, thresholds = spikes_st.shape
        if nodes != self.num_nodes or polarities != 2 or thresholds != self.num_thresholds:
            raise ValueError("Spike tensor dimensions do not match graph builder")
        if time_steps % self.window_length:
            raise ValueError("Spike time length must be divisible by graph window length")
        windows = time_steps // self.window_length

        st_blocks = (
            spikes_st.view(batch, nodes, windows, self.window_length, self.spike_channels)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        hard_blocks = (
            spikes_hard.view(batch, nodes, windows, self.window_length, self.spike_channels)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        flat_st = st_blocks.flatten(start_dim=-2)
        flat_hard = hard_blocks.flatten(start_dim=-2)
        mask = build_node_mask(spikes_hard, self.window_length)
        spike_part = self.spike_projection(flat_st)
        if self.use_node_identity:
            structure_part = self.structure_projection(self.node_structure_codes)
        else:

            structure_part = spike_part.new_zeros((self.num_nodes, self.hidden_dim))
        pre_norm = F.gelu(
            spike_part + structure_part.view(1, 1, self.num_nodes, self.hidden_dim)
        )
        h0 = self.node_norm(pre_norm) * mask.unsqueeze(-1)
        return h0, mask, flat_st, hard_blocks

    @staticmethod
    def selected_edge_weights(
        similarity: Tensor,
        undirected_edges: Sequence[Tuple[int, int]],
        eps: float,
    ) -> Tuple[List[Tuple[int, int]], List[Tensor]]:

        directed: List[Tuple[int, int]] = []
        weights: List[Tensor] = []
        neighbors_by_target: Dict[int, List[int]] = {}
        for node, other in undirected_edges:
            neighbors_by_target.setdefault(node, []).append(other)
            neighbors_by_target.setdefault(other, []).append(node)
        for target, sources in neighbors_by_target.items():
            denominator = (
                torch.stack([similarity[target, source] for source in sources]).sum()
                + eps
            )
            for source in sources:
                directed.append((source, target))
                weights.append(similarity[target, source] / denominator)
        return directed, weights

    def mutual_topk(
        self,
        flat_st: Tensor,
        hard_block: Tensor,
        mask: Tensor,
        stable_uids: Tensor,
    ) -> Tuple[List[Tuple[int, int]], List[Tensor], Dict[str, int]]:

        q = hard_block.sum(dim=-1).gt(0)
        support = pair_time_support(q)
        lag_support = lagged_pair_support(q, self.lag_radius)
        counts = flat_st.sum(dim=-1)
        similarity = (flat_st @ flat_st.transpose(0, 1)) / (
            (counts[:, None] * counts[None, :] + self.eps).sqrt()
        )

        similarity_values = similarity.detach().cpu().tolist()
        support_values = support.detach().cpu().tolist()
        lag_values = lag_support.detach().cpu().tolist()
        mask_values = mask.detach().cpu().tolist()
        uid_values = stable_uids.detach().cpu().tolist()
        candidates: List[List[int]] = [[] for _ in range(self.num_nodes)]
        selected: List[set[int]] = [set() for _ in range(self.num_nodes)]
        boundary_ties = 0
        hash_tied_candidates = 0
        candidate_count = 0
        for node in range(self.num_nodes):
            if not mask_values[node]:
                continue
            for other in range(self.num_nodes):
                if (
                    other != node
                    and mask_values[other]
                    and support_values[node][other] >= self.min_time_support
                    and similarity_values[node][other] > 0.0
                ):
                    candidates[node].append(other)
            candidate_count += len(candidates[node])
            triples: Dict[Tuple[float, int, int], int] = {}
            for other in candidates[node]:
                triple = (
                    similarity_values[node][other],
                    support_values[node][other],
                    lag_values[node][other],
                )
                triples[triple] = triples.get(triple, 0) + 1
            hash_tied_candidates += sum(count for count in triples.values() if count > 1)
            ordered = sorted(
                candidates[node],
                key=lambda other: (
                    similarity_values[node][other],
                    support_values[node][other],
                    lag_values[node][other],
                    symmetric_pair_hash(
                        uid_values[node],
                        uid_values[other],
                        self.pair_hash_seed,
                    ),
                ),
                reverse=True,
            )
            if len(ordered) > self.topk:
                kth = ordered[self.topk - 1]
                next_item = ordered[self.topk]
                kth_triple = (
                    similarity_values[node][kth],
                    support_values[node][kth],
                    lag_values[node][kth],
                )
                next_triple = (
                    similarity_values[node][next_item],
                    support_values[node][next_item],
                    lag_values[node][next_item],
                )
                boundary_ties += int(kth_triple == next_triple)
            selected[node].update(ordered[: self.topk])

        undirected: List[Tuple[int, int]] = []
        for node in range(self.num_nodes):
            for other in selected[node]:
                if node < other and node in selected[other]:
                    undirected.append((node, other))

        directed, weights = self.selected_edge_weights(
            similarity, undirected, self.eps
        )
        diagnostics = {
            "candidate_count": candidate_count,
            "boundary_ties": boundary_ties,
            "hash_tied_candidates": hash_tied_candidates,
            "undirected_edges": len(undirected),
        }
        return directed, weights, diagnostics

    def _lag_scale(self, lag: int) -> int:

        lag_steps = int(lag) * self.window_length
        if lag_steps <= 32:
            return 0
        if lag_steps <= 80:
            return 1
        return 2

    def build_temporal_edges(
        self,
        mask: Tensor,
        segment_starts_hard: Tensor,
        spikes_st: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        batch, windows, nodes = mask.shape
        if segment_starts_hard.shape[0] != batch or segment_starts_hard.shape[1] != nodes:
            raise ValueError("segment_starts_hard has incompatible shape")
        starts = segment_starts_hard.reshape(
            batch, nodes, windows, self.window_length
        ).permute(0, 2, 1, 3).detach().bool()

        xi = spikes_st.mean(dim=(-1, -2)).reshape(
            batch, nodes, windows, self.window_length
        ).permute(0, 2, 1, 3)
        all_indices: List[Tensor] = []
        all_weights: List[Tensor] = []
        all_types: List[Tensor] = []
        all_lags: List[Tensor] = []
        offset_priority = sorted(self.temporal_offsets, key=lambda value: (abs(value), value))
        for lag in self.temporal_lags:
            if lag >= windows:
                continue
            left = starts[:, :-lag]
            right = starts[:, lag:]
            match_stack, _ = segment_match_counts_by_offset(
                left, right, offset_priority
            )
            best_match, best_position = match_stack.max(dim=-1)
            node_valid = best_match >= self.temporal_min_segments
            rank_valid = node_valid.reshape(
                batch, windows - lag, 3, self.channels_per_branch
            ).any(dim=2)
            enough_ranks = rank_valid.sum(dim=-1) >= self.temporal_min_ranks
            endpoint_valid = mask[:, :-lag] & mask[:, lag:]
            valid = node_valid & endpoint_valid & enough_ranks.unsqueeze(-1)
            selected = valid.nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            b, left_w, node = selected.unbind(dim=1)
            right_w = left_w + lag
            chosen_position = best_position[b, left_w, node]
            score = xi.new_zeros((selected.shape[0],))
            for position, delta in enumerate(offset_priority):
                choose = chosen_position == position
                if not bool(choose.any()):
                    continue
                cb, clw, cn = b[choose], left_w[choose], node[choose]
                rw = clw + lag
                left_x = xi[cb, clw, cn]
                right_x = xi[cb, rw, cn]
                if delta < 0:
                    left_x = left_x[..., -delta:]
                    right_x = right_x[..., : self.window_length + delta]
                elif delta > 0:
                    left_x = left_x[..., : self.window_length - delta]
                    right_x = right_x[..., delta:]
                numerator = (left_x * right_x).sum(dim=-1)
                denominator = (
                    left_x.square().sum(dim=-1)
                    * right_x.square().sum(dim=-1)
                ).sqrt().clamp_min(self.eps)
                score[choose] = numerator / denominator
            score = score.clamp_min(0.0)
            scale = self._lag_scale(lag)
            forward_index = torch.stack((b, left_w, node, right_w, node), dim=0)
            all_indices.append(forward_index)
            all_weights.append(score)
            all_types.append(
                torch.full_like(score, scale, dtype=torch.long)
            )
            all_lags.append(torch.full_like(score, lag, dtype=torch.long))
            if self.temporal_direction == "bidirectional_offline":
                all_indices.append(
                    torch.stack((b, right_w, node, left_w, node), dim=0)
                )
                all_weights.append(score)
                all_types.append(
                    torch.full_like(score, 3 + scale, dtype=torch.long)
                )
                all_lags.append(torch.full_like(score, lag, dtype=torch.long))
        if not all_indices:
            return (
                self._empty_temporal_edges(mask.device),
                spikes_st.new_empty((0,)),
                torch.empty((0,), dtype=torch.long, device=mask.device),
            )
        edge_index = torch.cat(all_indices, dim=1)
        raw_weight = torch.cat(all_weights)
        edge_type = torch.cat(all_types)
        edge_lag = torch.cat(all_lags)

        b, _, source_node, target_w, target_node = edge_index
        group = (
            ((b * windows + target_w) * nodes + target_node) * 6 + edge_type
        ).long()
        order = torch.arange(group.numel(), device=group.device)
        for key, descending in (
            (source_node, False),
            (edge_lag, False),
            (raw_weight.detach(), True),
            (group, False),
        ):
            local = torch.argsort(
                key[order], descending=descending, stable=True
            )
            order = order[local]
        ordered_group = group[order]
        keep_ordered = torch.ones_like(ordered_group, dtype=torch.bool)
        keep_ordered[1:] = ordered_group[1:] != ordered_group[:-1]
        keep = order[keep_ordered]
        edge_index = edge_index[:, keep]
        raw_weight = raw_weight[keep]
        edge_type = edge_type[keep]
        b, _, _, target_w, target_node = edge_index
        flat_target = ((b * windows + target_w) * nodes + target_node).long()
        denominator = raw_weight.new_zeros((batch * windows * nodes,))
        denominator.index_add_(0, flat_target, raw_weight)
        edge_weight = raw_weight / denominator[flat_target].clamp_min(self.eps)
        return edge_index, edge_weight, edge_type

    @staticmethod
    def _stable_descending_reorder(order: Tensor, key: Tensor) -> Tensor:

        ordered_key = torch.gather(key.expand_as(order), dim=-1, index=order)
        local_order = torch.argsort(
            ordered_key,
            dim=-1,
            descending=True,
            stable=True,
        )
        return torch.gather(order, dim=-1, index=local_order)

    def _gpu_batched_spatial_edges(
        self,
        flat_st: Tensor,
        hard_blocks: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:

        batch, windows, nodes, _ = flat_st.shape
        q = hard_blocks.detach().sum(dim=-1).gt(0).to(torch.float32)
        support = torch.matmul(q, q.transpose(-1, -2)).to(torch.int64)
        lag_support = torch.zeros_like(support)
        length = q.shape[-1]
        for delta in range(-self.lag_radius, self.lag_radius + 1):
            if delta < 0:
                score = torch.matmul(
                    q[..., -delta:],
                    q[..., : length + delta].transpose(-1, -2),
                )
            elif delta > 0:
                score = torch.matmul(
                    q[..., : length - delta],
                    q[..., delta:].transpose(-1, -2),
                )
            else:
                score = torch.matmul(q, q.transpose(-1, -2))
            lag_support = torch.maximum(lag_support, score.to(torch.int64))

        counts = flat_st.sum(dim=-1)
        similarity = torch.matmul(flat_st, flat_st.transpose(-1, -2)) / (
            (counts.unsqueeze(-1) * counts.unsqueeze(-2) + self.eps).sqrt()
        )
        similarity_detached = similarity.detach()
        diagonal = torch.eye(nodes, dtype=torch.bool, device=flat_st.device).view(
            1, 1, nodes, nodes
        )
        candidate = (
            mask.unsqueeze(-1)
            & mask.unsqueeze(-2)
            & ~diagonal
            & (support >= self.min_time_support)
            & (similarity_detached > 0.0)
        )

        physical_hash_rank = self.pair_hash_rank_by_uid[
            self.stable_node_uids[:, None],
            self.stable_node_uids[None, :],
        ].view(1, 1, nodes, nodes)
        invalid_long = torch.full_like(support, -1)
        hash_key = torch.where(
            candidate,
            physical_hash_rank.expand(batch, windows, nodes, nodes),
            invalid_long,
        )
        lag_key = torch.where(candidate, lag_support, invalid_long)
        support_key = torch.where(candidate, support, invalid_long)
        similarity_key = torch.where(
            candidate,
            similarity_detached,
            torch.full_like(similarity_detached, -torch.inf),
        )

        order = torch.arange(nodes, device=flat_st.device, dtype=torch.long).view(
            1, 1, 1, nodes
        ).expand(batch, windows, nodes, nodes)

        order = self._stable_descending_reorder(order, hash_key)
        order = self._stable_descending_reorder(order, lag_key)
        order = self._stable_descending_reorder(order, support_key)
        order = self._stable_descending_reorder(order, similarity_key)

        chosen = order[..., : self.topk]
        chosen_valid = torch.gather(candidate, dim=-1, index=chosen)
        selected = torch.zeros_like(candidate).scatter(
            dim=-1,
            index=chosen,
            src=chosen_valid,
        )
        mutual = selected & selected.transpose(-1, -2)

        candidate_counts = candidate.sum(axis=-1)
        kth = order[..., self.topk - 1]
        next_item = order[..., self.topk]
        kth_valid = candidate_counts > self.topk
        kth_sim = torch.gather(similarity_detached, -1, kth.unsqueeze(-1)).squeeze(-1)
        next_sim = torch.gather(
            similarity_detached, -1, next_item.unsqueeze(-1)
        ).squeeze(-1)
        kth_support = torch.gather(support, -1, kth.unsqueeze(-1)).squeeze(-1)
        next_support = torch.gather(
            support, -1, next_item.unsqueeze(-1)
        ).squeeze(-1)
        kth_lag = torch.gather(lag_support, -1, kth.unsqueeze(-1)).squeeze(-1)
        next_lag = torch.gather(
            lag_support, -1, next_item.unsqueeze(-1)
        ).squeeze(-1)
        boundary_ties = (
            kth_valid
            & (kth_sim == next_sim)
            & (kth_support == next_support)
            & (kth_lag == next_lag)
        ).sum()

        ordered_sim = torch.gather(similarity_detached, -1, order)
        ordered_support = torch.gather(support, -1, order)
        ordered_lag = torch.gather(lag_support, -1, order)
        ordered_valid = torch.gather(candidate, -1, order)
        same_adjacent = (
            ordered_valid[..., :-1]
            & ordered_valid[..., 1:]
            & (ordered_sim[..., :-1] == ordered_sim[..., 1:])
            & (ordered_support[..., :-1] == ordered_support[..., 1:])
            & (ordered_lag[..., :-1] == ordered_lag[..., 1:])
        )
        tied_sorted = torch.zeros_like(ordered_valid)
        tied_sorted[..., :-1] |= same_adjacent
        tied_sorted[..., 1:] |= same_adjacent
        hash_tied_candidates = tied_sorted.sum()

        target_index = mutual.nonzero(as_tuple=False)
        if target_index.numel() == 0:
            return (
                self._empty_edges(flat_st.device),
                flat_st.new_empty((0,)),
                {
                    "candidate_count": candidate.sum(),
                    "boundary_ties": boundary_ties,
                    "hash_tied_candidates": hash_tied_candidates,
                    "undirected_edges": mutual.sum(),
                },
            )
        batch_index, window_index, target, source = target_index.unbind(dim=1)
        edge_index = torch.stack(
            (batch_index, window_index, source, target), dim=0
        )
        denominator = (
            similarity * mutual.to(dtype=similarity.dtype)
        ).sum(dim=-1) + self.eps
        edge_weight = similarity[
            edge_index[0], edge_index[1], edge_index[3], edge_index[2]
        ] / denominator[edge_index[0], edge_index[1], edge_index[3]]
        return edge_index, edge_weight, {
            "candidate_count": candidate.sum(),
            "boundary_ties": boundary_ties,
            "hash_tied_candidates": hash_tied_candidates,
            "undirected_edges": mutual.sum() // 2,
        }

    def forward(
        self, spike_state: ReliableSpikeState
    ) -> Tuple[GraphState, Dict[str, Tensor]]:
        if not isinstance(spike_state, ReliableSpikeState):
            raise TypeError("SpikeGraphBuilder accepts only ReliableSpikeState")
        h0, mask, flat_st, hard_blocks = self.project_spike_windows(
            spike_state.spikes_st, spike_state.spikes_hard.detach()
        )
        spatial_edge_index, spatial_edge_weight, stats = self._gpu_batched_spatial_edges(
            flat_st, hard_blocks, mask
        )
        (
            temporal_edge_index,
            temporal_edge_weight,
            temporal_edge_type,
        ) = self.build_temporal_edges(
            mask,
            spike_state.segment_starts_hard,
            spike_state.spikes_st,
        )
        state = GraphState(
            h=h0,
            mask=mask,
            spatial_edge_index=spatial_edge_index,
            spatial_edge_weight=spatial_edge_weight,
            temporal_edge_index=temporal_edge_index,
            temporal_edge_weight=temporal_edge_weight,
            temporal_edge_type=temporal_edge_type,
        )
        active_nodes = mask.sum()
        diagnostics = {
            "active_nodes": active_nodes.detach(),
            "spatial_directed_edges": torch.tensor(
                spatial_edge_index.shape[1], device=h0.device
            ),
            "spatial_undirected_edges": stats["undirected_edges"],
            "temporal_edges": torch.tensor(
                temporal_edge_index.shape[1], device=h0.device
            ),
            "topk_candidate_count": stats["candidate_count"],
            "topk_boundary_ties": stats["boundary_ties"],
            "pair_hash_tied_candidates": stats["hash_tied_candidates"],
        }
        return state, {key: value.detach() for key, value in diagnostics.items()}


