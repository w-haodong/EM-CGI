"""Independent search followed by visual candidate matching.

Search locations never depend on acceptance gates. Matched evidence updates
features; neither proposals nor attention centroids replace output landmarks.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

CHANNELS = (18, 36, 72, 144)


def sample(feature, xy):
    h, w = feature.shape[-2:]
    grid = 2 * (xy + xy.new_tensor([.5 / w, .5 / h])) - 1
    value = F.grid_sample(feature, grid, align_corners=False, padding_mode='border')
    return value.permute(0, 2, 3, 1)


class CandidateMatcher(nn.Module):
    """Score each candidate using its sampled appearance and structural prior."""

    def __init__(self, hidden, cfg):
        super().__init__()
        self.hidden = hidden
        self.radius = cfg['search_radius_cap_px']
        self.prior_limit = cfg['candidate_prior_limit']
        self.energy_cap = cfg['candidate_energy_cap']
        self.query = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.key = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.offset = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.prior_strength = nn.Linear(hidden, 1)
        nn.init.zeros_(self.prior_strength.weight)
        nn.init.constant_(self.prior_strength.bias, cfg['candidate_prior_bias'])

    def forward(self, query, sampled, xy, reference, candidate, ellipse, pixel_scale):
        # Read each candidate BEFORE deciding its weight.
        if getattr(self, 'use_appearance_score', True):
            content = (self.query(query)[:, :, None].float() *
                       self.key(sampled).float()).sum(-1) / math.sqrt(self.hidden)
        else:
            # Ablation hook: retain sampled evidence, position and prior scores.
            content = xy.new_zeros(xy.shape[:-1], dtype=torch.float32)
        to_current = (xy - reference[:, :, None]) * pixel_scale
        to_candidate = (xy - candidate[:, :, None]) * pixel_scale
        offsets = torch.cat((to_current, to_candidate), -1) / self.radius
        position = self.offset(offsets).squeeze(-1).float()
        # ellipse is the capped SPD square root in pixel units for this level.
        with torch.autocast(device_type=xy.device.type, enabled=False):
            whitened = torch.linalg.solve(
                ellipse.float(), to_candidate.float().transpose(-1, -2)
            ).transpose(-1, -2)
            energy = .5 * whitened.square().sum(-1).clamp_max(self.energy_cap)
        strength = self.prior_limit * self.prior_strength(query).float().sigmoid()
        scores = content + position - strength * energy
        weights = scores.softmax(-1)
        evidence = (sampled * weights[..., None].to(sampled.dtype)).sum(2)
        center = (xy * weights[..., None]).sum(2)
        support = torch.logsumexp(scores, -1) - math.log(scores.shape[-1])
        return evidence, center, weights, support


def acceptance_head(hidden, bias):
    head = nn.Sequential(nn.Linear(hidden * 4 + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
    nn.init.normal_(head[-1].weight, std=1e-3)
    nn.init.constant_(head[-1].bias, bias)
    return head


class CorrectionEvidence(nn.Module):
    def __init__(self, points, hidden, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden = hidden
        self.identity = nn.Embedding(points, hidden)
        self.level_identity = nn.Embedding(4, hidden)
        self.position = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.projection = nn.ModuleList(nn.Linear(c, hidden, bias=False) for c in CHANNELS)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.anchor_fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.level_weights = nn.Linear(hidden, 4)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.geometry_embedding = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.matcher = CandidateMatcher(hidden, cfg)
        self.evidence_gate = acceptance_head(hidden, cfg['evidence_gate_bias'])
        self.write_gate = acceptance_head(hidden, cfg['write_gate_bias'])

    def forward(self, features, history, proposal):
        current, old, new, clicked = (history[k] for k in ('current', 'old', 'confirmed', 'clicked'))
        _, k, _ = current.shape
        active = clicked.any(1)
        reference = torch.where(clicked[..., None], new, current)
        candidate = proposal['search_candidate'].to(reference)
        if not getattr(self, 'use_search_shift', True):
            # Keep the same ellipse and candidate count; change its center only.
            candidate = reference
        basis = proposal['basis_px'].to(reference)
        pixel_scale = current.new_tensor(self.cfg['image_hw'][::-1])
        displacement = (new - old) * clicked[..., None]
        state = torch.cat((current, reference, displacement, clicked[..., None].float()), -1)
        base = self.position(state) + self.identity(torch.arange(k, device=current.device))[None]
        geometry_state = torch.cat((candidate - reference,
                                    torch.stack((basis[..., 0, 0], basis[..., 0, 1], basis[..., 1, 1]), -1)
                                    / self.cfg['search_radius_cap_px']), -1)
        base = base + self.geometry_embedding(geometry_state)
        level_weight = self.level_weights(base)[..., :len(features)].softmax(-1)
        visual, contrast = torch.zeros_like(base), torch.zeros_like(base)
        for level, feature in enumerate(features):
            at_current = self.projection[level](sample(feature, reference[:, :, None]).squeeze(2))
            if getattr(self, 'use_appearance_contrast', True):
                at_old = self.projection[level](sample(feature, old[:, :, None]).squeeze(2))
            visual = visual + level_weight[..., level, None] * at_current
            if getattr(self, 'use_appearance_contrast', True):
                contrast = contrast + level_weight[..., level, None] * (at_current - at_old) * clicked[..., None]
        receiver = base + visual
        anchor = self.anchor_fusion(torch.cat((receiver, contrast), -1))
        scores = self.query(receiver) @ self.key(anchor).transpose(1, 2) / math.sqrt(self.hidden)
        safe = clicked | ((~active)[:, None] & (torch.arange(k, device=current.device)[None] == 0))
        attention = scores.masked_fill(~safe[:, None], -1e4).softmax(-1) * active[:, None, None]
        context = attention @ self.value(anchor)
        angles = current.new_tensor([2 * math.pi * i / 8 for i in range(8)])
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        directions = torch.cat((torch.zeros_like(directions[:1]), directions), 0)
        evidence = torch.zeros_like(base)
        local_centers, search_centers, evidence_gates, write_gates = [], [], [], []
        candidates, candidate_weights = [], []
        for level, feature in enumerate(features):
            ellipse = basis * self.cfg['search_scale_factors'][level]
            radius = torch.linalg.matrix_norm(ellipse, ord=2)
            ellipse = ellipse * (self.cfg['search_radius_cap_px'] / radius.clamp_min(1e-6)).clamp_max(1.)[..., None, None]
            offsets = torch.einsum('bkij,mj->bkmi', ellipse, directions)
            # Full neighborhoods in BOTH branches; acceptance never shrinks
            # the structural search center towards the current prediction.
            search_xy = (candidate[:, :, None] + offsets / pixel_scale).clamp(0., 1.)
            local_xy = (reference[:, :, None] + directions * (8. * 2 ** level) / pixel_scale).clamp(0., 1.)
            searched = self.projection[level](sample(feature, search_xy))
            local = self.projection[level](sample(feature, local_xy))
            query = receiver + context + self.level_identity.weight[level]
            local_evidence, local_center, local_weights, local_support = self.matcher(
                query, local, local_xy, reference, candidate, ellipse, pixel_scale)
            search_evidence, search_center, search_weights, search_support = self.matcher(
                query, searched, search_xy, reference, candidate, ellipse, pixel_scale)
            acceptance = torch.cat((receiver, context, local_evidence, search_evidence,
                                    local_support[..., None], search_support[..., None]), -1)
            evidence_gate = self.evidence_gate(acceptance).squeeze(-1).sigmoid() * active[:, None]
            write_gate = self.write_gate(acceptance).squeeze(-1).sigmoid() * active[:, None]
            selected = local_evidence * (1 - evidence_gate[..., None]) + search_evidence * evidence_gate[..., None]
            evidence = evidence + level_weight[..., level, None] * selected
            local_centers.append(local_center)
            search_centers.append(search_center)
            evidence_gates.append(evidence_gate)
            write_gates.append(write_gate)
            candidates.append(torch.stack((local_xy, search_xy), 2))
            candidate_weights.append(torch.stack((local_weights, search_weights), 2))
        tokens = self.fusion(torch.cat((receiver, context, evidence), -1)) * active[:, None, None]
        return dict(tokens=tokens, reference=reference, search_origin=candidate,
                    local_centers=local_centers, search_centers=search_centers,
                    evidence_gates=evidence_gates, write_gates=write_gates,
                    candidates=candidates, candidate_weights=candidate_weights,
                    level_weights=level_weight)


class EvidenceModulator(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.spatial = nn.Linear(hidden, channels, bias=False)
        self.channel = nn.Linear(hidden, channels * 2, bias=False)
        nn.init.normal_(self.spatial.weight, std=1e-3)
        nn.init.normal_(self.channel.weight, std=1e-3)

    def forward(self, feature, tokens, local_center, search_center, write_gate):
        h, w = feature.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=feature.device), torch.arange(w, device=feature.device), indexing='ij')
        scale_xy = local_center.new_tensor([w, h])
        local, searched = local_center * scale_xy, search_center * scale_xy
        local_maps = torch.exp(-((xx - local[..., 0, None, None]).square() + (yy - local[..., 1, None, None]).square()) / 8.)
        search_maps = torch.exp(-((xx - searched[..., 0, None, None]).square() + (yy - searched[..., 1, None, None]).square()) / 8.)
        maps = (1 - write_gate[..., None, None]) * local_maps + write_gate[..., None, None] * search_maps
        maps = maps / maps.sum(1, keepdim=True).clamp_min(1.)
        spatial = torch.einsum('bkc,bkhw->bchw', self.spatial(tokens), maps)
        scale, shift = self.channel(tokens.mean(1)).chunk(2, -1)
        return feature * (1 + .1 * scale.tanh()[..., None, None]) + .1 * shift[..., None, None] + .1 * spatial


# Import compatibility for existing V132 callers; no duplicate module registration.
AnchorEvidence = CorrectionEvidence
