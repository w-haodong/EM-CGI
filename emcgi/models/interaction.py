"""CVI: correction-conditioned visual inference shared by visual and structural paths."""
import torch
from torch import nn
from .evidence import CorrectionEvidence, EvidenceModulator, CHANNELS
from .relations import RelationWeights,RelativeCorrections


class CorrectionConditionedVisualInference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        points,hidden,edge_log_scale_limit = cfg['points'],cfg['hidden'],cfg['edge_log_scale_limit']
        self.cfg = cfg
        self.evidence = CorrectionEvidence(points,hidden,cfg)
        self.controllers = nn.ModuleDict({f'stage{s}':nn.ModuleList(EvidenceModulator(c,hidden) for c in CHANNELS[:s]) for s in (2,3,4)})
        self.relations = RelationWeights(hidden,edge_log_scale_limit)
        self.relative_corrections = RelativeCorrections(cfg)

    def condition_features(self,stage,features,history,proposal,*,apply_feedback=True):
        result = self.evidence(features,history,proposal)
        tokens,reference = result['tokens'],result['reference']
        conditioned = [controller(feature,tokens,local,center,gate)
                       for controller,feature,local,center,gate in zip(
                           self.controllers[stage],features,result['local_centers'],
                           result['search_centers'],result['write_gates'])] if apply_feedback else features
        remaining = (~history['clicked']) & history['clicked'].any(1)[:,None]
        count = remaining.sum().clamp_min(1)
        scale = reference.new_tensor([self.cfg['image_hw'][1],self.cfg['image_hw'][0]])
        average = lambda value: (value*remaining).sum()/count
        statistics = dict(evidence_gate=torch.stack([average(gate) for gate in result['evidence_gates']]).mean(),
                          write_gate=torch.stack([average(gate) for gate in result['write_gates']]).mean(),
                          search_shift_px=average(((result['search_origin']-reference)*scale).norm(dim=-1)),
                          matched_shift_px=torch.stack([average(((center-reference)*scale).norm(dim=-1)) for center in result['search_centers']]).mean(),
                          proposal_shift_px=average(((proposal['search_candidate']-reference)*scale).norm(dim=-1)),
                          candidate_entropy=torch.stack([average(-(w.clamp_min(1e-12).log()*w).sum(-1).mean(-1))
                                                         for w in result['candidate_weights']]).mean())
        # These tensors never feed prediction or the training objective.
        diagnostic = {key:torch.stack(result[key],dim=2).detach()
                      for key in ('candidates','candidate_weights','evidence_gates')}
        diagnostic['level_weights'] = result['level_weights'].detach()
        return conditioned,tokens,statistics,diagnostic

    def condition_structure(self,tokens,history,edges):
        return self.relations(tokens,history,edges)


# Import compatibility for existing V132 callers; no duplicate module registration.
AnchorFeatureEncoder = CorrectionConditionedVisualInference
