import torch
from torch import nn
from ..compatibility import normalize_config, normalize_model_state
from .visual import MultiScaleVisualEncoder
from .residual import residual_reference,solve_residual
from .interaction import CorrectionConditionedVisualInference
from .geometry import JointConstrainedLocalization,heatmap_unaries,local_heatmap_unaries


class EMCGI(nn.Module):
    def __init__(self, cfg, prior):
        super().__init__()
        cfg = normalize_config(cfg)
        self.cfg = cfg
        self.visual = MultiScaleVisualEncoder(cfg['points'],cfg['independent_heatmaps'])
        self.interaction = CorrectionConditionedVisualInference(cfg)
        self.geometry = JointConstrainedLocalization(prior, cfg['structure_prior_scale'])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(normalize_model_state(state_dict),
                                       strict=strict, assign=assign)

    def forward(self,image,history=None,prior_transform=None):
        last_tokens = None
        stage_statistics = []
        search_diagnostics = []
        active = history is not None and bool(history['clicked'].any())
        proposal = None
        if active:
            if any(k not in history for k in ('reference_q','reference_rhs','automatic_points','automatic_unary_mean','automatic_precision')):
                raise ValueError('V132 history must include the reference returned by its automatic forward')
            proposal = self.geometry.propose(history,self.cfg)
        def condition(stage,features):
            nonlocal last_tokens
            conditioned,last_tokens,statistics,diagnostic = self.interaction.condition_features(stage,features,history,proposal)
            stage_statistics.append(statistics)
            search_diagnostics.append(diagnostic)
            return conditioned
        logits,_,_ = self.visual(image,condition if active else None)
        if self.cfg['independent_heatmaps']:
            means,precision,peaks = local_heatmap_unaries(logits.float().sigmoid(),self.cfg)
        else:
            means,precision,peaks = heatmap_unaries(logits.float().softmax(1)[:,1:],self.cfg)
        if history is None:
            clicked = torch.zeros(means.shape[:2],device=image.device,dtype=torch.bool)
            confirmed = torch.zeros_like(means)
        else:
            clicked,confirmed = history['clicked'],history['confirmed']
        edge_scale = self.interaction.condition_structure(last_tokens,history,self.geometry.edges) if active else None
        relative = None
        if active:
            relative=self.interaction.relative_corrections(last_tokens,history,self.geometry.edges)
            visual_delta=means-history['automatic_unary_mean']
            points=solve_residual(self.geometry,history,visual_delta,relative,
                                  self.cfg['interaction_prior_scale'],edge_scale,prior_transform)
            points=torch.where(clicked.any(1)[:,None,None],points,history['automatic_points'])
            reference={key:history[key] for key in ('reference_q','reference_rhs','automatic_points',
                                                     'automatic_unary_mean','automatic_precision')}
        else:
            points=self.geometry(means,precision,clicked,confirmed,prior_transform=prior_transform)
            reference=residual_reference(self.geometry,points,means,precision,
                                          self.cfg['interaction_prior_scale'],prior_transform)
        statistics = {key:torch.stack([row[key] for row in stage_statistics]).mean()
                      for key in stage_statistics[0]} if stage_statistics else {}
        return dict(logits=logits,points=points,peaks=peaks,edge_scale=edge_scale,
                    relative_correction=relative,edges=self.geometry.edges,
                    unary_mean=means,unary_precision=precision,reference=reference,
                    search_stats=statistics,proposal=proposal,search_diagnostics=search_diagnostics)


# Import compatibility for existing V132 callers; no duplicate module registration.
ARIANet = EMCGI
