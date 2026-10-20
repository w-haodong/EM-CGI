"""Click-conditioned positive edge strengths; train-set displacement means fixed."""
import torch
from torch import nn


class RelationWeights(nn.Module):
    def __init__(self, hidden, log_scale_limit=.5):
        super().__init__()
        self.limit = log_scale_limit
        self.edge = nn.Sequential(nn.Linear(hidden*2+6,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)

    def forward(self, tokens, history, edges):
        i,j = edges[:,0],edges[:,1]
        current,clicked = history['current'],history['clicked']
        displacement = (history['confirmed']-history['old'])*clicked[...,None]
        state = torch.cat((tokens[:,i],tokens[:,j],current[:,i]-current[:,j],
                           clicked[:,i,None].float(),clicked[:,j,None].float(),
                           displacement[:,i]-displacement[:,j]),-1)
        log_scale = self.limit*self.edge(state).squeeze(-1).tanh()*clicked.any(1)[:,None]
        return log_scale.exp()


class RelativeCorrections(nn.Module):
    """Predict unequal neighboring corrections from image tokens and clicks."""
    def __init__(self,cfg):
        super().__init__()
        hidden=cfg['hidden']
        self.edge=nn.Sequential(nn.Linear(hidden*2+6,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.edge[-1].weight);nn.init.zeros_(self.edge[-1].bias)
        self.limit=cfg['relative_shift_limit_px']
        self.image_hw=cfg['image_hw']

    def forward(self,tokens,history,edges):
        i,j=edges[:,0],edges[:,1]
        clicked=history['clicked']
        origin=history['automatic_points'].to(tokens)
        correction=(history['confirmed'].to(tokens)-origin)*clicked[...,None]
        state=torch.cat((tokens[:,i],tokens[:,j],origin[:,i]-origin[:,j],
                         clicked[:,i,None].to(tokens),clicked[:,j,None].to(tokens),
                         correction[:,i]-correction[:,j]),-1)
        scale=tokens.new_tensor(self.image_hw[::-1])
        return self.edge(state).tanh()*self.limit/scale*clicked.any(1)[:,None,None]
