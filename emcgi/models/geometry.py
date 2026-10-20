"""JCL: joint constrained localization by differentiable Gaussian conditioning."""
import torch
from torch import nn


def heatmap_unaries(probabilities, cfg):
    b,k,h,w = probabilities.shape
    native = probabilities.flatten(2)
    p = native.double()
    yy,xx = torch.meshgrid(torch.arange(h,device=p.device,dtype=p.dtype)/h,
                           torch.arange(w,device=p.device,dtype=p.dtype)/w,indexing='ij')
    x,y = xx.flatten(),yy.flatten()
    moments = torch.stack((x,y,x*x,x*y,y*y),-1)
    # Preserve the original array dtype for the threshold comparison: casting
    # first would incorrectly retain float32 values exactly equal to 0.1.
    raw = p*(native>cfg['noise_threshold'])
    mass = raw.sum(-1,keepdim=True)
    weights = raw/mass.clamp_min(1e-30)
    values = weights @ moments
    means = torch.where(mass>0,values[...,:2],torch.full_like(values[...,:2],.5))
    cxx = values[...,2]-means[...,0].square()
    cxy = values[...,3]-means[...,0]*means[...,1]
    cyy = values[...,4]-means[...,1].square()
    covariance = torch.stack((cxx,cxy,cxy,cyy),-1).reshape(b,k,2,2)
    peak = p.max(-1).values
    scale = p.new_tensor([cfg['image_hw'][1],cfg['image_hw'][0]])
    floor = torch.diag((cfg['unary_covariance_floor_px']/scale).square())
    strong = (peak>cfg['confidence_threshold']) & ((raw>0).sum(-1)>=2)
    # Invert only valid covariance candidates; weak/empty maps get an SPD dummy.
    covariance = torch.where(strong[...,None,None],covariance/peak.clamp_min(1e-30)[...,None,None]+floor,
                             torch.eye(2,device=p.device,dtype=p.dtype))
    precision = torch.linalg.inv(covariance)
    precision = torch.where(strong[...,None,None],precision,torch.diag(1e-6*scale.square()))
    return means,precision,peak


class JointConstrainedLocalization(nn.Module):
    def __init__(self, prior, prior_scale=1.):
        super().__init__()
        if prior_scale <= 0:
            raise ValueError('Structure scale must be positive')
        self.prior_scale = float(prior_scale)
        k = int(prior['count'])
        self.register_buffer('edges',torch.as_tensor(prior['edges'],dtype=torch.long))
        self.register_buffer('means',torch.as_tensor(prior['means'],dtype=torch.float64))
        self.register_buffer('precisions',torch.as_tensor(prior['precisions'],dtype=torch.float64))
        self.register_buffer('incidence',torch.nn.functional.one_hot(self.edges[:,0],k).double()-torch.nn.functional.one_hot(self.edges[:,1],k).double())

    def system(self, means, precisions, edge_scale=None, prior_transform=None):
        means,precisions = means.double(),precisions.double()
        b,k,_ = means.shape
        edge_precision = self.precisions[None].expand(b,-1,-1,-1) * self.prior_scale
        edge_means = self.means[None].expand(b,-1,-1)
        if prior_transform is not None:
            transform = prior_transform.to(device=means.device,dtype=torch.double)
            inverse = torch.linalg.inv(transform)
            edge_means = torch.einsum('bij,bej->bei',transform,edge_means)
            edge_precision = inverse[:,None].transpose(-1,-2) @ edge_precision @ inverse[:,None]
        if edge_scale is not None:
            edge_precision = edge_precision*edge_scale.double()[...,None,None]
        identity = torch.eye(k,device=means.device,dtype=means.dtype)
        q = torch.einsum('kl,bkij->bkilj',identity,precisions)
        q = q + torch.einsum('ek,beij,el->bkilj',self.incidence,edge_precision,self.incidence)
        q = q.reshape(b,k*2,k*2)
        rhs = torch.einsum('bkij,bkj->bki',precisions,means)
        rhs = rhs + torch.einsum('ek,beij,bej->bki',self.incidence,edge_precision,edge_means)
        return q, rhs.reshape(b,-1)

    @staticmethod
    def constrain(q, rhs, clicked, confirmed):
        b,k = clicked.shape
        mask = clicked[...,None].expand(-1,-1,2).reshape(b,-1)
        values = torch.where(mask,confirmed.double().reshape(b,-1),torch.zeros_like(rhs.reshape(b,-1)))
        rhs = rhs.reshape(b,-1)-torch.bmm(q,values[...,None]).squeeze(-1)
        unknown = (~mask).double()
        constrained = q*unknown[:,:,None]*unknown[:,None,:]+torch.diag_embed(mask.double())
        rhs = torch.where(mask,values,rhs)
        return constrained, rhs, mask

    def forward(self, means, precisions, clicked, confirmed, edge_scale=None, prior_transform=None):
        q,rhs = self.system(means,precisions,edge_scale,prior_transform)
        constrained,rhs,_ = self.constrain(q,rhs,clicked,confirmed)
        result = torch.linalg.solve(constrained,rhs[...,None]).reshape_as(means)
        return torch.where(clicked[...,None],confirmed.double(),result)

    @torch.no_grad()
    def reference(self, means, precisions, prior_transform=None):
        # One immutable automatic quadratic per image/interaction sequence.
        # Detachment avoids retaining all historical graphs; the final visual
        # path and final JCL remain differentiable.
        q,rhs = self.system(means.detach(),precisions.detach(),prior_transform=prior_transform)
        return dict(reference_q=q.detach(),reference_rhs=rhs.detach())

    @torch.no_grad()
    def propose(self, history, cfg):
        """Condition the fixed quadratic jointly, never sum stale click offsets."""
        q,rhs = history['reference_q'],history['reference_rhs']
        clicked,confirmed = history['clicked'],history['confirmed']
        constrained,rhs,mask = self.constrain(q,rhs,clicked,confirmed)
        factor = torch.linalg.cholesky(constrained)
        candidate = torch.cholesky_solve(rhs[...,None],factor).reshape_as(confirmed)
        candidate = torch.where(clicked[...,None],confirmed.double(),candidate)
        covariance = torch.cholesky_inverse(factor)
        unknown = (~mask).double()
        covariance = covariance*unknown[:,:,None]*unknown[:,None,:]
        b,k = clicked.shape
        blocks = covariance.reshape(b,k,2,k,2).diagonal(dim1=1,dim2=3).permute(0,3,1,2)
        scale = candidate.new_tensor([cfg['image_hw'][1],cfg['image_hw'][0]])
        covariance_px = blocks*scale[None,None,:,None]*scale[None,None,None,:]
        covariance_px = .5*(covariance_px+covariance_px.transpose(-1,-2))
        eigenvalues,eigenvectors = torch.linalg.eigh(covariance_px)
        radii = eigenvalues.clamp_min(0).sqrt().clamp(cfg['search_std_min_px'],cfg['search_std_max_px'])
        # Symmetric square root, not an eigenvector orientation with arbitrary
        # sign; repeated eigenvalues do not introduce backward singularities.
        basis = (eigenvectors*radii[...,None,:])@eigenvectors.transpose(-1,-2)
        current = history['current'].double()
        reference = torch.where(clicked[...,None],confirmed.double(),current)
        bounded = candidate.clamp(0.,1.)
        delta_px = (bounded-reference)*scale
        length = torch.linalg.vector_norm(delta_px,dim=-1,keepdim=True)
        ratio = (cfg['search_max_shift_px']/length.clamp_min(1e-12)).clamp_max(1.)
        search_candidate = reference+delta_px*ratio/scale
        search_candidate = torch.where(clicked[...,None],reference,search_candidate)
        return dict(candidate=candidate, covariance_px=covariance_px,
                    basis_px=basis.float(), search_candidate=search_candidate.float())


# Import compatibility for existing V132 callers; no duplicate module registration.
JointAnchorReconstruction = JointConstrainedLocalization


def local_heatmap_unaries(probabilities,cfg):
    """Differentiable moments in one peak neighborhood; no distant-peak averaging."""
    b,k,h,w=probabilities.shape
    index=probabilities.flatten(2).argmax(-1)
    radius=cfg['local_peak_radius']
    offsets=torch.arange(-radius,radius+1,device=probabilities.device)
    dy,dx=torch.meshgrid(offsets,offsets,indexing='ij')
    x=index[...,None]%w+dx.flatten(); y=index[...,None]//w+dy.flatten()
    valid=(x>=0)&(x<w)&(y>=0)&(y<h)
    indices=y.clamp(0,h-1)*w+x.clamp(0,w-1)
    values=probabilities.flatten(2).gather(-1,indices).double()*valid
    mass=values.sum(-1,keepdim=True)
    weights=values/mass.clamp_min(1e-30)
    locations=torch.stack((x.double()/w,y.double()/h),-1)
    means=(weights[...,None]*locations).sum(-2)
    fallback=torch.stack(((index%w).double()/w,(index//w).double()/h),-1)
    means=torch.where(mass>0,means,fallback)
    # Fit the local log-heatmap curvature for subpixel coordinates. Fall back
    # to local moments at borders, flat peaks, or non-concave neighborhoods.
    neighbors=torch.tensor([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]],device=index.device)
    nx=(index[...,None]%w+neighbors[:,0]).clamp(0,w-1)
    ny=(index[...,None]//w+neighbors[:,1]).clamp(0,h-1)
    logs=probabilities.flatten(2).gather(-1,ny*w+nx).double().clamp_min(1e-9).log()
    gx=(logs[...,5]-logs[...,3])/2; gy=(logs[...,7]-logs[...,1])/2
    hxx=logs[...,5]-2*logs[...,4]+logs[...,3]
    hyy=logs[...,7]-2*logs[...,4]+logs[...,1]
    hxy=(logs[...,8]-logs[...,6]-logs[...,2]+logs[...,0])/4
    hessian=torch.stack((hxx,hxy,hxy,hyy),-1).reshape(b,k,2,2)
    good=(hxx< -1e-6)&(hyy< -1e-6)&(hxx*hyy-hxy*hxy>1e-8)
    good=good&(index%w>0)&(index%w<w-1)&(index//w>0)&(index//w<h-1)
    safe=torch.where(good[...,None,None],hessian,-torch.eye(2,device=index.device,dtype=torch.double))
    delta=torch.linalg.solve(safe,-torch.stack((gx,gy),-1)[...,None])[...,0]
    good=good&(delta.abs().amax(-1)<=1.)
    refined=fallback+delta/means.new_tensor([w,h])
    means=torch.where(good[...,None],refined,means)
    centered=locations-means[...,None,:]
    covariance=torch.einsum('bkn,bkni,bknj->bkij',weights,centered,centered)
    peak=probabilities.flatten(2).max(-1).values.double()
    scale=means.new_tensor(cfg['image_hw'][::-1])
    floor=torch.diag((cfg['unary_covariance_floor_px']/scale).square())
    strong=(peak>cfg['confidence_threshold']) & (mass[...,0]>0)
    covariance=torch.where(strong[...,None,None],covariance/peak.clamp_min(1e-30)[...,None,None]+floor,
                           torch.eye(2,device=means.device,dtype=torch.double))
    precision=torch.linalg.inv(covariance)
    precision=torch.where(strong[...,None,None],precision,torch.diag(1e-6*scale.square()))
    return means,precision,peak
