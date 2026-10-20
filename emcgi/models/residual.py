"""Click-conditioned displacement system, centered on immutable automatic points.

Energy: ||delta-v||_U^2 + lambda ||B delta-d||_W^2.
No population mean displacement enters the residual right-hand side.
"""
import torch


def residual_system(geometry, precision, visual_delta, edge_delta, strength,
                    edge_scale=None, prior_transform=None):
    if strength <= 0:
        raise ValueError('Interaction structure strength must be positive')
    precision,visual_delta = precision.double(),visual_delta.double()
    batch,count = visual_delta.shape[:2]
    weights = geometry.precisions[None].expand(batch,-1,-1,-1)*strength
    if prior_transform is not None:
        inverse = torch.linalg.inv(prior_transform.to(precision))
        weights = inverse[:,None].transpose(-1,-2) @ weights @ inverse[:,None]
    if edge_scale is not None:
        weights = weights*edge_scale.double()[...,None,None]
    identity = torch.eye(count,device=precision.device,dtype=torch.double)
    q = torch.einsum('kl,bkij->bkilj',identity,precision)
    q = q+torch.einsum('ek,beij,el->bkilj',geometry.incidence,weights,geometry.incidence)
    rhs = torch.einsum('bkij,bkj->bki',precision,visual_delta)
    if edge_delta is not None:
        rhs = rhs+torch.einsum('ek,beij,bej->bki',geometry.incidence,weights,edge_delta.double())
    return q.reshape(batch,count*2,count*2),rhs.reshape(batch,-1)


@torch.no_grad()
def residual_reference(geometry, points, means, precision, strength, prior_transform=None):
    q,_ = residual_system(geometry,precision,torch.zeros_like(points),None,strength,
                           prior_transform=prior_transform)
    # Existing proposal/search operates on absolute coordinates. Recenter Q
    # exactly at p0 so stronger propagation cannot change the unclicked origin.
    rhs = torch.bmm(q,points.double().flatten(1)[...,None]).squeeze(-1)
    return dict(reference_q=q.detach(),reference_rhs=rhs.detach(),
                automatic_points=points.detach(),automatic_unary_mean=means.detach(),
                automatic_precision=precision.detach())


def solve_residual(geometry, history, visual_delta, edge_delta, strength,
                   edge_scale=None, prior_transform=None):
    origin = history['automatic_points'].double()
    q,rhs = residual_system(geometry,history['automatic_precision'],visual_delta,
                           edge_delta,strength,edge_scale,prior_transform)
    confirmed_delta = history['confirmed'].double()-origin
    constrained,rhs,_ = geometry.constrain(q,rhs,history['clicked'],confirmed_delta)
    delta = torch.linalg.solve(constrained,rhs[...,None]).reshape_as(origin)
    points = origin+delta
    return torch.where(history['clicked'][...,None],history['confirmed'].double(),points)
