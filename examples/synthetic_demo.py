"""Synthetic interface check; random weights, no patient data."""
import torch
from emcgi import EMCGI, model_config


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    count = 4
    cfg = model_config(points=count, image_hw=(128, 128))
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    prior = dict(count=count, edges=edges,
                 means=torch.zeros(3, 2),
                 precisions=torch.eye(2).repeat(3, 1, 1) * 100.)
    model = EMCGI(cfg, prior).eval()
    image = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        automatic = model(image)
        current = automatic['points'].detach().float().clone()
        history = dict(current=current.clone(), old=current.clone(),
                       confirmed=torch.zeros_like(current),
                       clicked=torch.zeros(1, count, dtype=torch.bool),
                       **automatic['reference'])
        for index, location in [(0, (.3, .4)), (2, (.6, .7))]:
            history['old'][:, index] = history['current'][:, index]
            history['confirmed'][:, index] = torch.tensor(location, dtype=current.dtype)
            history['clicked'][:, index] = True
            corrected = model(image, history)
            assert torch.isfinite(corrected['points']).all()
            mask = history['clicked']
            torch.testing.assert_close(corrected['points'][mask], history['confirmed'][mask].double(), rtol=0, atol=0)
            history['current'] = corrected['points'].detach().float().clone()
    print('PASS: automatic forward, two cumulative clicks, finite outputs and exact click constraints.')
    print('Synthetic data and random weights: this does not measure localization accuracy.')


if __name__ == '__main__':
    main()
