#@title Sampling (double click to expand or collapse)
import  torch
from torchvision.utils import make_grid, save_image

from utils import (
marginal_prob_std,
diffusion_coeff,
Euler_Maruyama_sampler,
pc_sampler,
ode_sampler
)
from models import  ScoreNet

from argparse import ArgumentParser
import functools
import numpy as np

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help = 'cpu / cuda')
    parser.add_argument('--sampler', type=str, default='euler_maruyama', help = 'Euler_Maruyama / pc / ode')
    parser.add_argument('--sigma', type = float, default=25.0, help='Standard deviation')
    parser.add_argument('--sample-batch-size', type = int, default = 64, help = 'Batch size')
    inputs = parser.parse_args()
    print(inputs)
    device = inputs.device
    sampler = ode_sampler if inputs.sampler == 'ode' else pc_sampler if inputs.sampler == 'pc' else Euler_Maruyama_sampler
    sample_batch_size = inputs.sample_batch_size
    sigma = inputs.sigma


    ## Load the pre-trained checkpoint from disk.
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma = sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma = sigma)

    score_model = score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    ckpt = torch.load('ckpt.pth', map_location = device)
    score_model.load_state_dict(ckpt)

    ## Generate samples using the specified sampler.
    samples = sampler(
        score_model, marginal_prob_std_fn, diffusion_coeff_fn, sample_batch_size, device = device
    )

    ## Save samples.
    samples = samples.clamp(0.0, 1.0)
    # % matplotlib
    # inline
    # import matplotlib.pyplot as plt

    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    print(sample_grid.shape)
    save_image(sample_grid, 'img1.png')

    # plt.figure(figsize=(6, 6))
    # plt.axis('off')
    # plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    # plt.show()

