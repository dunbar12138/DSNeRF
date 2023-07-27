import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SigmaLoss:
#     def __init__(self, N_samples, perturb, raw_noise_std):
#         super(SigmaLoss, self).__init__()
#         self.N_samples = N_samples
#         self.perturb = perturb
#         self.raw_noise_std = raw_noise_std

#     def calculate_loss(self, rays_o, rays_d, viewdirs, near, far, depths, run_func, network):
#         # print(near.mean(), depths[0], far.mean())
#         # assert near.mean() <= depths[0] and depths[0] <= far.mean()
#         N_rays = rays_o.shape[0]
#         t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
#         t_vals = t_vals.expand([N_rays, self.N_samples])
#         z_vals = near * (1.-t_vals) + depths[:,None] * (t_vals)
#         if self.perturb > 0.:
#             # get intervals between samples
#             mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
#             upper = torch.cat([mids, z_vals[...,-1:]], -1)
#             lower = torch.cat([z_vals[...,:1], mids], -1)
#             # stratified samples in those intervals
#             t_rand = torch.rand(z_vals.shape).to(device)

#             z_vals = lower + (upper - lower) * t_rand
#         pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
#         raw = run_func(pts, viewdirs, network)

#         noise = 0.
#         if self.raw_noise_std > 0.:
#             noise = torch.randn(raw[...,3].shape) * self.raw_noise_std

#         sigma = F.relu(raw[...,3] + noise)
#         # sigma_sigmoid = torch.sigmoid(sigma)    # [N_rays, N_samples]
#         # assert sigma_sigmoid.shape[0] == N_rays and sigma_sigmoid.shape[1] == self.N_samples
#         # # sigma_sigmoid = torch.mean(sigma_sigmoid, axis=0)
#         # loss = torch.sum(sigma_sigmoid[:,:-1], axis=1) - sigma_sigmoid[:,-1]
#         loss = -torch.exp(sigma[:,-1]) / (torch.sum(torch.exp(sigma), axis=1) + 1)
#         return loss


class SigmaLoss:
    def __init__(self, N_samples, perturb, raw_noise_std):
        super(SigmaLoss, self).__init__()
        self.N_samples = N_samples
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std

    def calculate_loss(self, rays_o, rays_d, viewdirs, near, far, depths, run_func, network, err=1):
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        N_rays = rays_o.shape[0]
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        t_vals = t_vals.expand([N_rays, self.N_samples])
        z_vals = near * (1.-t_vals) + far * (t_vals)
        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = run_func(pts, viewdirs, network)

        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * self.raw_noise_std

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        # sigma = F.relu(raw[...,3] + noise)
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        
        
        loss = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - depths[:,None]) ** 2 / (2 * err)) * dists
        loss = torch.sum(loss, dim=1)
        
        return loss
