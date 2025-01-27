import time
import warnings

import torch




if torch.__version__=='1.4.0':
    old_torch=True
    from torchsearchsorted import searchsorted
else:
    print(f'rendering.py: torch version ={torch.__version__}')
    searchsorted=torch.searchsorted
    old_torch=False

__all__ = ['render_rays']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()


    if old_torch:
        inds = searchsorted(cdf, u, side='right')
    else:
        inds = searchsorted(cdf, u, right=True)

    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False,
                chroma_unembedded_=None,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction, or chromatic embedding defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(model, embedding_xyz, xyz_0, dir_, dir_embedded, z_vals, weights_only=False,
                  chroma_embedded_siamese=None):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module (the function of embedding) for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_0.shape[1]
        # Embed directions
        xyz_ = xyz_0.view(-1, 3)  # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)

            if chroma_embedded_siamese is not None:

                if chroma_embedded_siamese.shape[0] > 1:

                    chroma_embedded_siamese = torch.repeat_interleave(chroma_embedded_siamese, repeats=N_samples_, dim=0)
                else:
                    assert chroma_embedded_siamese.shape[0] == 1
                    chroma_embedded_siamese = torch.repeat_interleave(chroma_embedded_siamese, repeats=dir_embedded.shape[0], dim=0)

                if dir_embedded.shape[0] != chroma_embedded_siamese.shape[0]:
                    time.sleep(3)
                assert dir_embedded.shape[0] == chroma_embedded_siamese.shape[0]

                # (N_rays*N_samples_, embed_dir_channels)
        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []

        # for each ray, the chroma code should be identical

        # if round(B) % chunk !=0:
        #    warnings.warn('B cannot be divided by chunk')
        #    time.sleep(100)
        #    time.sleep(200)
        #    raise AssertionError
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i + chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i + chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded

            # for each ray, the chroma code should be identical
            xyz_size = xyzdir_embedded.shape[0]

            if chroma_embedded_siamese is not None:

                if chroma_embedded_siamese.shape[0] < xyz_size:

                    # should be (1, xxx) for the initial validating process

                    assert chroma_unembedded_.shape[0] == 1
                    '''
                    if chroma_unembedded_.shape[0] != 1:
                        print('caution?')

                        time.sleep(10)
                        assert chroma_unembedded_.shape[0] == 1

                        print('caution?')
                    '''
                    chroma_embedded_siamese = chroma_embedded_siamese.repeat(xyz_size, 1)
                else:
                    chroma_embedded_siamese = chroma_embedded_siamese

                embedded_input = torch.cat([xyzdir_embedded,
                                            chroma_embedded_siamese[i:i + chunk]], 1)
            else:
                embedded_input = xyzdir_embedded
            if torch.cuda.is_available():
                embedded_input = embedded_input.cuda()
            out_chunks += [model(embedded_input, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        elif chroma_embedded_siamese is None:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 3 + 3 + 1)
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            rgbs2 = rgbsigma[..., 3:6]  # (N_rays, N_samples_, 3)

            sigmas = rgbsigma[..., 6]  # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]  # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        if chroma_embedded_siamese is None:
            # compute final weighted outputs
            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
            depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

            if white_back:
                rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

            return rgb_final, depth_final, weights



        else:

            # compute final weighted outputs
            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
            rgb2_final = torch.sum(weights.unsqueeze(-1) * rgbs2, -2)  # (N_rays, 3)

            depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

            if white_back:
                rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)
                rgb2_final = rgb2_final + 1 - weights_sum.unsqueeze(-1)

            return rgb_final, rgb2_final, depth_final, weights

    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    if len(embeddings) == 2:
        '''
        DO NOT have the chroma embedding.
        '''

        if test_time:
            weights_coarse = \
                inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                          dir_embedded, z_vals, weights_only=True)
            result = {'opacity_coarse': weights_coarse.sum(1)}
        else:
            rgb_coarse, depth_coarse, weights_coarse = \
                inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                          dir_embedded, z_vals, weights_only=False)
            result = {'rgb_coarse': rgb_coarse,
                      'depth_coarse': depth_coarse,
                      'opacity_coarse': weights_coarse.sum(1)
                      }

        if N_importance > 0:  # sample points for fine model
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
            z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                                 N_importance, det=(perturb == 0)).detach()
            # detach so that grad doesn't propogate to weights_coarse from here

            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

            xyz_fine_sampled = rays_o.unsqueeze(1) + \
                               rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
            # (N_rays, N_samples+N_importance, 3)

            model_fine = models[1]
            rgb_fine, depth_fine, weights_fine = \
                inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                          dir_embedded, z_vals, weights_only=False)

            result['rgb_fine'] = rgb_fine
            result['depth_fine'] = depth_fine
            result['opacity_fine'] = weights_fine.sum(1)

        return result
    else:
        '''
        DO  have the chroma embedding.
        '''
        assert chroma_unembedded_ is not None
        chroma_embedding_func = embeddings[2]

        identical_transformation_code0=torch.zeros_like(chroma_unembedded_)
        identical_transformation_code=chroma_embedding_func(identical_transformation_code0)
        chroma_embedding0 = chroma_embedding_func(chroma_unembedded_)

        chroma_siamese_embedding=torch.cat([identical_transformation_code,chroma_embedding0],dim=1)

        if test_time:
            weights_coarse = \
                inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                          dir_embedded, z_vals,
                          weights_only=True,
                          chroma_embedded_siamese=chroma_siamese_embedding)
            result = {'opacity_coarse': weights_coarse.sum(1)}
        else:
            rgb_coarse, rgb2_coarse, depth_coarse, weights_coarse = \
                inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                          dir_embedded, z_vals, weights_only=False,
                          chroma_embedded_siamese=chroma_siamese_embedding)

            result = {'rgb_coarse': rgb_coarse,
                      'rgb2_coarse': rgb2_coarse,
                      'depth_coarse': depth_coarse,
                      'opacity_coarse': weights_coarse.sum(1)
                      }

        if N_importance > 0:  # sample points for fine model
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
            z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                                 N_importance, det=(perturb == 0)).detach()
            # detach so that grad doesn't propogate to weights_coarse from here

            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

            xyz_fine_sampled = rays_o.unsqueeze(1) + \
                               rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
            # (N_rays, N_samples+N_importance, 3)

            model_fine = models[1]
            rgb_fine, rgb2_fine, depth_fine, weights_fine = \
                inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                          dir_embedded, z_vals, weights_only=False,
                          chroma_embedded_siamese=chroma_siamese_embedding)

            result['rgb_fine'] = rgb_fine
            result['rgb2_fine'] = rgb2_fine
            result['depth_fine'] = depth_fine
            result['opacity_fine'] = weights_fine.sum(1)

        return result
