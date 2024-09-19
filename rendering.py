"""
This script renders the input rays that are used to feed the NeRF model
It discretizes each ray in the input batch into a set of 3d points at different depths of the scene
Then the nerf model takes these 3d points (and the ray direction, optionally, as in the original nerf)
and predicts a volume density at each location (sigma) and the color with which it appears
"""

import torch
import math
import numpy as np
import train_utils

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Args:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Returns:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

def sample_3sigma(low_3sigma_, high_3sigma_, N, det, near, far, device=None, gt=False, dRange=3., eps=1e-5, mode='test'):
    '''
    Args:
        low_3sigma: (N_rays)
        high_3sigma: (N_rays)
        N: =N_samples
    '''
    high_3sigma = high_3sigma_ #.clamp(near, far)
    low_3sigma = low_3sigma_ #.clamp(near, far)
    t_vals = torch.linspace(0., 1., steps=N, device=device) #(N_samples)
    step_size = (high_3sigma - low_3sigma) / (N - 1)  #(N_rays)
    bin_edges_ = (low_3sigma.unsqueeze(-1) * (1.-t_vals) + high_3sigma.unsqueeze(-1) * (t_vals))
    bin_edges = bin_edges_ #.clamp(near, far) #(N_rays, N_samples)
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / (step_size.unsqueeze(-1) + eps) #(N_rays, N_samples-1)
    x_in_3sigma = torch.linspace(-dRange, dRange, steps=(N - 1), device=device) #(N_samples-1)
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(*bin_edges.shape[:-1], N - 1) #(N_rays, N_samples-1)
    res = sample_pdf(bin_edges, bin_weights, N, det=det) #(N_rays, N_samples)

    res, _ = torch.sort(res, -1)

    return res

def sample_3sigma_asym(depth, low_3sigma_, high_3sigma_, N, det, near, far, device=None, gt=False, dRange=3., eps=1e-5, mode='test'):
    low_3sigma = low_3sigma_.clamp(near, far)
    high_3sigma = high_3sigma_.clamp(near, far)
    high_range = torch.abs(high_3sigma - depth)
    low_range = torch.abs(low_3sigma - depth)
    range = torch.min(high_range, low_range)
    low_3sigma = depth - range
    high_3sigma = depth + range

    depth_check = (low_3sigma + high_3sigma)/2.
    if torch.max(torch.abs(depth_check - depth)) > 1e-5:
        train_utils.PrintMMM('depth_check - depth in sample_3sigma_asym', depth_check - depth)

    result = sample_3sigma(low_3sigma, high_3sigma, N, det, 0, 1, device=device, gt=gt, dRange=dRange, mode=mode)

    return result

def sample_3sigma_asym_todel(depth, low_3sigma_, high_3sigma_, N, det, near, far, device=None, gt=False, dRange=3., eps=1e-5, mode='test'):
    low_3sigma = low_3sigma_.clamp(near, far)
    high_3sigma = 2*depth - low_3sigma
    res = sample_3sigma(low_3sigma, high_3sigma, N, det, 0, 1, device=device, gt=gt, dRange=dRange, mode=mode)
    idx = 0
    high_3sigma = high_3sigma_.clamp(near, far)
    low_3sigma = 2*depth - high_3sigma
    res_ = sample_3sigma(low_3sigma, high_3sigma, N, det, 0, 1, device=device, gt=gt, dRange=dRange, mode=mode)
    half_len = int(res_.shape[-1]/2.)
    res[:,half_len:] = res_[:,half_len:]
    if 1:
        res = res.clamp(near, far)

    if torch.min(res) < near or torch.max(res) > far:
        print('sample range [{:.3f}, {:.3f}] is OUT of near-far range [{:.3f}, {:.3f}]'.format(torch.min(res), torch.max(res), near, far))
        print('res.shape[0]', res.shape[0])
        for i in range(res.shape[0]):
            res1 = res[i, :]
            if torch.min(res1) < near or torch.max(res1) > far:
                print('---i, res1.shape, torch.min(res1), torch.max(res1)', i, res1.shape, torch.min(res1), torch.max(res1), res1)

    return res

def compute_samples_around_depth(res, N_samples, z_vals, perturb, near, far, device=None, dRange=3., mode='test'):
    pred_depth = res['depth']
    pred_weight = res['weights']
    sampling_std = train_utils.calc_depth_std(z_vals, pred_depth, pred_weight)

    pred_depth, _ = train_utils.check_nan(pred_depth, func='pred_depth')
    pred_weight, _ = train_utils.check_nan(pred_weight, func='pred_weight')
    sampling_std, _ = train_utils.check_nan(sampling_std, func='sampling_std')

    depth_min = pred_depth - dRange * sampling_std
    depth_max = pred_depth + dRange * sampling_std

    z_vals_2 = sample_3sigma_asym(pred_depth, depth_min, depth_max, N_samples, perturb == 0., near, far, device=device, dRange=dRange, mode=mode)

    return z_vals_2

def GenerateGuidedSamples(res, z_vals, N_samples, perturb, near, far, mode='test', valid_depth=None, target_depths=None, target_std=None, device=None, margin=0, stdscale=1, dRange=3.):
    z_vals_2 = compute_samples_around_depth(res, N_samples, z_vals, perturb, near[0, 0], far[0, 0], device=device, dRange=dRange, mode='test')

    if mode == 'train' and valid_depth != None:
        target_depth = torch.flatten(target_depths[:, 0][np.where(valid_depth.cpu()>0)])
        
        target_weight = target_depths[:, 1][np.where(valid_depth.cpu()>0)]
        target_std = torch.flatten(target_std[np.where(valid_depth.cpu()>0)])
        depth_min = target_depth - dRange * target_std
        depth_max = target_depth + dRange * target_std

        z_vals_2_bkp = z_vals_2.clone()
        gt_samples = sample_3sigma_asym(target_depth, depth_min, depth_max, N_samples, perturb == 0., near[0, 0], far[0, 0], device=device,gt=True, dRange=dRange, mode=mode)
        z_vals_2[np.where(valid_depth.cpu()>0)] = gt_samples

    return z_vals_2

def get_z_vals(N_samples, device, near, far, use_disp = False, perturb = 1.0):
    # sample depths for coarse model
    z_steps = torch.linspace(0, 1, N_samples, device=device)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    return z_vals

def render_rays(models, args, rays, ts, mode='test', valid_depth=None, target_depths=None, target_std=None, apply_brdf=False, print_debuginfo=False, bTestNormal=False, bTestSun_v=False, gsam_only=False, rows=None, cols=None, percent=0, apply_theta=False, cos_irra_on=False):
    brdf_type = 'Lambertian'
    guided_samples = args.guided_samples
    # get config values
    N_samples = args.n_samples
    N_importance = args.n_importance
    variant = args.model
    perturb = 1.0
    dRange = args.std_range
    sun_res = {}

    # get rays
    rays_o, rays_d, near, far = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]
    z_vals = get_z_vals(N_samples, rays.device, near, far, perturb=perturb)

    # discretize rays into a set of 3d points (N_rays, N_samples_, 3), one point for each depth of each ray
    xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    # run coarse model
    typ = "coarse"
    sun_d = None
    sun_d = torch.ones_like(rays_o)

    if args.data == 'sat':
        sun_d = rays[:, 8:11]

    if variant == "s-nerf":
        from models.snerf import inference
        # render using main set of rays
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d)
        if args.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_ = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d)
            result['weights_sc'] = result_["weights"]
            result['transparency_sc'] = result_["transparency"]
            result['sun_sc'] = result_["sun"]
    elif variant == "sat-nerf" or variant == "sps-nerf":
        from models.satnerf import inference
        rays_t = None 
        if args.beta == True:
            rays_t = models['t'](ts) if ts is not None else None
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
        if(guided_samples > 0 and variant == 'sps-nerf'):       #guidedsample is only for sps-nerf
            z_vals_2 = GenerateGuidedSamples(result, z_vals, guided_samples, perturb, near, far, mode=mode, valid_depth=valid_depth, target_depths=target_depths, target_std=target_std, device=rays.device, margin=args.margin, stdscale=args.stdscale).detach()
            z_vals_2, _ = torch.sort(z_vals_2, -1)
            z_vals_unsort = torch.cat([z_vals, z_vals_2], -1)
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_2], -1), -1)
            xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples+guided_samples, 3)
            result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t, z_vals_unsort=z_vals_unsort)        
        if args.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_tmp = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
            result['weights_sc'] = result_tmp["weights"]
            result['transparency_sc'] = result_tmp["transparency"]
            result['sun_sc'] = result_tmp["sun"]
    elif variant == "spsbrdf-nerf":
        from models.spsbrdfnerf import inference    #, inference_sun
        rays_t = None 
        if args.beta == True:
            rays_t = models['t'](ts) if ts is not None else None
        #only calculate sigma in the first inference
        if guided_samples > 0:
            result, _ = inference(models[typ], args, xyz_coarse, z_vals, rays_d=rays_d, sun_d=sun_d, rays_t=rays_t, print_debuginfo=print_debuginfo, mode=mode, sigma_only=True)
            models[typ].check_nan_parms(keyword='1st')
        else:
            print('guided_samples <= 0')
            result, brdf_type = inference(models[typ], args, xyz_coarse, z_vals, rays_d=rays_d, sun_d=sun_d, rays_t=rays_t, apply_brdf=apply_brdf, print_debuginfo=print_debuginfo, bTestNormal=bTestNormal, rows=rows, cols=cols, percent=percent, mode=mode, apply_theta=apply_theta, cos_irra_on=cos_irra_on)
            return result, brdf_type

        guided_samples_r = guided_samples
        if guided_samples == 2:
            dRange = 0.0001
            guided_samples_r = 1

        if sun_d != None and ((models[typ].sun_v == 'analystic' and apply_brdf == True) or bTestSun_v == True):
            pt_surf = rays_o + rays_d * result['depth'].unsqueeze(-1)
            far_sun = result['depth'].clone().unsqueeze(-1)
            if torch.abs(sun_d[0,2]) > 0.00001:
                far_sun = torch.abs(rays_d[0,2]/sun_d[0,2])*far_sun
            if gsam_only == True:
                N_samples_1 = guided_samples_r
            else:
                N_samples_1 = N_samples  #N_samples+guided_samples_r
            z_vals_sun = get_z_vals(N_samples_1, rays.device, far_sun*0.01, far_sun, perturb=perturb)
            xyz_coarse = pt_surf.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals_sun.unsqueeze(2)

            result_tmp, _ = inference(models[typ], args, xyz_coarse, z_vals_sun, rays_d=sun_d, mode=mode, sigma_only=True)
            models[typ].check_nan_parms(keyword='sunv')
            sun_res['sun'] = result_tmp["transparency"].unsqueeze(-1).detach() #(N_rays, N_samples, 1), only the (N_rays, -1) represents the sun visibility
            sun_res['weights_sc'] = result_tmp["weights"].detach()

        if guided_samples > 0:
            z_vals_2 = GenerateGuidedSamples(result, z_vals, guided_samples, perturb, near, far, mode=mode, valid_depth=valid_depth, target_depths=target_depths, target_std=target_std, device=rays.device, margin=args.margin, stdscale=args.stdscale, dRange=dRange).detach()
            z_vals_2, _ = torch.sort(z_vals_2, -1)
            if guided_samples_r == 1:
                z_vals_2 = torch.mean(z_vals_2, dim=1).unsqueeze(-1)
            if gsam_only == True:
                z_vals_unsort = z_vals_2
                z_vals = z_vals_2
                idx = None
            else:
                z_vals_unsort = torch.cat([z_vals, z_vals_2], -1)
                z_vals, idx = torch.sort(z_vals_unsort, -1)
            xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples+guided_samples, 3)

            result, brdf_type = inference(models[typ], args, xyz_coarse, z_vals, rays_d=rays_d, sun_d=sun_d, rays_t=rays_t, z_vals_unsort=z_vals_unsort, apply_brdf=apply_brdf, print_debuginfo=print_debuginfo, bTestNormal=bTestNormal, sun_res=sun_res, sort_idx=idx, rows=rows, cols=cols, percent=percent, mode=mode, apply_theta=apply_theta, cos_irra_on=cos_irra_on)
            models[typ].check_nan_parms(keyword='2nd')

        if args.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_tmp = inference_sun(models[typ], args, xyz_coarse, z_vals, rays_d=rays_d, sun_d=sun_d, rays_t=rays_t, mode=mode)
            result['weights_sc'] = result_tmp["weights"]
            result['transparency_sc'] = result_tmp["transparency"]
            result['sun_sc'] = result_tmp["sun"]
    else:
        # classic nerf
        from models.nerf import inference
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=rays_d)
    result_ = {}
    for k in result.keys():
        result_[f"{k}_{typ}"] = result[k]

    # run fine model
    if N_importance > 0:

        # sample depths for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, result_['weights_coarse'][:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach so that grad doesn't propogate to weights_coarse from here
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        # discretize rays for fine model
        xyz_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples+N_importance, 3)

        typ = "fine"
        if variant == "s-nerf":
            # render using main set of rays
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=rays_d_, sun_d=sun_d)
            if args.sc_lambda > 0:
                # solar correction
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
                result_ = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=None)
                result['weights_sc'] = result_["weights"]
                result['transparency_sc'] = result_["transparency"]
                result['sun_sc'] = result_["sun"]
        elif variant == "sat-nerf" or variant == "sps-nerf":
            rays_t = None  
            if args.beta == True:
                rays_t = models['t'](ts) if ts is not None else None
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
            if args.sc_lambda > 0:
                # solar correction
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
                result_ = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
                result['weights_sc'] = result_["weights"]
                result['transparency_sc'] = result_["transparency"]
                result['sun_sc'] = result_["sun"]           
        else:
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=rays_d)
        for k in result.keys():
            result_["{}_{}".format(k, typ)] = result[k]

    return result_, brdf_type
