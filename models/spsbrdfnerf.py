from .nerf import Mapping, Siren, sine_init, first_layer_sine_init
import torch
import BRDF.microfacet as mcf
import BRDF.Hapke as hpk
import BRDF.RPV as rpv
import numpy as np
import train_utils 

def eval_RPV(pts2l, pts2c, normal, w, k, theta, rhoc, mode='train'):
    rpv_ = rpv.RPV()
    brdf, M1, G, H, ci, cv = rpv_(pts2l, pts2c, normal, w, k, theta, rhoc, mode=mode)

    return brdf, M1, G, H, ci, cv

def eval_Hapke(pts2l, pts2c, normal, w, b, c=None, theta=None, h=None, B0=None, mode='train', args=None):
    hapke = hpk.Hapke(args=args)
    brdf, P, B, Hi, Hv, ShadFunc, ci, cv = hapke(pts2l, pts2c, normal, w, b, c, theta, h, B0, mode)
    
    return brdf, P, B, Hi, Hv, ShadFunc, ci, cv

def eval_microfacet_brdf(pts2l, pts2c, normal, albedo, roughness, lvis=True, glossy_scale=1., print_debuginfo=False, fresnel_f0=0.04, mode='train'):
    """Fixed to microfacet (GGX).
    fresnel_f0 = 0.91    #almost metal
    fresnel_f0 = 0.04   #plastic
    fresnel_f0 = 0.18   #concrete
    """
    microfacet = mcf.Microfacet(f0=fresnel_f0, lvis=lvis, glossy_scale=glossy_scale, print_debuginfo=print_debuginfo)
    brdf_glossy, brdf, f, g, d, l_dot_n, v_dot_n, h, n_h = microfacet(pts2l, pts2c, normal, albedo=albedo, rough=roughness, mode=mode)
    # NxL and NxLx3
    return brdf_glossy, brdf, f, g, d, l_dot_n, v_dot_n, h, n_h

def checknan(result, xyz_, input_dir, input_sun_dir, input_t, model):
    contain_nan = False
    for k, v in result.items():
        if torch.is_tensor(result[k]): #result[k] != None:
            _, contain_nan_ = train_utils.check_nan(result[k], func='result '+k)
            if contain_nan_ == True:
                contain_nan = True
        else:
            pass
    if contain_nan == True:
        model.print_parms()
        train_utils.check_nan(xyz_, func='xyz_')
        train_utils.check_nan(input_dir, func='input_dir')
        train_utils.check_nan(input_sun_dir, func='input_sun_dir')
        train_utils.check_nan(input_t, func='input_t')

    return contain_nan

def cal_weight(z_vals, sigmas, args):
    # define deltas, i.e. the length between the points in which the ray is discretized
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples)

    # compute alpha as in the formula (3) of the nerf paper
    noise_std = args.noise_std
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    if args.noise_std > 1e-5:
        train_utils.PrintMMM('sigmas', sigmas)
        train_utils.PrintMMM('noise', noise)
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples)
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
    transparency = torch.cumprod(alphas_shifted, -1)[:, :-1]  # T in the paper
    weights = alphas * transparency # (N_rays, N_samples)
    depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
    
    return alphas, transparency, weights, depth_final

def inference(model, args, rays_xyz, z_vals, rays_d=None, sun_d=None, rays_t=None, z_vals_unsort=None, apply_brdf=False, print_debuginfo=False, bTestNormal=False, sun_res=[], sort_idx=None, rows=None, cols=None, percent=0, mode='train', apply_theta=False, sigma_only=False, cos_irra_on=False):
    """
    Runs the nerf model using a batch of input rays
    Args:
        model: NeRF model (coarse or fine)
        args: all input arguments
        rays_xyz: (N_rays, N_samples_, 3) sampled positions in the object space
                  N_samples_ is the number of sampled points in each ray;
                            = N_samples for coarse model
                            = N_samples+N_importance for fine model
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        rays_d: (N_rays, 3) direction vectors of the rays
        sun_d: (N_rays, 3) sun direction vectors
        z_vals_unsort: (N_rays, N_samples_) depths of the sampled positions before sorting, which is only used for visualizing the guided samples for the SpS-NeRF article
    Returns:
        result: dictionary with the output magnitudes of interest
    """
    brdf_type = 'Lambertian'

    #MultiBRDF = False
    N_rays = rays_xyz.shape[0]
    N_samples = rays_xyz.shape[1]
    xyz_ = rays_xyz.view(-1, 3)  # (N_rays*N_samples, 3)

    # check if there are additional inputs, which are used or not depending on the nerf variant
    rays_d_ = None if rays_d is None else torch.repeat_interleave(rays_d, repeats=N_samples, dim=0)
    sun_d_ = None if sun_d is None else torch.repeat_interleave(sun_d, repeats=N_samples, dim=0)
    rays_t_ = None if rays_t is None else torch.repeat_interleave(rays_t, repeats=N_samples, dim=0)

    # the input batch is split in chunks to avoid possible problems with memory usage
    chunk = args.chunk
    batch_size = xyz_.shape[0]

    out_channels = model.number_of_outputs
    if apply_brdf == True:
        out_channels = model.number_of_outputs_brdf
    nr_an_on = nr_lr_on = False
    if (model.normal == 'analystic_learned' or model.normal == 'analystic') or bTestNormal == True:
        out_channels += 3 # + normal (3)
        nr_an_on = True
    if model.normal == 'analystic_learned' or model.normal == 'learned':
        out_channels += 3 # + normal (3)
        nr_lr_on = True
    if apply_theta == True and model.args.theta == True:
        out_channels += 1

    # run model
    out_chunks = []
    for i in range(0, batch_size, chunk):
        out_chunks += [model(xyz_[i:i+chunk],
                             input_dir=None if rays_d_ is None else rays_d_[i:i + chunk],
                             input_sun_dir=None if sun_d_ is None else sun_d_[i:i + chunk],
                             input_t=None if rays_t_ is None else rays_t_[i:i + chunk],
                             apply_brdf=apply_brdf, apply_theta=apply_theta, nr_an_on=nr_an_on, nr_lr_on=nr_lr_on, mode=mode, sigma_only=sigma_only)]
    out = torch.cat(out_chunks, 0)

    if sigma_only == True:
        sigmas = out.view(N_rays, N_samples)
        alphas, transparency, weights, depth_final = cal_weight(z_vals, sigmas, args)
        result = {'sigmas': sigmas.unsqueeze(-1),   # (N_rays, N_samples, 1)
                  'depth': depth_final,             # (N_rays)
                  'alphas': alphas,                 # (N_rays, N_samples)
                  'weights': weights,               # (N_rays, N_samples)
                  'transparency': transparency,
                  'z_vals': z_vals}
        return result, brdf_type

    out = out.view(N_rays, N_samples, out_channels)
    albedo = out[..., :3]  # (N_rays, N_samples, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples)
    idx = 4

    apply_sun_v = False
    if model.sun_v == 'learned':
        apply_sun_v = True
        sun_v = out[..., 4:5]  # (N_rays, N_samples, 1)
        idx += 1
    elif model.sun_v == 'analystic':
        if "sun" in sun_res:
            apply_sun_v = True
            sun_v = sun_res['sun'] # (N_rays, N_samples, 1), only the (N_rays, -1) represents the sun visibility

    if model.indirect_light == True:
        sky_rgb = out[..., 5:8]  # (N_rays, N_samples, 3)
        idx += 3
    if model.beta == True: 
        uncertainty = out[..., idx:idx+1]
        idx += 1
    if nr_an_on == True:
        normal_an = out[..., idx:idx+3]     # (N_rays, N_samples, 3)
        idx += 3

    if nr_lr_on == True:
        normal_lr = out[..., idx:idx+3]     # (N_rays, N_samples, 3)
        idx += 3

    if apply_brdf == True:
        if model.roughness == True:
            roughness = out[..., idx:idx+1]
            idx += 1
        elif model.RPV == True:
            if model.args.funcM == True:
                rpv_k = out[..., idx:idx+3]
                idx += 3
            if model.args.funcF == True:
                rpv_theta = out[..., idx:idx+3]
                idx += 3
            if model.args.funcH == True:
                rpv_rhoc = out[..., idx:idx+3]
                idx += 3
        else:
            if model.args.b == True:
                hpk_b = out[..., idx:idx+3]
                idx += 3
            if model.args.c == True:
                hpk_c = out[..., idx:idx+3]
                idx += 3
            if apply_theta == True and model.args.theta == True:
                hpk_theta = out[..., idx:idx+1]
                idx += 1

    if z_vals.shape[1] == 1:
        weights = torch.ones_like(z_vals)       
        transparency = torch.ones_like(z_vals)
    else:
        alphas, transparency, weights, depth_final = cal_weight(z_vals, sigmas, args)

    albedo_accu = torch.sum(weights.unsqueeze(-1) * albedo, -2)  # (N_rays, 3)
    albedo_accu = torch.clamp(albedo_accu, min=0., max=1.)

    #Simple version (emitted rgb):
    result = {'sigmas': sigmas.unsqueeze(-1),   # (N_rays, N_samples, 1)
              'albedo': albedo,                 # (N_rays, N_samples, 3)
              'albedo_accu': albedo_accu,       # (N_rays, 3)
              'depth': depth_final,             # (N_rays)
              'alphas': alphas,                 # (N_rays, N_samples)
              'weights': weights,               # (N_rays, N_samples)
              'transparency': transparency,     
              'z_vals': z_vals}

    if apply_sun_v == True:
        result['sun'] = sun_v
        result['weights_sc'] = sun_res['weights_sc']
        if model.indirect_light == True:
            result['sky'] = sky_rgb
    else:
        if "sun" in sun_res:        
            result['sun'] = sun_res['sun']
            result['weights_sc'] = sun_res['weights_sc']

    if sort_idx != None:
        result['sort_idx'] = sort_idx
    if z_vals_unsort != None:
        result['z_vals_unsort'] = z_vals_unsort
    if model.beta == True:
        result['beta'] = uncertainty

    normal_none = True
    if nr_an_on == True:
        result['normal_an'] = normal_an
        normal = normal_an
        normal_none = False

    if nr_lr_on == True:
        result['normal_lr'] = normal_lr
        #if both nr_an and nr_lr is available, use nr_lr for calculating BRDF
        if 1: #nr_an_on == False:
            normal = normal_lr
            normal_none = False

    if normal_none == False:
        normal_s = torch.sum(weights.unsqueeze(-1) * normal.reshape(N_rays, N_samples, 3), -2)    #(N_rays, 3)
        normal_s = train_utils.l2_normalize(normal_s)
        train_utils.check_vec0('normal_s', normal_s)
        view_dir = -1 * rays_d      #(N_rays, 3)
        sun_dir = sun_d             #(N_rays, 3)
        dot_product = torch.einsum('ij,ij->ij', [normal_s, view_dir])
        dot_product = torch.einsum('ij->i', [dot_product])
        result['nr_vw'] = dot_product.reshape(N_rays, 1, 1)         #(N_rays, 1)

        dot_product1 = torch.einsum('ij,ij->ij', [normal_s, sun_dir])
        dot_product1 = torch.einsum('ij->i', [dot_product1])
        result['nr_sun'] = dot_product1.reshape(N_rays, 1, 1)         #(N_rays, 1)

        result['hpk_scl'] = 1.0 / (args.hpk_scl * (result['nr_vw'] + result['nr_sun']))

    #if the above values in result are modified, the relevant value should be updated

    irradiance = torch.ones_like(albedo)  # (N_rays, N_samples, 3)
    if (cos_irra_on == True and normal_none == False):
        normal__ = torch.zeros_like(normal)
        normal__[..., -1] = 1.          #use upward normal instead for simplification to avoid noise
        nr_sun = (normal__ * sun_d_.reshape(normal.shape)).sum(dim=-1).unsqueeze(-1)
        irradiance = irradiance * torch.abs(nr_sun)
    elif apply_sun_v == True:
        irradiance = torch.tile(sun_v, (1, 1, 3)) # (N_rays, N_samples, 3)
        if model.indirect_light == True:   #indirect_light is valid only when sun_v is not none
            irradiance = sun_v + (1 - sun_v) * sky_rgb # equation 2 of the s-nerf paper

    albedo_ = albedo * (1 + 2 * model.rgb_padding) - model.rgb_padding
    rgb_final = torch.sum(weights.unsqueeze(-1) * albedo_ * irradiance, -2)  # (N_rays, 3)
    rgb_final = torch.clamp(rgb_final, min=0., max=1.)
    result['rgb'] = rgb_final

    albedo_s = torch.sum(weights.unsqueeze(-1) * albedo_.reshape(N_rays, N_samples, 3), -2)    #(N_rays, 3)

    input_dir=None if rays_d_ is None else rays_d_[i:i + chunk]
    input_sun_dir=None if sun_d_ is None else sun_d_[i:i + chunk]
    input_t=None if rays_t_ is None else rays_t_[i:i + chunk]
    checknan(result, xyz_, input_dir, input_sun_dir, input_t, model)
    if idx == 4:
        return result, brdf_type

    lvis = False  #if True, calculate light visibility in microfacet BRDF Geometry function
    #Microfacet BRDF version
    if model.roughness == True and apply_brdf == True:
        brdf_type = 'Microfacet'

        if model.MultiBRDF == True:   #one BRDF for each sample
            brdf_glossy, brdf, f, g, d, l_dot_n, v_dot_n, h, n_h = eval_microfacet_brdf(sun_d_.unsqueeze(1), -1*rays_d_, normal.reshape(-1, 3), albedo.reshape(-1, 3), roughness.reshape(-1, 1), lvis=lvis, glossy_scale=model.glossy_scale, print_debuginfo=print_debuginfo, fresnel_f0=args.fresnel_f0, mode=mode)   #here we use inversed view direction
        else:    #one BRDF for each ray
            roughness_s = torch.sum(weights * roughness.reshape(N_rays, N_samples), -1).unsqueeze(-1)    #(N_rays, 1)
            brdf_glossy, brdf, f, g, d, l_dot_n, v_dot_n, h, n_h = eval_microfacet_brdf(sun_d.unsqueeze(1), -1*rays_d, normal_s, albedo_s, roughness_s, lvis=lvis, glossy_scale=model.glossy_scale, print_debuginfo=print_debuginfo, fresnel_f0=args.fresnel_f0, mode=mode)
    #RPV BRDF version
    elif (model.RPV == True and apply_brdf == True):
        brdf_type = 'RPV'
        if model.MultiBRDF == True:   #one BRDF for each sample
            if apply_brdf == False:
                rpv_k_ = None
                rpv_theta_ = None
                rpv_rhoc_ = None
            else:
                rpv_k_ = rpv_k.reshape(-1, 3) if model.args.funcM == True else None
                rpv_theta_ = rpv_theta.reshape(-1, 3) if model.args.funcF == True else None
                rpv_rhoc_tmp = rpv_rhoc.reshape(-1, 3) if model.args.funcH == True else None
                rpv_rhoc_ = albedo.reshape(-1, 3) if model.args.funcH == 2 else rpv_rhoc_tmp
            brdf, M1, G, H, ci, cv = eval_RPV(sun_d_.unsqueeze(1), -1*rays_d_, normal.reshape(-1, 3), albedo.reshape(-1, 3), rpv_k_, rpv_theta_, rpv_rhoc_, mode=mode)
        else:
            if apply_brdf == False:
                k_s = None
                theta_s = None
                rhoc_s = None
            else:
                k_s = torch.sum(weights.unsqueeze(-1) * rpv_k.reshape(N_rays, N_samples, 3), -2)
                theta_s = torch.sum(weights.unsqueeze(-1) * rpv_theta.reshape(N_rays, N_samples, 3), -2) if model.args.funcF == True else None
                rhoc_s_tmp = torch.sum(weights.unsqueeze(-1) * rpv_rhoc.reshape(N_rays, N_samples, 3), -2) if model.args.funcH == True else None
                rhoc_s = albedo_s if model.args.funcH == 2 else rhoc_s_tmp
            brdf, M1, G, H, ci, cv = eval_RPV(sun_d.unsqueeze(1), -1*rays_d, normal_s, albedo_s, k_s, theta_s, rhoc_s, mode=mode)
    #Hapke BRDF version
    elif (apply_brdf == True and model.args.b == True) or model.args.shell_hapke > 0: #shell_hapke can be launched with apply_brdf=False
        brdf_type = 'Hapke'
        h = B0 = None
        h_s = B0_s = None
        if apply_brdf == True and model.args.b == True:
            b = hpk_b.reshape(-1, 3)
            b_s = torch.sum(weights.unsqueeze(-1) * hpk_b.reshape(N_rays, N_samples, 3), -2)    #(N_rays, 3)
        else:
            b = None
            b_s = None
        if apply_brdf == True and model.args.c == True:
            c = hpk_c.reshape(-1, 3)
            c_s = torch.sum(weights.unsqueeze(-1) * hpk_c.reshape(N_rays, N_samples, 3), -2)    #(N_rays, 3)
        else:
            c = None
            c_s = None
        if apply_theta == True and model.args.theta == True:
            theta = hpk_theta.squeeze()
            theta_s = torch.sum(weights * hpk_theta.reshape(N_rays, N_samples), -1)    #(N_rays)
        else:
            theta = None
            theta_s = None

        if model.MultiBRDF == True:   #one BRDF for each sample
            brdf, P, B, Hi, Hv, ShadFunc, ci, cv = eval_Hapke(sun_d_.unsqueeze(1), -1*rays_d_, normal.reshape(-1, 3), albedo.reshape(-1, 3), b, c, theta, h, B0, mode=mode, args=model.args)
        else:
            brdf, P, B, Hi, Hv, ShadFunc, ci, cv = eval_Hapke(sun_d.unsqueeze(1), -1*rays_d, normal_s, albedo_s, b_s, c_s, theta_s, h_s, B0_s, mode=mode, args=model.args)

    if apply_brdf == True or model.args.shell_hapke > 0:
        if model.MultiBRDF == True:
            brdf = brdf.reshape(N_rays, N_samples, 3)
            brdf = brdf * (1 + 2 * model.rgb_padding) - model.rgb_padding
            rgb_final = torch.sum(weights.unsqueeze(-1) * brdf * irradiance, -2)  # (N_rays, 3)
        else:
            rgb_final = irradiance[:, -1, :].reshape(N_rays, 3) * brdf.reshape(N_rays, 3) # (N_rays, 3)

    rgb_final = torch.clamp(rgb_final, min=0., max=1.)
    result['rgb'] = rgb_final
    result['irradiance'] = irradiance

    if apply_brdf == True:
        N_samples_BRDF = N_samples
        if model.MultiBRDF == False:
            N_samples_BRDF = 1

        if model.roughness == True:
            result['roughness'] = roughness
            result['glossy'] = brdf_glossy.reshape(N_rays, N_samples_BRDF, 1)
            result['brdf'] = brdf.reshape(N_rays, N_samples_BRDF, 3)
            result['f'] = f.reshape(N_rays, N_samples_BRDF, 1)
            result['g'] = g.reshape(N_rays, N_samples_BRDF, 1)
            result['d'] = d.reshape(N_rays, N_samples_BRDF, 1)
            result['l_dot_n'] = l_dot_n.reshape(N_rays, N_samples_BRDF, 1)
            result['v_dot_n'] = v_dot_n.reshape(N_rays, N_samples_BRDF, 1)
            result['halfvec'] = h.reshape(N_rays, N_samples_BRDF, 3)
            result['n_h'] = n_h.reshape(N_rays, N_samples_BRDF, 1)
        elif model.RPV == True:
            if model.args.funcM == True:
                result['rpv_k'] = rpv_k
            if model.args.funcF == True:
                result['rpv_theta'] = rpv_theta
            if model.args.funcH == True:
                result['rpv_rhoc'] = rpv_rhoc
        elif model.args.b == True or model.args.shell_hapke > 0:
            result['brdf'] = brdf.reshape(N_rays, N_samples_BRDF, 3)
            result['hpk_P'] = P.reshape(N_rays, N_samples_BRDF, 3)
            result['hpk_Hi'] = Hi.reshape(N_rays, N_samples_BRDF, 3)
            result['hpk_Hv'] = Hi.reshape(N_rays, N_samples_BRDF, 3)
            result['hpk_ci'] = ci.reshape(N_rays, N_samples_BRDF, 1)
            result['hpk_cv'] = cv.reshape(N_rays, N_samples_BRDF, 1)
            result['hpk_ShadFunc'] = ShadFunc.reshape(N_rays, N_samples_BRDF, 1)
            if model.args.b == True:
                result['hpk_b'] = hpk_b
            if model.args.c == True:
                result['hpk_c'] = hpk_c
            if apply_theta == True and model.args.theta == True:
                result['hpk_theta'] = hpk_theta

    if rays_d != None:
        result['rays_d'] = -1 * rays_d.reshape(N_rays, 1, 3)

    if sun_d != None:
        result['sun_d'] = sun_d.reshape(N_rays, 1, 3)

    if rows != None and cols != None:
        rows = torch.tile(rows, (1, N_samples, 1)).permute(2, 1, 0)
        cols = torch.tile(cols, (1, N_samples, 1)).permute(2, 1, 0)
        ref_sphere = torch.ones_like(rays_d.reshape(N_rays, 1, 3))
        N_samples_r = N_samples
        ref_sphere[:,:,0] = cols.reshape(N_rays, N_samples_r)[:,0].reshape(N_rays, 1)
        ref_sphere[:,:,1] = -rows.reshape(N_rays, N_samples_r)[:,0].reshape(N_rays, 1) #rows: [-1, 1], here we reverse rows to match the origin of tensorboard
        ref_sphere[:,:,2] = torch.sqrt(torch.abs(1-rows*rows-cols*cols)).reshape(N_rays, N_samples_r)[:,0].reshape(N_rays, 1)
        result['ref_sphere'] = ref_sphere

    checknan(result, xyz_, input_dir, input_sun_dir, input_t, model)

    return result, brdf_type

class SpSBRDFNeRF(torch.nn.Module):
    def check_nan_parms(self, keyword=''):
        for name, parms in self.named_parameters():
            _, contain_nan = train_utils.check_nan(parms.data, func=keyword + ' check_nan_parms ' + name)
            if contain_nan == True:
                self.print_parms()
                break

    def print_parms(self, only_name=False):
        para_nb = 0
        for name, parms in self.named_parameters():
            str_print = '{} | gra {} | '.format(name, parms.requires_grad)
            if only_name == False:
                train_utils.PrintMMM(str_print, parms.data, places=5)
            else:
                print(str_print)

            if parms.grad != None:
                train_utils.PrintMMM('', parms.grad, Print=False)

            para_nb_cur = 1
            for i in range(len(parms.data.shape)):
                para_nb_cur *= parms.data.shape[i]
            para_nb += para_nb_cur

        print('Total parameter number: ', para_nb)

    def __init__(self, args, layers=8, feat=256, mapping=False, mapping_sizes=[10, 4], skips=[4], siren=True, t_embedding_dims=16, beta=True, roughness = True, normal = 'none', sun_v = 'none', indirect_light=False, glossy_scale=1., MultiBRDF=False, dim_RPV=3):
        '''
        ModifNet = False
        if ModifNet == True:
            skips = []
            feat = 256
            siren = True
        '''
        super(SpSBRDFNeRF, self).__init__()
        self.layers = layers
        self.skips = skips
        self.t_embedding_dims = t_embedding_dims
        self.mapping = mapping
        self.input_sizes = [3, 3] if args.input_viewdir == True else [3, 0]
        self.rgb_padding = 0.001
        self.beta = beta
        self.roughness = roughness
        self.sun_v = sun_v
        self.indirect_light = indirect_light
        self.normal = normal
        self.glossy_scale = glossy_scale
        self.MultiBRDF = MultiBRDF
        self.args = args
        self.RPV = True if (self.args.funcM == True or self.args.funcF == True or self.args.funcH == True) else False
        print('SpSBRDFNeRF: layers, feat, skips, mapping, siren, mapping_sizes, RPV: ', layers, feat, skips, mapping, siren, mapping_sizes, self.RPV)   #8 512 [4] True True [10, 4] False

        self.number_of_outputs = 4 # rgb (3) + sigma (1)
        if self.sun_v == 'learned':
            self.number_of_outputs += 1   # + sun visibility (1)
        if self.indirect_light == True:
            self.number_of_outputs += 3   # + rgb from sky color (3)
        if self.beta == True:
            self.number_of_outputs += 1 # + beta (1)

        self.number_of_outputs_brdf = self.number_of_outputs

        self.dim_RPV = dim_RPV
        # Microfacet
        if self.roughness == True:
            self.number_of_outputs_brdf += 1 # + roughness (1)
        elif self.RPV == True:
            if self.args.funcM == True:
                self.number_of_outputs_brdf += 3
            if self.args.funcF == True:
                self.number_of_outputs_brdf += 3
            if self.args.funcH == True:
                self.number_of_outputs_brdf += 3
        else:
            if self.args.b == True:
                self.number_of_outputs_brdf += 3 # + b (3)
            if self.args.c == True:
                self.number_of_outputs_brdf += 3 # + c (3)

        print("self.number_of_outputs_brdf, self.number_of_outputs", self.number_of_outputs_brdf, self.number_of_outputs)

        # activation function
        nl = Siren() if siren else torch.nn.ReLU()

        # use positional encoding if specified
        in_size = self.input_sizes.copy()
        print('*init spsbrdfnerf*, in_size before mapping: ', in_size)
        if mapping:
            self.mapping = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(mapping_sizes, self.input_sizes)]
            in_size = [2 * map_sz * in_sz for map_sz, in_sz in zip(mapping_sizes, self.input_sizes)]
        else:
            self.mapping = [torch.nn.Identity(), torch.nn.Identity()]
        print('*init spsbrdfnerf*, in_size after mapping: ', in_size)

        # define the main network of fully connected layers, i.e. FC_NET
        fc_layers = []
        fc_layers.append(torch.nn.Linear(in_size[0], feat))
        fc_layers.append(Siren(w0=30.0) if siren else nl)
        for i in range(1, layers):
            if i in skips:
                print(i, ' in skips')
                fc_layers.append(torch.nn.Linear(feat + in_size[0], feat))
            else:
                fc_layers.append(torch.nn.Linear(feat, feat))
            fc_layers.append(nl)
        self.fc_net = torch.nn.Sequential(*fc_layers)  # shared 8-layer structure that takes the encoded xyz vector

        # FC_NET output 1: volume density
        self.sigma_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, 1), torch.nn.Softplus())

        # FC_NET output 2: vector of features from the spatial coordinates
        self.feats_from_xyz = torch.nn.Linear(feat, feat) # No non-linearity here in the original paper

        # the FC_NET output 2 is concatenated to the encoded viewing direction input
        # and the resulting vector of features is used to predict the rgb color
        self.rgb_from_xyzdir = torch.nn.Sequential(torch.nn.Linear(feat + in_size[1], feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, 3), torch.nn.Sigmoid()) #[0,1]

        if siren:
            self.fc_net.apply(sine_init)
            self.fc_net[0].apply(first_layer_sine_init)

        print('net_version: ')
        print('  component 1. simple version:   sigma + emitted rgb')

        sun_dir_in_size = 3
        if self.sun_v == 'learned':
            sun_v_layers = []
            sun_v_layers.append(torch.nn.Linear(feat + sun_dir_in_size, feat // 2))
            sun_v_layers.append(Siren() if siren else nl)
            for i in range(1, 3):
                sun_v_layers.append(torch.nn.Linear(feat // 2, feat // 2))
                sun_v_layers.append(nl)
            sun_v_layers.append(torch.nn.Linear(feat // 2, 1))
            sun_v_layers.append(torch.nn.Sigmoid()) #[0,1]
            self.sun_v_net = torch.nn.Sequential(*sun_v_layers)

            if siren:
                self.sun_v_net.apply(sine_init)
                self.sun_v_net[0].apply(first_layer_sine_init)

            print('  component 2. learned shadow:   + sun_v')

        if self.indirect_light == True:
            self.sky_color = torch.nn.Sequential(
                torch.nn.Linear(sun_dir_in_size, feat // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(feat // 2, 3),
                torch.nn.Sigmoid(),
            ) #[0,1]
            print('  component 3. indirect_light:   + sky_rgb')

        if self.beta == True:
            self.beta_from_xyz = torch.nn.Sequential(
                torch.nn.Linear(self.t_embedding_dims + feat, feat // 2), nl,
                torch.nn.Linear(feat // 2, 1), torch.nn.Softplus()) #[0,+00)
            print('  component 4. transient scalar: + beta')

        if self.normal == 'analystic_learned' or self.normal == 'learned':
            self.grad_from_xyz = torch.nn.Linear(feat, 3)
            print('  component 5. learned normal:   + normal')

        # new layers to predict roughness
        if self.roughness == True:
            self.roughness_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, 1), torch.nn.Sigmoid()) #[0,1]
            print('  component 6. roughness:        + roughness')

        #if self.args.RPV == True:
        if self.args.funcM == True:
            self.k_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, self.dim_RPV), torch.nn.Sigmoid()) #[0,1]
            print('  component 7. rpv:        + k')
        if self.args.funcF == True:
            self.theta_rpv_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, self.dim_RPV), torch.nn.Sigmoid()) #[0,1]
            print('  component 7. rpv:        + theta_rpv')
        if self.args.funcH == True:
            self.rhoc_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, self.dim_RPV), torch.nn.Sigmoid()) #[0,1]
            print('  component 7. rpv:        + rhoc')

        # new layers to predict b for Hapke model
        if self.args.b == True:
            self.b_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, 1), torch.nn.Sigmoid()) #[0,1]
            print('  component 8. Hapke:        + b')
        if self.args.c == True:
            self.c_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, 1), torch.nn.Sigmoid()) #[0,1]
            print('  component 8. Hapke:        + c')
        if self.args.theta == True:
            self.theta_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, 1), torch.nn.Sigmoid()) #[0,1]
            print('  component 8. Hapke:        + theta')

        self.print_parms()

    def freeze(self, layer_name):
        for name, parms in self.named_parameters():
            if (layer_name in name) or layer_name=='all':
                parms.requires_grad = False
                print(name, ' freezed: ')

    def unfreeze(self, layer_name):
        for name, parms in self.named_parameters():
            if layer_name in name:
                parms.requires_grad = True
                print(name, ' unfreezed: ')

    def freeze_rest(self, layer_name):
        for name, parms in self.named_parameters():
            if layer_name not in name:
                parms.requires_grad = False
                print(name, ' freezed: ')

    # compute shared features
    def calc_features(self, input_xyz):
        input_xyz_ = self.mapping[0](input_xyz)
        xyz_ = input_xyz_
        for i in range(self.layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz_, xyz_], -1)
            xyz_ = self.fc_net[2*i](xyz_)
            xyz_ = self.fc_net[2*i + 1](xyz_)
        shared_features = xyz_

        return shared_features

    def calc_normals(self, input_xyz, graph=True):
        with torch.enable_grad():
            input_xyz.requires_grad_()
            sigma = self.sigma_from_xyz(self.calc_features(input_xyz))
            grad_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)
            gradients = torch.autograd.grad(
                outputs=sigma,
                inputs=input_xyz,
                grad_outputs=grad_output,
                create_graph=graph,
                retain_graph=graph,
                only_inputs=True, allow_unused=graph)[0]
            return gradients

    def forward(self, input_xyz_, input_dir=None, input_sun_dir=None, input_t=None, sigma_only=False, apply_brdf=False, apply_theta=False, nr_an_on=False, nr_lr_on=False, sun_ray=False, mode='train'):
        """
        Predicts the values rgb, sigma from a batch of input rays
        the input rays are represented as a set of 3d points xyz

        Args:
            input_xyz_: (B, 3) input tensor, with the 3d spatial coordinates, B is batch size
            sigma_only: boolean, infer sigma only if True, otherwise infer both sigma and color

        Returns:
            if sigma_ony:
                sigma: (B, 1) volume density
            else:
                out: (B, 4) first 3 columns are rgb color, last column is volume density
        """
        shared_features = self.calc_features(input_xyz_)

        assert torch.isnan(shared_features).sum() == 0, self.print_parms()

        # compute volume density
        sigma = self.sigma_from_xyz(shared_features)

        if sigma_only:
            return sigma

        # compute color
        xyz_features = self.feats_from_xyz(shared_features)
        if self.input_sizes[1] > 0:  #and input_dir != None:
            input_xyzdir = torch.cat([xyz_features, self.mapping[1](input_dir)], -1)
        else:
            input_xyzdir = xyz_features
        rgb = self.rgb_from_xyzdir(input_xyzdir)
        out = torch.cat([rgb, sigma], 1) # (B, 4)

        if self.sun_v == 'learned':
            input_sun_v_net = torch.cat([xyz_features_, input_sun_dir], -1)
            sun_v = self.sun_v_net(input_sun_v_net)
            out = torch.cat([out, sun_v], 1) # (B, 5)

        if sun_ray == True:
            return out

        if self.indirect_light == True:
            sky_color = self.sky_color(input_sun_dir)
            out = torch.cat([out, sky_color], 1) # (B, 8)

        if self.beta == True:
            input_for_beta = torch.cat([xyz_features, input_t], -1)
            beta = self.beta_from_xyz(input_for_beta)
            out = torch.cat([out, beta], 1) # (B, 9)

        if nr_an_on == True:
            grad_an = self.calc_normals(input_xyz_)
            normal_an = -train_utils.l2_normalize(grad_an, keyword='nr_an_ori')
            out = torch.cat([out, normal_an], 1)
        if nr_lr_on == True: #self.normal == 'analystic_learned' or self.normal == 'learned':
            grad_lr = self.grad_from_xyz(shared_features)
            normal_lr = -train_utils.l2_normalize(grad_lr)
            out = torch.cat([out, normal_lr], 1)

        if apply_brdf == True:
            if self.roughness == True:
                # compute roughness
                roughness = self.roughness_from_xyz(xyz_features)
                out = torch.cat([out, roughness], 1) # (B, 10) or (B, 9)
            elif self.RPV == True:
                if self.args.funcM == True:
                    k = self.k_from_xyz(xyz_features)
                    k = (k - 0.5) * 2 + 1                   #scale to [0, 2]
                    k = torch.tile(k, (1, 3)) if k.shape[1] == 1 else k
                    out = torch.cat([out, k], 1)
                if self.args.funcF == True:
                    theta_rpv = self.theta_rpv_from_xyz(xyz_features)
                    theta_rpv = (theta_rpv - 0.5) * 2       #scale to [-1, 1]
                    theta_rpv = torch.tile(theta_rpv, (1, 3)) if theta_rpv.shape[1] == 1 else theta_rpv
                    out = torch.cat([out, theta_rpv], 1)
                if self.args.funcH == True:
                    rhoc = self.rhoc_from_xyz(xyz_features)
                    #rhoc = (rhoc - 0.5) * 2 + 1             #scale to [0, 2]
                    rhoc = torch.tile(rhoc, (1, 3)) if rhoc.shape[1] == 1 else rhoc
                    out = torch.cat([out, rhoc], 1)
            else:
                if self.args.b == True:
                    b = self.b_from_xyz(xyz_features)   # [B, 1 or 3]
                    b = torch.tile(b, (1, 3)) if b.shape[1] == 1 else b
                    out = torch.cat([out, b], 1)
                if self.args.c == True:
                    c = self.c_from_xyz(xyz_features)   # [B, 1 or 3]
                    c = torch.tile(c, (1, 3)) if c.shape[1] == 1 else c
                    out = torch.cat([out, c], 1)
                if apply_theta == True and self.args.theta == True:
                    theta = self.theta_from_xyz(xyz_features)   # [B, 1]
                    theta = theta * (np.pi * 30.0 / 180.0)      #scale to [0-30] degre
                    out = torch.cat([out, theta], 1)

        return out

