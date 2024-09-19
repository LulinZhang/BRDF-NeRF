"""
This script defines the evaluation metrics and the loss functions
"""

import torch
import numpy as np
from kornia.losses import ssim as ssim__
import train_utils


class NerfLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        loss_dict['coarse_color'] = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss_dict['fine_color'] = self.loss(inputs['rgb_fine'], targets)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

def uncertainty_aware_loss(loss_dict, inputs, gt_rgb, typ, beta_min=0.05):
    beta = torch.sum(inputs[f'weights_{typ}'].unsqueeze(-1) * inputs['beta_coarse'], -2) + beta_min
    loss_dict[f'{typ}_color'] = ((inputs[f'rgb_{typ}'] - gt_rgb) ** 2 / (2 * beta ** 2)).mean()
    loss_dict[f'{typ}_logbeta'] = (3 + torch.log(beta).mean()) / 2  # +3 to make c_b positive since beta_min = 0.05
    return loss_dict

def solar_correction(loss_dict, inputs, typ, lambda_sc=0.05):
    # computes the solar correction terms defined in Shadow NeRF and adds them to the dictionary of losses
    sun_sc = inputs[f'sun_sc_{typ}'].squeeze()
    term2 = torch.sum(torch.square(inputs[f'transparency_sc_{typ}'].detach() - sun_sc), -1)
    term3 = 1 - torch.sum(inputs[f'weights_sc_{typ}'].detach() * sun_sc, -1)
    loss_dict[f'{typ}_sc_term2'] = lambda_sc/3. * torch.mean(term2)
    loss_dict[f'{typ}_sc_term3'] = lambda_sc/3. * torch.mean(term3)
    return loss_dict

class SNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.05, lambda_rgb=1.):
        super().__init__()
        self.lambda_sc = lambda_sc
        self.lambda_rgb = lambda_rgb
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, exp=0):
        loss_dict = {}
        typ = 'coarse'
        loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_rgb * loss_dict[k]
        loss = sum(l for l in loss_dict.values())

        return loss, loss_dict

class SatNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.0):
        super().__init__()
        self.lambda_sc = lambda_sc

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

class DepthLoss(torch.nn.Module):
    def __init__(self, lambda_ds=1.0, GNLL=False, usealldepth=True, margin=0, stdscale=1, subset=False):
        super().__init__()
        self.lambda_ds = lambda_ds/3.
        self.GNLL = GNLL
        self.usealldepth = usealldepth
        self.margin=margin
        self.stdscale=stdscale
        self.subset=subset
        if self.GNLL == True:
            self.loss = torch.nn.GaussianNLLLoss()
        else:
            self.loss = torch.nn.MSELoss(reduce=False)

    def forward(self, inputs, targets, weights=1., target_valid_depth=None, target_std=None):

        def is_not_in_expected_distribution(pred_depth, pred_std, target_depth, target_std):
            depth_greater_than_expected = ((pred_depth - target_depth).abs() - target_std) > 0.
            std_greater_than_expected = target_std < pred_std
            return torch.logical_or(depth_greater_than_expected, std_greater_than_expected)

        def ComputeSubsetDepthLoss(inputs, typ, target_depth, target_weight, target_valid_depth, target_std):
            if target_valid_depth == None:
                print('target_valid_depth is None! Use all the target_depth by default! target_depth.shape[0]', target_depth.shape[0])
                target_valid_depth = torch.ones(target_depth.shape[0])
            z_vals = inputs[f'z_vals_{typ}'][np.where(target_valid_depth.cpu()>0)]
            pred_depth = inputs[f'depth_{typ}'][np.where(target_valid_depth.cpu()>0)]

            pred_weight = inputs[f'weights_{typ}'][np.where(target_valid_depth.cpu()>0)]
            if pred_depth.shape[0] == 0:
                print('ZERO target_valid_depth in this depth loss computation! target_weight.device: ', target_weight.device)
                return torch.zeros((1,), device=target_weight.device, requires_grad=True)

            pred_std = train_utils.calc_depth_std(z_vals, pred_depth, pred_weight)
            target_weight = target_weight[np.where(target_valid_depth.cpu()>0)]
            target_depth = target_depth[np.where(target_valid_depth.cpu()>0)]
            target_std = target_std[np.where(target_valid_depth.cpu()>0)]

            apply_depth_loss = torch.ones(target_depth.shape[0])
            if self.usealldepth == False:
                apply_depth_loss = is_not_in_expected_distribution(pred_depth, pred_std, target_depth, target_std)

            pred_depth = pred_depth[apply_depth_loss]
            if pred_depth.shape[0] == 0:
                print('ZERO apply_depth_loss in this depth loss computation!')
                return torch.zeros((1,), device=target_weight.device, requires_grad=True)

            pred_std = pred_std[apply_depth_loss]
            target_depth = target_depth[apply_depth_loss]

            numerator = float(pred_depth.shape[0])
            denominator = float(target_valid_depth.shape[0])

            if self.GNLL == True:   
                loss = numerator/denominator*self.loss(pred_depth, target_depth, pred_std)
                return loss
            else:
                loss = numerator/denominator*target_weight[apply_depth_loss]*self.loss(pred_depth, target_depth)
                return loss

        loss_dict = {}
        typ = 'coarse'
        if self.subset == True:
            loss_dict[f'{typ}_ds'] = ComputeSubsetDepthLoss(inputs, typ, targets, weights, target_valid_depth, target_std)
        else:
            loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_coarse'], targets)

        if 'depth_fine' in inputs:
            typ = 'fine'
            if self.subset == True:
                loss_dict[f'{typ}_ds'] = ComputeSubsetDepthLoss(inputs, typ, targets, weights, target_valid_depth, target_std)
            else:
                loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_coarse'], targets)

        # no need to apply weights here because it is already done in function ComputeSubsetDepthLoss
        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_ds * torch.mean(loss_dict[k])

        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

def load_loss(args):
    if args.model == "nerf":
        loss_function = NerfLoss()
    elif args.model == "s-nerf":
        loss_function = SNerfLoss(lambda_sc=args.sc_lambda)
    elif args.model == "sat-nerf" or args.model == "sps-nerf":  #or args.model == "spsbrdf-nerf":
        if args.beta == True:
            loss_function = SatNerfLoss(lambda_sc=args.sc_lambda)
        else:
            loss_function = SNerfLoss(lambda_sc=args.sc_lambda)  
    elif args.model == "spsbrdf-nerf":
        loss_function = SNerfLoss(lambda_sc=args.sc_lambda, lambda_rgb=args.lambda_rgb)
    else:
        raise ValueError(f'model {args.model} is not valid')
    return loss_function

class NormalRegLoss(torch.nn.Module):
    def __init__(self, lambda_nr_reg=0.1, keyword='normal_an'):      
        super().__init__()
        self.lambda_nr_reg = lambda_nr_reg
        self.keyword = keyword

    def forward(self, inputs):

        def normal_regularization(loss_dict, inputs, typ):
            normal = inputs[f'{self.keyword}_{typ}']
            normal = normal.reshape(-1,3)                       #[N_rays * N_samples, 3]
            weights = inputs[f"weights_{typ}"].reshape(-1,1).squeeze()   #[N_rays * N_samples]
            view_dir = inputs[f'rays_d_{typ}'].squeeze()   #[N_rays, 3], here it is facing toward the camera, because it is flipped when assigned to inputs
            repeat_num = int(normal.size(0)/view_dir.size(0))
            view_dir_ = torch.repeat_interleave(view_dir, repeats=repeat_num, dim=0) #[N_rays * N_samples, 3], facing toward the camera
            n_dot_v = (normal * view_dir_).sum(dim=-1)  #[N_rays * N_samples]

            count_bad_nr = (n_dot_v < 0).sum().item()
            count_total = n_dot_v.numel()
            perc_ng_nr = torch.tensor(count_bad_nr * 100.0 / count_total) #, device=normal.device)
            
            zero = torch.zeros_like(n_dot_v)
            loss_dict[f'{typ}_nr_reg_{self.keyword[-2:]}'] = (weights * (torch.minimum(zero, n_dot_v)**2)).sum(dim=-1)

            return loss_dict, perc_ng_nr

        loss_dict = {}
        typ = 'coarse'
        loss_dict, perc_ng_nr = normal_regularization(loss_dict, inputs, typ)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict, perc_ng_nr = normal_regularization(loss_dict, inputs, typ)

        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_nr_reg * torch.mean(loss_dict[k]) #* (perc_ng_nr/100.)

        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict, perc_ng_nr

class NormalLoss(torch.nn.Module):
    def __init__(self, lambda_nr_spv=0.001, bend_nr_lr=False):
        super().__init__()
        self.lambda_nr_spv = lambda_nr_spv
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.bend_nr_lr = bend_nr_lr

    def forward(self, weights, normal_gt, normal_pred, target_weight=None, target_valid_depth=None, keyword='an_lr'): #inputs):
        """
        weights:        [N_rays, N_samples]
        normal_gt:      [N_rays, N_samples, 3] or [N_rays, 3]
        normal_pred:    [N_rays, N_samples, 3]
        """
        def calc_subset_normal_loss(weights, target_normal, normal_pred, typ, target_weight=None, target_valid_depth=None, keyword='an_lr'):
            normal_pred_s = torch.sum(weights.unsqueeze(-1) * normal_pred, -2)
            if target_valid_depth == None:
                print('target_valid_depth is None! Use all the target_depth by default! target_depth.shape[0]', target_depth.shape[0])
                target_valid_depth = torch.ones(target_depth.shape[0])

            target_weight = target_weight[np.where(target_valid_depth.cpu()>0)]
            normal_pred_s = normal_pred_s[np.where(target_valid_depth.cpu()>0)]
            target_normal = target_normal[np.where(target_valid_depth.cpu()>0)]

            loss = self.l1_loss(target_weight.unsqueeze(-1)*target_normal, target_weight.unsqueeze(-1)*normal_pred_s) #/ float(target_normal.shape[0])
            return loss

        def calc_normal_loss(loss_dict, weights, normal_gt, normal_pred, typ, target_weight=None, target_valid_depth=None, keyword='an_lr'):
            if keyword == 'an_lr':
                loss_dict[f'{typ}_nrspv_{keyword}'] = weights.reshape(-1) * (self.l1_loss(normal_gt, normal_pred))
            else:
                loss_dict[f'{typ}_nrspv_{keyword}'] = calc_subset_normal_loss(weights, normal_gt, normal_pred, typ, target_weight=target_weight, target_valid_depth=target_valid_depth, keyword=keyword)

            return loss_dict

        loss_dict = {}
        typ = 'coarse'
        loss_dict = calc_normal_loss(loss_dict, weights, normal_gt, normal_pred, typ, target_weight=target_weight, target_valid_depth=target_valid_depth, keyword=keyword)

        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_nr_spv * torch.mean(loss_dict[k])

        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

class HardSurfaceLoss(torch.nn.Module):
    def __init__(self, lambda_hs=0.5):      #(zlltocheck) 0.5 comes from nowhere, should finetune
        super().__init__()
        self.lambda_hs = lambda_hs

    def forward(self, inputs):

        def hardsurface_regularization(loss_dict, inputs, typ):
            z_vals = inputs[f'z_vals_{typ}']
            pred_depth = inputs[f'depth_{typ}']
            pred_weight = inputs[f'weights_{typ}']
            sampling_var = train_utils.calc_depth_std_2(z_vals, pred_depth, pred_weight)
            loss_dict[f'{typ}_hs_reg'] = sampling_var

            return loss_dict

        loss_dict = {}
        typ = 'coarse'
        loss_dict = hardsurface_regularization(loss_dict, inputs, typ)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict = hardsurface_regularization(loss_dict, inputs, typ)

        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_hs * torch.mean(loss_dict[k])

        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    value = value / ((torch.max(image_gt))**2)
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def sclimg(img1, img2, Print=False):
    min_ = 0.
    max_ = torch.max(img2)
    if Print == True:
        print("sclimg: [{:.3f}, {:.3f}] scaled by factor {:.3f}".format(min_, max_, 1./(max_ - min_)))
    img1_ = (img1 - min_)/(max_ - min_)
    img2_ = (img2 - min_)/(max_ - min_)
    return img1_, img2_

def psnr_(image_pred, image_gt, valid_mask=None, reduction='mean', Print=False):
    psnr_1 = -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))
    if Print == True:
        psnr_2 =  20*torch.log10(torch.max(image_gt))
        psnr_3 =  20*torch.log10(torch.max(image_pred))
        print('psnr_1, psnr_2, psnr_3: {:.3f}, {:.3f}, {:.3f}'.format(psnr_1, psnr_2, psnr_3))
    return psnr_1 #+ psnr_2

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean', Print=False, scl=False):
    psnr = psnr_(image_pred, image_gt, valid_mask=valid_mask, reduction=reduction, Print=Print)
    if scl == True:
        image_pred_, image_gt_ = sclimg(image_pred, image_gt, Print=Print)
        psnr_scl = psnr_(image_pred_, image_gt_, valid_mask=valid_mask, reduction=reduction, Print=Print)
    else:
        psnr_scl = -1
    return psnr, psnr_scl

def ssim_(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    max_val = np.max(image_gt.cpu().numpy()).astype(float)
    return torch.mean(ssim__(image_pred, image_gt, 3, max_val=max_val))

def ssim(image_pred, image_gt, scl=False):
    ssim = ssim_(image_pred, image_gt)
    if scl == True:
        image_pred_, image_gt_ = sclimg(image_pred, image_gt)
        ssim_scl = ssim_(image_pred_, image_gt_)
    else:
        ssim_scl = -1
    return ssim, ssim_scl

"""
def lpips(image_pred, image_gt):
    from lpips import LPIPS as lpips_
    #image_pred and image_gt: (1, 3, H, W)
    lpips_func = lpips_()
    return lpips_func(image_pred, image_gt, normalize=True)[0]
"""
