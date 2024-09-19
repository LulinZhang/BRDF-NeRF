"""
Additional functions
"""
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import os
import rasterio
import torch

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, StepLR

def check_vec0(keyword, x, Print=False):
    x = x.reshape(-1, 3)
    norm = torch.sqrt(torch.sum(x**2, dim=-1))
    ones = torch.ones_like(norm)
    zeros = torch.zeros_like(norm)

    vec0_ = torch.where(norm>0.99999, ones, zeros)
    vec0_nb = int(torch.sum(ones) - torch.sum(vec0_))
    if vec0_nb > 0 and Print == True:
        places = 3
        print('vecnonunit in {}: {:.0f} / {:.0f} ({:.2f}%)'.format(keyword, vec0_nb, x.shape[0], vec0_nb*100./x.shape[0])) 

    return 100. * vec0_nb / x.shape[0]

def l2_normalize(x, eps=torch.tensor(torch.finfo(torch.float32).eps), keyword=None):
    """Normalize x to unit length along last axis."""
    norm = torch.sum(x**2, dim=-1, keepdims=True)
    res = x / torch.sqrt(torch.maximum(norm, eps))

    return res

def calc_depth_std(z_vals, pred_depth, pred_weight):
    return calc_depth_std_2(z_vals, pred_depth, pred_weight).sqrt()

def calc_depth_std_2(z_vals, pred_depth, pred_weight):
    return ((z_vals - pred_depth.unsqueeze(-1)).pow(2) * pred_weight).sum(-1)

def calc_std(variant):
    me = torch.mean(variant)
    nb_ele = 1
    for i in range(variant.dim()):
        nb_ele *= variant.shape[i]
    std = torch.sqrt(torch.sum((variant - me).pow(2))/(1.*nb_ele))
    return std, nb_ele

def check_badnr(tmp, str='', Print=True):
    weights = torch.ones_like(tmp)
    wei_total = weights.sum()
    if 1:
        weights = torch.where(tmp>0, weights, torch.zeros_like(tmp))
    wei_valid = weights.sum()
    wei_inval = wei_total - wei_valid
    if wei_inval > 0 and Print == True:
        print('{}: {:.0f} / {:.0f}'.format(str, wei_inval, wei_total))

    return wei_total, wei_valid, wei_inval, weights

def check_nan(val_in, val_rep=None, Print=False, func='none'):
    if torch.is_tensor(val_in) == True:
        str_print = '----nan nb in {}, val_in: {:.0f} / {:.0f}'.format(func, torch.isnan(val_in).sum(), torch.ones_like(val_in).sum())

        if torch.isnan(val_in).sum()!=0:
            if val_rep != None:
                val_in_ = torch.where(torch.isnan(val_in)==False, val_in, val_rep)
                str_print += ', val_in_: {:.0f}'.format(torch.isnan(val_in_).sum())
    
                print(str_print)
                return val_in_, True
            else:
                print(str_print)
        else:
            if Print == True:
                print(str_print)

    return val_in, False

def PrintMMM_(str, variant, show_val=False, Print=True, poststr='', places=2):
    details = ''
    postfix = ''
    if torch.is_tensor(variant) == False:
        variant = torch.from_numpy(variant).type(torch.FloatTensor)
        postfix = ' np'

    if 1:
        if variant != None:
            leng = ''
            if variant.shape != None:
                leng = '{}'.format(variant.shape)
                leng = leng[10:]
                #out_num = ':.{}f'.format(place)
                std, nb_ele = calc_std(variant)
                details = 'me {:.{places}f}, std {:.{places}f}, [{:.{places}f}, {:.{places}f}] ({:.{places}f}) | sz {}{}'.format(torch.mean(variant), std, torch.min(variant), torch.max(variant), torch.max(variant)-torch.min(variant), leng, postfix, places=places)
                #details = 'me {out_num}, [{out_num}, {out_num}] ({out_num}) | sz {}'.format(torch.mean(variant), torch.min(variant), torch.max(variant), torch.max(variant)-torch.min(variant), leng)
                #details = 'me {:.2f}, [{:.2f}, {:.2f}] ({:.2f}) | sz {}'.format(torch.mean(variant), torch.min(variant), torch.max(variant), torch.max(variant)-torch.min(variant), leng)
                if Print == True:
                    print('{}: {} {}'.format(str, details, poststr))
                    #print('{}: me {:.2f}, [{:.2f}, {:.2f}], (ran {:.2f}), sz {}'.format(str, torch.mean(variant), torch.min(variant), torch.max(variant), torch.max(variant)-torch.min(variant), leng))
                    if show_val == True:
                        print('  ', variant)
                return torch.mean(variant), torch.min(variant), torch.max(variant), details
        else:
            print(str, 'is None')

    return 0, 0, 0, details

def PrintMMM(str, variant, show_val=False, Print=True, poststr='', places=2, dim=1):
    if dim == 1:
        PrintMMM_(str, variant, show_val=show_val, Print=Print, poststr=poststr, places=places)
    else:
        for i in range(int(dim)):
            str_ = str if i==0 else ' '*len(str)
            PrintMMM_(str_+'->{}'.format(i), variant[..., i], show_val=show_val, Print=Print, poststr=poststr, places=places)

def get_epoch_number_from_train_step(train_step, dataset_len, batch_size):
    return int(train_step // (dataset_len // batch_size))

def get_learning_rate(optimizer):
    """
    Get learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_parameters(models):
    """
    Get all model parameters recursively
    models can be a list, a dictionary or a single pytorch model
    """
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def get_scheduler(optimizer, lr_scheduler, num_epochs):

    eps = 1e-8
    if lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eps)
    elif lr_scheduler == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.01)
    elif lr_scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[2,4,8], gamma=0.5)
    elif lr_scheduler == 'step':
        gamma = 0.9
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    else:
        raise ValueError('lr scheduler not recognized!')

    return scheduler

def get_surface_feature(feature, idx):
    surface_feature = torch.ones_like(feature[:,-1,:])
    for [i, j] in idx:
        for k in range(feature.shape[2]):
            surface_feature[i,k] = feature[i,j,k]

    return surface_feature

def visu_normal(data):
    data = data.cpu().numpy()
    data = (data + 1) * 255 / 2
    img_normal = Image.fromarray((np.clip(data, 0, 255)).astype(np.uint8), 'RGB')
    img_normal = T.ToTensor()(img_normal)

    return img_normal

def calc_normal(depth, scale=-1):
    if scale == -1:
        scale = 100/(np.max(depth) - np.min(depth))
    print("**{}; ori_ran: [{:.3f}, {:.3f}], ori_me: {:.3f}; scale: {}".format("depth", np.min(depth), np.max(depth), np.mean(depth), scale))
    normals = np.zeros((depth.shape[0], depth.shape[1], 3))
    dzdx = (depth[2:, :] - depth[:-2, :]) / 2.0
    dzdy = (depth[:, 2:] - depth[:, :-2]) / 2.0
    normals[1:-1, :, 0] = dzdx
    normals[:, 1:-1, 1] = dzdy  #delete the negative because the depth is reversed to elevation
    normals[:, :, 2] = 1/scale

    normalized_v = np.linalg.norm(normals, axis=2, ord=2)
    for i in range(3):
        normals[:,:,i] = normals[:,:,i] / normalized_v

    img_normal = visu_normal(normals)

    return img_normal

def ToImage(x, norm_type='none', variant='none', tile=False, min_=None, max_=None):
    mi = np.min(x)
    ma = np.max(x)
    if mi < 0 and ma > 0:
        mm = np.min(np.abs(x))
    mean = np.mean(x)
    std = np.std(x)
    if min_!=None and max_!=None:
        min_v, max_v = min_, max_
    elif norm_type == 'minmax':
        min_v, max_v = mi, ma
    elif norm_type == '-1et1':
        min_v, max_v = -1, 1
    elif norm_type == 'enhance':
        min_v, max_v = mean - 3*std, mean + 3*std
    else:
        min_v, max_v = 0, 1
    x_ = (x - min_v) / (max_v - min_v + 1e-8) # normalize to 0~1

    x_ = (np.clip((255*x_), 0, 255)).astype(np.uint8)
    PrintMMM('{}_ori'.format(variant), x, dim=x.shape[-1])
    PrintMMM('{}_scl'.format(variant), x_, dim=x.shape[-1]) 
    x_ = np.clip(x_, 0, 255)
    if x_.shape[-1] == 1:
        if tile == True:
            x_ = np.tile(x_, 3)
        else:
            x_ = cv2.applyColorMap(x_, cv2.COLORMAP_RAINBOW)
    res = Image.fromarray(x_)
    return res

def visualize_accumulated_feature(results, typ, H, W, D, variant, Accum=False, bUnsqz=False, thres=-1, idx=np.random.rand(2,2), norm_type='none', tile=False, min_=None, max_=None):
    feature = results[f'{variant}_{typ}'].clone()
    if bUnsqz == True:
        feature = feature.unsqueeze(-1)
    if idx.shape[0] == results[f'{variant}_{typ}'].shape[0]:
        feature = get_surface_feature(feature, idx)
    if feature.shape[1] == 1:
        Accum = False   # for MultiBRDF = False
    if Accum == True:
        if variant == 'sun':
            feature = torch.sum(results[f"weights_sc_{typ}"].unsqueeze(-1) * feature, -2)  #weights = alphas * transparency
        else:
            #print(variant, "Accum: True", results[f"weights_{typ}"].unsqueeze(-1).shape, feature.shape)
            feature = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * feature, -2)  #weights = alphas * transparency
    else:
        if len(feature.shape) == 3:
            feature = feature[:,-1,:]       #(N_rays, N_samples, D) ->  (N_rays, D)
    feature = feature.view(H, W, D)
    x = feature.cpu().numpy()
    x_ = ToImage(x, norm_type=norm_type, variant=variant, tile=tile, min_=min_, max_=max_)
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def generate_std_img(results, typ, H, W, norm_type='none'):
    z_vals = results[f'z_vals_{typ}']
    pred_depth = results[f'depth_{typ}']
    pred_weight = results[f'weights_{typ}']
    sampling_std = calc_depth_std(z_vals, pred_depth, pred_weight)
    sampling_std = sampling_std.view(H, W, 1)
    x = sampling_std.cpu().numpy()
    x_ = ToImage(x, variant='depth_std', norm_type=norm_type)
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_, sampling_std, np.mean(x)

def visualize_depth(depth, cmap=cv2.COLORMAP_RAINBOW, min_dep=None, max_dep=None, keyword='depth'):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    xstd = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) if min_dep == None else min_dep
    ma = np.max(x) if max_dep == None else max_dep
    me = np.mean(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    print('{}: ori| mi: {:.2f}, ma: {:.2f}, me: {:.2f} |scl mi: {:.2f}, ma: {:.2f}, me: {:.2f}'.format(keyword, mi, ma, me, np.min(x), np.max(x), np.mean(x)))
    x = (np.clip((255*x), 0, 255)).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_, mi, ma

def visualize_diff(diff, cmap=cv2.COLORMAP_RAINBOW, H=-1, W=-1, keyword='diff', min_dep=None, max_dep=None):
    x = diff
    x = np.nan_to_num(x) # change nan to 0
    H = x.shape[0] if H < 0 else H
    W = x.shape[1] if W < 0 else W
    mi = np.min(x) if min_dep == None else min_dep
    ma = np.max(x) if max_dep == None else max_dep
    me = np.mean(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    print('{}: ori| mi: {:.2f}, ma: {:.2f}, me: {:.2f} |scl mi: {:.2f}, ma: {:.2f}, me: {:.2f}'.format(keyword, mi, ma, me, np.min(x), np.max(x), np.mean(x)))
    x = (np.clip((255*x), 0, 255)).astype(np.uint8)
    x = cv2.resize(x, (int(W), int(H)))
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_, mi, ma

def save_output_image_tiff(input, output_path, source_path):
    """
    input: (D, H, W) where D is the number of channels (3 for rgb, 1 for grayscale)
           can be a pytorch tensor or a numpy array
    """
    # convert input to numpy array float32
    if torch.is_tensor(input):
        im_np = input.type(torch.FloatTensor).cpu().numpy()
    else:
        im_np = input.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    im_np = im_np.transpose((1, 2, 0))
    if im_np.shape[-1] == 1:
        im = Image.fromarray(im_np.squeeze(-1).astype(np.float32), mode='L')
    else:
        im = Image.fromarray(im_np.astype(np.float32), mode='RGB')
    im.save(output_path)

def save_output_image(input, output_path, source_path):
    """
    input: (D, H, W) where D is the number of channels (3 for rgb, 1 for grayscale)
           can be a pytorch tensor or a numpy array
    """
    # convert input to numpy array float32
    if torch.is_tensor(input):
        im_np = input.type(torch.FloatTensor).cpu().numpy()
    else:
        im_np = input.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(source_path, 'r') as src:
        profile = src.profile
        profile["dtype"] = rasterio.float32
        profile["height"] = im_np.shape[1]
        profile["width"] = im_np.shape[2]
        profile["count"] = im_np.shape[0]
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(im_np)
    
                                          
