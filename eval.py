import torch
import yaml
import os
import json
import train_utils
from models import load_model
from rendering import render_rays
from collections import defaultdict
import metrics
import numpy as np
import sat_utils
import train_utils
import argparse
import glob
import shutil
import re

import warnings
warnings.filterwarnings("ignore")

import argparse
from opt import Test_parser, printArgs

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[], drop_len=-1):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    key_loaded = ''
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            #skip the layers with names different from model_name
            #print("extract_model_state_dict: {} is not in {}, hence skipped".format(k, model_name))
            continue
        if drop_len < 0:
            drop_len = len(model_name)
        k = k[drop_len+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            key_loaded = "{} {}".format(key_loaded, k)
            checkpoint_[k] = v
    return checkpoint_, key_loaded

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[], drop_len=-1):
    model_dict = model.state_dict()
    checkpoint_, key_loaded = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore, drop_len=drop_len)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
    print('Load {} successed with {}'.format(model_name, key_loaded))

@torch.no_grad()
def batched_inference(models, rays, ts, args, apply_brdf=False, cos_irra_on=False):
    """Do batched inference on rays using chunk."""
    chunk_size = args.chunk
    batch_size = rays.shape[0]

    results = defaultdict(list)
    for i in range(0, batch_size, chunk_size):
        rendered_ray_chunks, brdf_type = \
            render_rays(models, args, rays[i:i + chunk_size],
                        ts[i:i + chunk_size] if ts is not None else None, apply_brdf=apply_brdf, cos_irra_on=cos_irra_on)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        if results[k][0] is None:
            results[k] = None
        else:
            results[k] = torch.cat(v, 0)

    return results, brdf_type

def load_nerf(run_id, logs_dir, ckpts_dir, epoch_number):
    log_path = os.path.join(logs_dir, run_id)
    assert os.path.exists(log_path), f"ckpt_path {log_path} does not exist"
    print('ckpts loaded: ', log_path)
    with open('{}/opts.json'.format(log_path), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    print(ckpts_dir)
    checkpoint_path = ckpts_dir + "{}/epoch={}.ckpt".format(run_id, epoch_number)
    print("Using", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Could not find checkpoint {}".format(checkpoint_path))

    # load models
    models = {}
    nerf_coarse = load_model(args)
    load_ckpt(nerf_coarse, checkpoint_path, model_name='nerf_coarse')
    models["coarse"] = nerf_coarse.cuda().eval()
    if args.n_importance > 0:
        nerf_fine = load_model(args)
        load_ckpt(nerf_coarse, checkpoint_path, model_name='nerf_fine')
        models['fine'] = nerf_fine.cuda().eval()
    if args.model == "sat-nerf":
        embedding_t = torch.nn.Embedding(args.t_embbeding_vocab, args.t_embbeding_tau)
        load_ckpt(embedding_t, checkpoint_path, model_name='embedding_t')
        models["t"] = embedding_t.cuda().eval()

    return models

def quickly_interpolate_nans_from_singlechannel_img(image, method='nearest', Print=False):
    from scipy import interpolate
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    mask = np.isnan(image.reshape(h, w))
    missing_x = xx[mask]
    missing_y = yy[mask]
    if Print:
        print('mask, missing_x, missing_y: ', mask, missing_x, missing_y)
    interp = True
    if interp == True:
        known_x = xx[~mask]
        known_y = yy[~mask]
        known_v = image[~mask]
        if Print:
            print('~mask, known_x, known_y: ', ~mask, known_x, known_y)
            print("known_v: ", known_v)
        interp_values = interpolate.griddata(
            (known_x, known_y), known_v, (missing_x, missing_y), method=method
        )
        interp_image = image.copy()
        interp_image[missing_y, missing_x] = interp_values
        return interp_image
    else:
        interp_image = image.copy()
        interp_image[missing_y, missing_x] = 100
        return interp_image

def save_dsm_grid(in_dsm_path, out_dsm_path=None):
    import rasterio
    if out_dsm_path == None:
        out_dsm_path = in_dsm_path[:-4] + '_Grid.tif'

    print('in_dsm_path, out_dsm_path', in_dsm_path, out_dsm_path)

    with rasterio.open(in_dsm_path) as f:
        profile = f.profile
        array = f.read(1)

    array = quickly_interpolate_nans_from_singlechannel_img(array)
    
    with rasterio.open(out_dsm_path, 'w', **profile) as dst:
        dst.write(array, 1)

    #os.remove(in_dsm_path)

def save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number, sun_s=False, rvptclouds=True):

    rays = sample["rays"].squeeze()
    rgbs = sample["rgbs"].squeeze()
    src_id = sample["src_id"][0]
    src_path = os.path.join(dataset.img_dir, src_id + ".tif")

    typ = "fine" if "rgb_fine" in results else "coarse"
    if "h" in sample and "w" in sample:
        W, H = sample["w"][0], sample["h"][0]
    else:
        W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))  # assume squared images
    img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    depth = results[f"depth_{typ}"]

    # save depth prediction
    _, _, alts = dataset.get_latlonalt_from_nerf_prediction(rays.cpu(), depth.cpu())
    out_path = "{}/depth/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    train_utils.save_output_image(alts.reshape(1, H, W), out_path, src_path)
    # save dsm
    out_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    dsm = dataset.get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
    save_dsm_grid(out_path)
    if rvptclouds == True:
        os.remove(out_path)
    # save rgb image
    out_path = "{}/rgb/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    train_utils.PrintMMM('img to save', img)
    train_utils.save_output_image_tiff(img, out_path, src_path)

def find_best_embbeding_for_val_image(models, rays, conf, gt_rgbs, train_indices=None):

    best_ts = None
    best_psnr = 0.

    if train_indices is None:
        train_indices = torch.arange(conf.N_vocab)
    for t in train_indices:
        ts = t.long() * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        results, brdf_type = batched_inference(models, rays, ts, conf)
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), gt_rgbs.cpu())
        if psnr_ > best_psnr:
            best_ts = ts
            best_psnr = psnr_

    return best_ts

def find_best_embeddings_for_val_dataset(val_dataset, models, conf, train_indices):
    print("finding best embedding indices for validation dataset...")
    list_of_image_indices = [0]
    for i in np.arange(1, len(val_dataset)):
        sample = val_dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id = sample["src_id"]
        aoi_id = src_id[:7]
        if aoi_id in ["JAX_068", "JAX_004", "JAX_214"]:
            t = predefined_val_ts(src_id)
        else:
            ts = find_best_embbeding_for_val_image(models, rays, conf, rgbs, train_indices=train_indices)
            t = torch.unique(ts).cpu().numpy()
        print("{}: {}".format(src_id, t))
        list_of_image_indices.append(t)
    print("... done!")
    return list_of_image_indices

def predefined_val_ts(img_id):    
    return 0

def eval_aoi(logs_dir, output_dir, epoch_number, split, infile_postfix=None, checkpoints_dir=None, root_dir=None, img_dir=None, gt_dir=None, run_id=''):
    #from datasets import SatelliteDataset
    from datasets import SatelliteRGBDEPDataset
    with open('{}/opts.json'.format(os.path.join(logs_dir, run_id)), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    if infile_postfix != '.txt':
        args.infile_postfix = infile_postfix

    if gt_dir is not None:
        assert os.path.isdir(gt_dir)
        args.gt_dir = gt_dir
    if img_dir is not None:
        assert os.path.isdir(img_dir)
        args.img_dir = img_dir
    if root_dir is not None:
        assert os.path.isdir(root_dir)
        args.root_dir = root_dir
    if not os.path.isdir(args.cache_dir):
        args.cache_dir = None

    printArgs(args)

    # load pretrained nerf
    if checkpoints_dir is None:
        checkpoints_dir = args.ckpts_dir
    print(checkpoints_dir, args.ckpts_dir)
    models = load_nerf(run_id, logs_dir, checkpoints_dir, epoch_number-1)
    models['coarse'].print_parms()

    dataset = SatelliteRGBDEPDataset(args, split="val")

    if split == "train":
        with open(os.path.join(args.root_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        dataset.json_files = [os.path.join(args.root_dir, json_p) for json_p in json_files]
        dataset.all_ids = [i for i, p in enumerate(dataset.json_files)]
        samples_to_eval = np.arange(0, len(dataset))
    else:
        samples_to_eval = np.arange(0, len(dataset), 1)
        #samples_to_eval = np.arange(len(dataset)-1, len(dataset))
        #samples_to_eval = np.arange(0, 1)        
        print('samples_to_eval', samples_to_eval)

    pt2d_BRF = [-1, -1]
    #pt2d_BRF = [400, 682]   #site A
    neib = 1
    strOuts = {}
    
    nbs = 0
    if pt2d_BRF[0] > 0 and pt2d_BRF[1] > 0:
        all_rgbs = []
        for i in range(-neib, neib+1):
            for j in range(-neib, neib+1):
                radius = (i*i + j*j) ** 0.5
                print('-----------i, j, radius: {}, {}, {:.3f}'.format(i, j, radius))
                if radius > neib:
                    continue
                strOut = eval_pixel_variedvw(args, dataset, samples_to_eval, models, pt2d_BRF[0]+i, pt2d_BRF[1]+j)
                strOuts[f'{i}_{j}'] = strOut
                nbs += 1
        mean = np.zeros((nbs, 9))
        i=0
        for k, v in strOuts.items():
            print(k, v)
        if 0:
            numbers = re.findall(r'\d+', v)
            print(numbers)
            numbers = np.array(numbers)
            mean[i] = numbers[2:]
            i += 1
    else:
        eval_images_fixedvw(args, dataset, samples_to_eval, models, output_dir, run_id, split, epoch_number)

def get_view_dirs(view_elevation_deg, view_azimuth_deg):
    """
    Get view direction vectors
    Args:
        view_elevation_deg: float, view elevation in  degrees
        view_azimuth_deg: float, view azimuth in degrees
    Returns:
        view_d: (90*360, 3) 3-valued unit vector encoding the view direction, repeated n_rays times
    """
    view_el = np.radians(view_elevation_deg).flatten()
    view_az = np.radians(view_azimuth_deg).flatten()
    view_d = np.array([np.sin(view_az) * np.cos(view_el), np.cos(view_az) * np.cos(view_el), np.sin(view_el)])
    view_d = np.transpose(view_d, (1, 0))
    view_dirs = torch.from_numpy(view_d).type(torch.FloatTensor)
    return view_dirs

def get_s(weights, sample, N_rays, keyword, typ='coarse'):
    res = torch.sum(weights.unsqueeze(-1) * sample[f'{keyword}_{typ}'].reshape(N_rays, -1, 3), -2)
    return res

def eval_pixel_variedvw(args, dataset, samples_to_eval, models, x, y):
    strOut = ''
    apply_brdf = False
    if args.funcM > 0:
        apply_brdf = True
    for i in samples_to_eval:
        sample = dataset[i]
        rays = sample["rays"].cuda()
        rgbs = sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id  = sample["src_id"]
        if "h" in sample and "w" in sample:
            W, H = sample["w"], sample["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))
        if 1: #pt2d_BRF[0] > 0 and pt2d_BRF[1] > 0:
            index = y*W + x
            print('x, y, W, H, index', x, y, W, H, index)
            rays = rays[index, :]
            rgbs = rgbs[index, :]
            ts = None
            N_rays = 5
            rays_ = torch.tile(rays, (N_rays, 1))
            results, brdf_type = batched_inference(models, rays_, ts, args, apply_brdf=apply_brdf)
            typ = "fine" if "rgb_fine" in results else "coarse"
            weights = results[f'weights_{typ}']
            rpv_k_s = get_s(weights, results, N_rays, 'rpv_k')
            rpv_theta_s = get_s(weights, results, N_rays, 'rpv_theta')
            rpv_rhoc_s = get_s(weights, results, N_rays, 'rpv_rhoc')
            normal = get_s(weights, results, N_rays, 'normal_an')
            normal = train_utils.l2_normalize(normal)
            albedo_s = get_s(weights, results, N_rays, 'albedo')
            print('--rpv_k_s: {:.5f}\n{}'.format(torch.mean(rpv_k_s), rpv_k_s))
            print('--rpv_theta_s: {:.5f}\n{}'.format(torch.mean(rpv_theta_s), rpv_theta_s))
            print('--rpv_rhoc_s: {:.5f}\n{}'.format(torch.mean(rpv_rhoc_s), rpv_rhoc_s))
            print('--normal: {}\n{}'.format(torch.mean(normal, dim=0), normal))
            print('--albedo_s: {}\n{}'.format(torch.mean(albedo_s, dim=0), albedo_s))
            train_utils.PrintMMM('irradiance', results[f'irradiance_{typ}'])
            print('-->>Final')
            strOut += '{:.5f}, '.format(torch.mean(rpv_k_s)) + '{:.5f}, '.format(torch.mean(rpv_theta_s)) + '{:.5f} '.format(torch.mean(rpv_rhoc_s)) + '{} '.format(torch.mean(normal, dim=0)) + '{}'.format(torch.mean(albedo_s, dim=0))
            print(strOut)
            continue 
    return strOut

def PrintRay(rays):
    if 1:
        train_utils.PrintMMM('rays_o x', rays[..., 0])
        train_utils.PrintMMM('rays_o y', rays[..., 1])
        train_utils.PrintMMM('rays_o z', rays[..., 2])
        train_utils.PrintMMM('rays_d x', rays[..., 3])
        train_utils.PrintMMM('rays_d y', rays[..., 4])
        train_utils.PrintMMM('rays_d z', rays[..., 5])
        train_utils.PrintMMM('near', rays[..., 6])
        train_utils.PrintMMM('far', rays[..., 7])
        train_utils.PrintMMM('sun_d x', rays[..., 8])
        train_utils.PrintMMM('sun_d y', rays[..., 9])
        train_utils.PrintMMM('sun_d z', rays[..., 10])
        
def eval_images_fixedvw(args, dataset, samples_to_eval, models, output_dir, run_id, split, epoch_number):
    import pytorch_lightning as pl
    logger = pl.loggers.TensorBoardLogger(save_dir=output_dir, default_hp_metric=False)
    psnr, ssim, mae = [], [], []
    apply_brdf = True if args.brdf_on < 1 else False
    cos_irra_on = True if args.cos_irra_on < 1 else False

    print('apply_brdf', apply_brdf)

    for i in samples_to_eval:
        sample = dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        PrintRay(rays)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        mask = sample["mask"]
        src_id  = sample["src_id"]
        if "h" in sample and "w" in sample:
            W, H = sample["w"], sample["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))

        ts = None

        results, brdf_type = batched_inference(models, rays, ts, args, apply_brdf=apply_brdf, cos_irra_on=cos_irra_on)

        print('brdf_type', brdf_type)
        for k, v in results.items():
            print('results item', k)

        typ = "fine" if "rgb_fine" in results else "coarse"
        depth_expan = torch.tile(results[f'depth_{typ}'].unsqueeze(-1), (1, results[f'z_vals_{typ}'].shape[1]))
        deviation = torch.abs(results[f'z_vals_{typ}'] - depth_expan).cpu().numpy()
        idx = np.argmin(deviation, axis=1)  #find the indix which is closet to the predicted depth
        xxx = np.arange(0,idx.shape[0])
        idx = np.vstack((xxx, idx)).T

        rgb = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        rgb = rgb * mask.view(1, H, W).cpu()
        train_utils.PrintMMM('rgb', rgb)
        rgb_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        rgb_gt = rgb_gt * mask.view(1, H, W).cpu()
        train_utils.PrintMMM('rgb_gt', rgb_gt)
        if np.abs(args.visu_scale - 1.) > 1e-5:
            rgb = np.clip(rgb * args.visu_scale, 0, 1.)
            rgb_gt = np.clip(rgb_gt * args.visu_scale, 0, 1.)
            train_utils.PrintMMM('scaled_rgb', rgb)
            train_utils.PrintMMM('scaled_rgb_gt', rgb_gt)        
            if f"albedo_{typ}" in results:
                results[f"albedo_{typ}"] = torch.clamp(results[f"albedo_{typ}"] * args.visu_scale, min=0, max=1.)
        depth, _, _ = train_utils.visualize_depth(results[f'depth_{typ}'].view(H, W))
        sigmas = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'sigmas', idx=idx, norm_type='minmax')
        alphas = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'alphas', idx=idx, bUnsqz=True, norm_type='none') 
        transparency = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'transparency', idx=idx, bUnsqz=True, norm_type='none')
        weights = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'weights', idx=idx, bUnsqz=True, norm_type='none')
        nr_from_depth, _ = dataset.calc_normal_from_depth_v2(rays.cpu(), results[f"depth_{typ}"].cpu(), H, W)
        nr_from_depth = train_utils.visu_normal(nr_from_depth.view(H, W, 3))
        stack = torch.stack([rgb, rgb_gt, depth, nr_from_depth])
        stack1 = torch.stack([rgb, rgb_gt])
        if f"albedo_{typ}" in results: # and brdf_type != 'Lambertian':
            albedo = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'albedo', Accum=True, norm_type='none')
            stack = torch.cat((stack, albedo.unsqueeze(0)), dim=0)
        if f"normal_an_{typ}" in results:
            normal_an = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'normal_an', Accum=True, norm_type='-1et1') # (3, H, W)
            nr_vw = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_vw', Accum=False, norm_type='-1et1') # (1, H, W)
            nr_vw_enh = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_vw', Accum=False, norm_type='enhance') # (1, H, W)
            nr_sun = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_sun', Accum=False, norm_type='-1et1') # (1, H, W)
            nr_sun_enh = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_sun', Accum=False, norm_type='enhance') # (1, H, W)
            stack = torch.cat((stack, normal_an.unsqueeze(0)), dim=0)
        if f"rpv_k_{typ}" in results:
            rpv_k = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'rpv_k', Accum=True, norm_type='minmax') #, min_=0, max_=2)
            stack = torch.cat((stack, rpv_k.unsqueeze(0)), dim=0)
            if f"rpv_theta_{typ}" in results:
                rpv_theta = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'rpv_theta', Accum=True, norm_type='minmax') #, min_=-1, max_=1)
                stack = torch.cat((stack, rpv_theta.unsqueeze(0)), dim=0)
            if f"rpv_rhoc_{typ}" in results:
                rpv_rhoc = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'rpv_rhoc', Accum=True, norm_type='minmax')
                stack = torch.cat((stack, rpv_rhoc.unsqueeze(0)), dim=0)

        for k in sample.keys():
            if torch.is_tensor(sample[k]):
                sample[k] = sample[k].unsqueeze(0)
            else:
                sample[k] = [sample[k]]
        out_dir = os.path.join(output_dir, run_id, split)
        os.makedirs(out_dir, exist_ok=True)
        save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number, rvptclouds=False)

        # image metrics
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_, psnr_scl = metrics.psnr(results[f"rgb_{typ}"].cpu(), rgbs.cpu(), valid_mask=torch.tile(mask.view(H*W, 1), (1,3)), scl=True)
        psnr.append(psnr_)
        ssim_, ssim_scl = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W).cpu()*mask.view(1, 1, H, W), rgbs.view(1, 3, H, W).cpu()*mask.view(1, 1, H, W), scl=True)
        ssim.append(ssim_)

        # geometry metrics
        pred_dsm_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        mae_, mae_in, mae_out, diff, mae_nr, diff_nr = sat_utils.compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, args.aoi_id, args.gt_dir, out_dir, epoch_number, calc_mae_nr=True)
        os.remove(pred_dsm_path)
        mae.append(mae_)
        print("{}: pnsr {:.3f} / ssim {:.3f} / mae {:.3f}, mae_in {:.3f}, mae_out {:.3f}, mae_nr {:.3f}".format(src_id, psnr_, ssim_, mae_, mae_in, mae_out, mae_nr))
        print('pnsr_scl {:.3f} / ssim_scl {:.3f}'.format(psnr_scl, ssim_scl))

        # clean files
        in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_epoch*.tif"))[0]
        out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm"))
        os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
        shutil.copyfile(in_tmp_path, out_tmp_path)
        os.remove(in_tmp_path)
        in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_diff_epoch*.tif"))[0]
        out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm_diff"))
        os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
        shutil.copyfile(in_tmp_path, out_tmp_path)
        os.remove(in_tmp_path)

        if 1:
            logger.experiment.add_images(f'val_{i}', stack, 1)
            logger.experiment.add_images(f'val_{i}1', stack1, 1)
            print('visualization saved in logger val_{}, image size: {}'.format(i, stack.shape))

    if 1:
        for dirpath, dirnames, filenames in os.walk(output_dir + '/default/version_0/'):
             for filename in filenames:
                if 'events' in filename:
                    print(os.path.join(dirpath, filename))

    print("\nMean PSNR: {:.3f}".format(np.mean(np.array(psnr))))
    print("Mean SSIM: {:.3f}".format(np.mean(np.array(ssim))))
    print("Mean MAE: {:.3f}\n".format(np.mean(np.array(mae))))

    print('eval finished !')

if __name__ == '__main__':
    args = Test_parser()

    print("args.logs_dir, args.output_dir, args.epoch_number, args.split, args.run_id", args.logs_dir, args.output_dir, args.epoch_number, args.split, args.run_id)

    eval_aoi(args.logs_dir, args.output_dir, args.epoch_number, args.split, infile_postfix=args.infile_postfix, run_id=args.run_id)

