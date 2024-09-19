import argparse

import shutil

import torch
import pytorch_lightning as pl

from opt import Train_parser, printArgs
from datasets import load_dataset
from metrics import load_loss, DepthLoss, SNerfLoss, NormalLoss, NormalRegLoss, HardSurfaceLoss
from torch.utils.data import DataLoader
from collections import defaultdict
import torchvision.transforms as T

from rendering import render_rays
from models import load_model
import train_utils
import metrics
import os
import numpy as np
import datetime
from sat_utils import compute_mae_and_save_dsm_diff, Cloud2Grid

from eval import find_best_embbeding_for_val_image, save_nerf_output_to_images, predefined_val_ts, load_ckpt

import warnings
warnings.filterwarnings("ignore", category=Warning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

class NeRF_pl(pl.LightningModule):
    """NeRF network"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.normal_lr_loss_reg_applied = False
        self.normal_an_loss_reg_applied = False

        if np.abs(args.nr_spv_lambda) > 1e-5:
            self.normal_loss = NormalLoss(lambda_nr_spv=args.nr_spv_lambda)

        if args.normal == 'analystic_learned' or args.normal == 'learned':
            if args.nr_reg_lr_lambda > 0:
                print('Initialization: Use regularization loss for learned normal')
                self.normal_lr_loss_reg_applied = True
                self.normal_lr_loss_reg = NormalRegLoss(lambda_nr_reg=args.nr_reg_lr_lambda, keyword='normal_lr')

        if args.normal == 'analystic_learned' or args.normal == 'analystic':
            if args.nr_reg_an_lambda > 0:
                print('Initialization: Use regularization loss for analystics normal')
                self.normal_an_loss_reg_applied = True
                self.normal_an_loss_reg = NormalRegLoss(lambda_nr_reg=args.nr_reg_an_lambda, keyword='normal_an')

        print('args.normal, args.nr_reg_lr_lambda, args.nr_reg_an_lambda', args.normal, args.nr_reg_lr_lambda, args.nr_reg_an_lambda)

        self.loss = load_loss(args)
        self.depth = args.ds_lambda > 0
        self.brdf_on = np.round(args.brdf_on * args.max_train_steps)
        self.nrrg_on = np.round(args.nrrg_on * args.max_train_steps)
        self.gsam_only_on = np.round(args.gsam_only_on * args.max_train_steps)
        self.cos_irra_on = np.round(args.cos_irra_on * args.max_train_steps)
        self.ds_drop = 0
        if self.depth:
            print('Initialization: Use depth loss')
            self.depth_loss = DepthLoss(lambda_ds=args.ds_lambda, GNLL=args.GNLL, usealldepth=args.usealldepth, margin=args.margin, stdscale =args.stdscale, subset=False if args.model == 'sat-nerf' else True)
            self.ds_drop = np.round(args.ds_drop * args.max_train_steps)
        #HardSurfaceLoss
        self.hardsurface_loss_applied = args.hs_lambda > 0
        if self.hardsurface_loss_applied:
            print('Initialization: Use hardsurface loss')
            self.hardsurface_loss = HardSurfaceLoss(lambda_hs=args.hs_lambda)
        self.in_ckpts = args.in_ckpts
        self.define_models()
        self.outdir = "{}/".format(args.logs_dir)
        self.val_im_dir = "{}/val".format(args.logs_dir)
        self.train_im_dir = "{}/train".format(args.logs_dir)
        self.train_steps = 0

        self.use_ts = False
        if self.args.beta == True:
            print('Initialization: Use beta loss')
            if self.args.model == "sat-nerf" or self.args.model == "sps-nerf" or self.args.model == 'spsbrdf-nerf':
                self.loss_without_beta = SNerfLoss(lambda_sc=args.sc_lambda)
                self.use_ts = True

    def define_models(self):
        self.models = {}
        self.nerf_coarse = load_model(self.args)
        if self.args.eval == 1:
            assert os.path.exists(self.in_ckpts), f"{self.in_ckpts} not found"
            load_ckpt(self.nerf_coarse, self.in_ckpts, model_name='nerf_coarse')
            self.nerf_coarse.freeze('all')
            print('ckpts {} loaded'.format(self.in_ckpts))
            self.nerf_coarse.print_parms()
        elif self.in_ckpts != 'none':
            assert os.path.exists(self.in_ckpts), f"{self.in_ckpts} not found"
            load_ckpt(self.nerf_coarse, self.in_ckpts, model_name='nerf_coarse.fc_net', drop_len=11)
            load_ckpt(self.nerf_coarse, self.in_ckpts, model_name='nerf_coarse.sigma_from_xyz', drop_len=11)
            load_ckpt(self.nerf_coarse, self.in_ckpts, model_name='nerf_coarse.feats_from_xyz', drop_len=11)
            if self.args.b != True:     #if not Hapke
                load_ckpt(self.nerf_coarse, self.in_ckpts, model_name='nerf_coarse.rgb_from_xyzdir', drop_len=11)
            print('ckpts {} loaded'.format(self.in_ckpts))
        self.models['coarse'] = self.nerf_coarse
        if self.args.print_debuginfo == True:
            self.models["coarse"].print_parms()
        if self.args.n_importance > 0:
            self.nerf_fine = load_model(self.args)
            if self.in_ckpts != 'none':
                load_ckpt(self.nerf_fine, self.in_ckpts, model_name='nerf_fine')
            self.models['fine'] = self.nerf_fine
        if self.args.beta == True:
            if self.args.model == "sat-nerf" or self.args.model == "sps-nerf" or self.args.model == 'spsbrdf-nerf':
                self.embedding_t = torch.nn.Embedding(self.args.t_embbeding_vocab, self.args.t_embbeding_tau)
                if self.in_ckpts != 'none':
                    load_ckpt(self.embedding_t, self.in_ckpts, model_name='embedding_t')
                self.models["t"] = self.embedding_t

    def forward(self, rays, ts, mode='test', valid_depth=None, target_depths=None, target_std=None, apply_brdf=False, bTestNormal=False, bTestSun_v=False, gsam_only=False, rows=None, cols=None, percent=0, apply_theta=False, cos_irra_on=False):

        chunk_size = self.args.chunk
        batch_size = rays.shape[0]

        results = defaultdict(list)
        for i in range(0, batch_size, chunk_size):
            rows_chunk=None
            cols_chunk=None
            if rows != None:
                rows_chunk=rows[i:i + chunk_size]
            if cols != None:
                cols_chunk = cols[i:i + chunk_size]
            rendered_ray_chunks, brdf_type = \
                render_rays(self.models, self.args, rays[i:i + chunk_size],
                            ts[i:i + chunk_size] if ts is not None else None, mode=mode, valid_depth=valid_depth, target_depths=target_depths, target_std=target_std, apply_brdf=apply_brdf, print_debuginfo=self.args.print_debuginfo, bTestNormal=bTestNormal, bTestSun_v=bTestSun_v, gsam_only=gsam_only, rows=rows_chunk, cols=cols_chunk, percent=percent, apply_theta=apply_theta, cos_irra_on=cos_irra_on)
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results, brdf_type

    def prepare_data(self):
        self.train_dataset = [] + load_dataset(self.args, split="train")
        self.val_dataset = [] + load_dataset(self.args, split="val")

    def configure_optimizers(self):
        parameters = train_utils.get_parameters(self.models)
        lr_parameters = filter(lambda p: p.requires_grad, parameters)
        self.optimizer = torch.optim.Adam(lr_parameters, lr=self.args.lr, weight_decay=0)

        max_epochs = self.get_current_epoch(self.args.max_train_steps)
        print('datalen, batch_size, datalen/batch_size',len(self.train_dataset[0]), self.args.batch_size, len(self.train_dataset[0]) // self.args.batch_size)
        print("***************max_epochs: ", max_epochs)
        print('gsam_only_on: step {}/{}, ep {}/{}'.format(self.gsam_only_on, self.args.max_train_steps, self.get_current_epoch(self.gsam_only_on), max_epochs))
        print('brdf_on:   step {}/{}, ep {}/{}'.format(self.brdf_on, self.args.max_train_steps, self.get_current_epoch(self.brdf_on), max_epochs))
        print('nrrg_on:   step {}/{}, ep {}/{}'.format(self.nrrg_on, self.args.max_train_steps, self.get_current_epoch(self.nrrg_on), max_epochs))
        print('cos_irra_on:   step {}/{}, ep {}/{}'.format(self.cos_irra_on, self.args.max_train_steps, self.get_current_epoch(self.cos_irra_on), max_epochs))
        print('ds_drop:   step {}/{}, ep {}/{}'.format(self.ds_drop, self.args.max_train_steps, self.get_current_epoch(self.ds_drop), max_epochs))

        scheduler = train_utils.get_scheduler(optimizer=self.optimizer, lr_scheduler='step', num_epochs=max_epochs)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def train_dataloader(self):
        a = DataLoader(self.train_dataset[0],
                       shuffle=True, 
                       num_workers=4,
                       batch_size=self.args.batch_size,
                       pin_memory=True)
        loaders = {"color": a}
        if self.args.model == "sat-nerf" and self.depth:
            b = DataLoader(self.train_dataset[1],
                           shuffle=True,
                           num_workers=4,
                           batch_size=self.args.batch_size,
                           pin_memory=True)
            loaders["depth"] = b
        return loaders

    def val_dataloader(self):
        a = DataLoader(self.val_dataset[0],
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)
        return a

    def training_step(self, batch, batch_nb):
        self.log("lr", train_utils.get_learning_rate(self.optimizer))
        self.train_steps += self.args.gpu_id 

        if self.args.print_debuginfo == True:
            print('-----------------batch_nb in training: ', batch_nb)
            self.models["coarse"].print_parms()

        gsam_only = False
        if self.train_steps > self.gsam_only_on:
            gsam_only=True
        apply_brdf = False
        apply_theta = False
        if self.train_steps > self.brdf_on: #trun BRDF on in the middle of training where the network is well initialized
            apply_brdf = True
        if self.train_steps > self.brdf_on * 2:
            apply_theta = True
        bTestNormal = False

        rays = batch["color"]["rays"] # (B, 11)
        rgbs = batch["color"]["rgbs"] # (B, 3)
        ts = None if not self.use_ts else batch["color"]["ts"].squeeze() # (B, 1)
        epoch = self.get_current_epoch(self.train_steps)

        valid_depth = None
        depths = None
        target_std = None
        if self.args.model == 'sps-nerf' or self.args.model == 'spsbrdf-nerf':
            if 'depths' in batch["color"]:
                valid_depth = batch["color"]["valid_depth"] # (B)
                depths = batch["color"]["depths"] # (B,2)
                target_std = batch["color"]["depth_std"]  #(B)

                if self.args.ds_noweights:
                    depths[:, 1] = torch.ones_like(depths[:, 1])

        results, brdf_type = self(rays, ts, mode='train', valid_depth=valid_depth, target_depths=depths, target_std=target_std, apply_brdf=apply_brdf, gsam_only=gsam_only, bTestNormal=bTestNormal, apply_theta=apply_theta, cos_irra_on=(self.train_steps > self.cos_irra_on))

        if batch_nb == 0:
            print('brdf_type: ', brdf_type)
        typ = "fine" if "rgb_fine" in results else "coarse"

        #loss: RGB
        if 'beta_coarse' in results and epoch < 2:
            loss, loss_dict = self.loss_without_beta(results, rgbs)
        else:
            if 0: #self.args.model == 'spsbrdf-nerf':
                exp = 1000
                loss, loss_dict = self.loss(results, rgbs, exp)
            else:
                loss, loss_dict = self.loss(results, rgbs)

        self.args.noise_std *= 0.9

        #loss: depth
        if self.depth:
            if self.args.model == 'sps-nerf' or self.args.model == 'spsbrdf-nerf':
                kp_depths = depths[:, 0]
                kp_weights = depths[:, 1]
                if self.args.ds_noweights:
                    kp_weights = torch.ones_like(kp_weights)
                loss_depth, tmp = self.depth_loss(results, kp_depths, kp_weights, target_valid_depth=valid_depth, target_std=target_std)
            elif self.args.model == 'sat-nerf':
                tmp, _ = self(batch["depth"]["rays"], batch["depth"]["ts"].squeeze())
                kp_depths = torch.flatten(batch["depth"]["depths"][:, 0])
                kp_weights = torch.flatten(batch["depth"]["depths"][:, 1])
                if self.args.ds_noweights:
                    kp_weights = torch.ones_like(kp_weights)
                loss_depth, tmp = self.depth_loss(tmp, kp_depths, kp_weights, target_valid_depth=valid_depth, target_std=target_std)

            if self.train_steps < self.ds_drop :
                loss += loss_depth
            for k in tmp.keys():
                loss_dict[k] = tmp[k]

        perc_ng_nr = -1
        if self.args.model == 'spsbrdf-nerf':
            if self.normal_an_loss_reg_applied:     #and apply_brdf == True:
                if f"normal_an_{typ}" in results and self.train_steps > self.nrrg_on:
                    loss_nr_an_reg, tmp, perc_ng_nr = self.normal_an_loss_reg(results)
                    self.log("train/bad_nr_an%", perc_ng_nr)
                    loss += loss_nr_an_reg
                    for k in tmp.keys():
                        loss_dict[k] = tmp[k]

            if self.normal_lr_loss_reg_applied:     #and apply_brdf == True:
                if f"normal_lr_{typ}" in results and self.train_steps > self.nrrg_on:
                    loss_nr_lr_reg, tmp, perc_ng_nr = self.normal_lr_loss_reg(results)
                    self.log("train/bad_nr_lr%", perc_ng_nr)
                    loss += loss_nr_lr_reg
                    for k in tmp.keys():
                        loss_dict[k] = tmp[k]

        if batch_nb == 0:
            print("step: {}, train/bad_nr: {:.3f}%".format(self.train_steps, perc_ng_nr))

        #loss: hardsurface
        if self.hardsurface_loss_applied:     #and apply_brdf == True:
            if self.args.model == 'spsbrdf-nerf' and epoch > 2:
                if f"depth_{typ}" in results and f"z_vals_{typ}" in results:
                    loss_hs_reg, tmp = self.hardsurface_loss(results)
                    loss += loss_hs_reg
                    for k in tmp.keys():
                        loss_dict[k] = tmp[k]
                else:
                    print("Error: no depth or z_vals in results for hardsurface_loss")

        #loss: normal_spv
        if self.args.model == 'spsbrdf-nerf' and np.abs(self.args.nr_spv_lambda) > 1e-5:
            loss_applied = False
            if self.args.nr_spv_type == 1:
                if f"normal_an_{typ}" in results and f"normal_lr_{typ}" in results: #and self.train_steps > self.cos_irra_on:
                    loss_nr, tmp = self.normal_loss(results[f"weights_{typ}"], results[f'normal_an_{typ}'], results[f'normal_lr_{typ}'], keyword="an_lr")
                    loss_applied = True
                else:
                    print("Error: no normal_an or normal_lr in results for normal_loss")
            elif self.args.nr_spv_type == 2:
                if f"normal_lr_{typ}" in results: #and self.train_steps > self.cos_irra_on:
                    loss_nr, tmp = self.normal_loss(results[f"weights_{typ}"], batch["color"]["normals"], results[f'normal_lr_{typ}'], keyword="lr")
                    loss_applied = True
                else:
                    print("Error: no normal_lr in results for normal_loss")
            elif self.args.nr_spv_type == 3:
                if f"normal_an_{typ}" in results: #and self.train_steps > self.cos_irra_on:   #epoch > 1:
                    #loss_nr, tmp = self.normal_loss(results[f"weights_{typ}"], batch["color"]["normals"], results[f'normal_an_{typ}'], target_weight=depths[:, 1], target_valid_depth=valid_depth, keyword="an")
                    loss_nr, tmp = self.normal_loss(results[f"weights_{typ}"], batch["color"]["normals"], results[f'normal_an_{typ}'], target_weight=batch["color"]["valid_normal"], target_valid_depth=valid_depth, keyword="an")
                    loss_applied = True
                else:
                    print("Error: no normal_an in results for normal_loss")

            if loss_applied == True:
                loss += loss_nr
                for k in tmp.keys():
                    loss_dict[k] = tmp[k]

        if f"normal_an_{typ}" in results:
            perc_vec0 = train_utils.check_vec0(f'normal_an_{typ}', results[f'normal_an_{typ}'])
            self.log("train/nr_an0%", perc_vec0)

        self.log("train_loss/toal", loss)

        with torch.no_grad():
            psnr_, _ = metrics.psnr(results[f"rgb_{typ}"], rgbs)

            self.log("train/psnr", psnr_)
            if f'irradiance_{typ}' in results:
                self.log("train/irradiance", torch.mean(results[f'irradiance_{typ}']))

            if 1:
                z_vals = results[f'z_vals_{typ}']
                pred_depth = results[f'depth_{typ}']
                pred_weight = results[f'weights_{typ}']
                sampling_std = train_utils.calc_depth_std(z_vals, pred_depth, pred_weight)
                depth_std = np.mean(sampling_std.cpu().numpy())
                self.log("train/depth_std", depth_std)

        for k in loss_dict.keys():
            self.log("train_loss/{}".format(k), loss_dict[k])

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        epoch = self.get_current_epoch(self.train_steps)
        max_epochs = self.get_current_epoch(self.args.max_train_steps)
        apply_eval = True if (epoch % self.args.eval_every_n_epochs == 0) else False

        if epoch < 2 and batch_nb == 0:
            for dirpath, dirnames, filenames in os.walk(self.outdir + '/default/version_0/'):
                for filename in filenames:
                    if 'events' in filename:
                        print(epoch, ' ', os.path.join(dirpath, filename))

        time = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        if apply_eval == False and epoch < max_epochs-1:
            if epoch<=self.args.save_first_n_visu and batch_nb==0:
                pass
            else:
                print("--VALIDATION epoch: {}, step: {}, batch {}, image: {}, time: {}".format(epoch, self.train_steps, batch_nb, batch["src_id"], time))
                return {"loss": 0}

        strOut = "--VALIDATION epoch: {}, step: {}, batch {}, image: {}, time: {}\n".format(epoch, self.train_steps, batch_nb, batch["src_id"], time)
        strOut += " lr: {:.7f}".format(train_utils.get_learning_rate(self.optimizer))

        save_visu_every_n_epochs = self.args.save_visu_every_n_epochs
        save_file_every_n_epochs = self.args.save_file_every_n_epochs
        self.is_validation_image = True
        if self.args.data == 'sat':
            self.is_validation_image = False if batch_nb == 0 else True
            if "is_val" in batch:
                self.is_validation_image = batch["is_val"] #True
        apply_brdf = False
        apply_theta = False
        bTestSun_v = False      #bTestSun_v=True will slow down validation
        bTestNormal = False
        add_images_in_logger = False
        if (epoch<=self.args.save_first_n_visu) or ((epoch % save_visu_every_n_epochs == 0)) or self.args.eval > 0:
            add_images_in_logger = True
            if self.args.TestSun_v == True:
                bTestSun_v = True
            if self.args.TestNormal == True:
                bTestNormal = True
        if self.train_steps > self.brdf_on: # and self.is_validation_image == False:   #to speed up
            apply_brdf = True
        if self.train_steps > self.brdf_on * 2:
            apply_theta = True
        gsam_only = False
        if self.train_steps > self.gsam_only_on:
            gsam_only=True

        rays = batch["rays"] # (1, B, 11)
        rgbs = batch["rgbs"] # (1, B, 3)
        mask = batch["mask"] # (1, B)

        if batch_nb == 0: #batch["idx"] == 0:
            self.models['coarse'].print_parms()

        rows = None
        cols = None

        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        if self.args.model == "sat-nerf" or self.args.model == "sps-nerf" or self.args.model == 'spsbrdf-nerf':
            t = predefined_val_ts(batch["src_id"][0])
            ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        else:
            ts = None
        results, brdf_type = self(rays, ts, mode='test', apply_brdf=apply_brdf, bTestNormal=bTestNormal, bTestSun_v=bTestSun_v, gsam_only=gsam_only, rows=rows, cols=cols, percent=self.train_steps/self.args.max_train_steps, apply_theta=apply_theta, cos_irra_on=(self.train_steps > self.cos_irra_on))
        typ = "fine" if "rgb_fine" in results else "coarse"
        if batch_nb == 0:
            strOut += ' brdf_type: {}; '.format(brdf_type)
            if f'irradiance_{typ}' in results:
                strOut += ' irradiance mean: {:.2f}\n'.format(torch.mean(results[f'irradiance_{typ}']))
        loss, loss_dict = self.loss(results, rgbs)

        if "h" in batch and "w" in batch:
            W, H = batch["w"], batch["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float())) # assume squared images
       
        perc_ng_nr = -1
        if self.args.model == 'spsbrdf-nerf':
            if self.normal_an_loss_reg_applied:     #and apply_brdf == True:
                if f"normal_an_{typ}" in results and self.train_steps > self.nrrg_on:
                    loss_nr_an_reg, tmp, perc_ng_nr = self.normal_an_loss_reg(results)
                    self.log("val/bad_nr_an%", perc_ng_nr)
                    loss += loss_nr_an_reg
                    for k in tmp.keys():
                        loss_dict[k] = tmp[k]

            if self.normal_lr_loss_reg_applied:     #and apply_brdf == True:
                if f"normal_lr_{typ}" in results and self.train_steps > self.nrrg_on:
                    loss_nr_lr_reg, tmp, perc_ng_nr = self.normal_lr_loss_reg(results)
                    self.log("val/bad_nr_lr%", perc_ng_nr)
                    loss += loss_nr_lr_reg
                    for k in tmp.keys():
                        loss_dict[k] = tmp[k]

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
        rgb_diff = torch.abs(rgb - rgb_gt)
        if np.abs(self.args.visu_scale - 1.) > 1e-5:
            rgb = np.clip(rgb * self.args.visu_scale, 0, 1.)
            rgb_gt = np.clip(rgb_gt * self.args.visu_scale, 0, 1.)
            train_utils.PrintMMM('scaled_rgb', rgb)
            train_utils.PrintMMM('scaled_rgb_gt', rgb_gt)
            if f"albedo_{typ}" in results:
                results[f"albedo_{typ}"] = torch.clamp(results[f"albedo_{typ}"] * self.args.visu_scale, min=0, max=1.)
        depth, _, _ = train_utils.visualize_depth(results[f'depth_{typ}'].view(H, W)) #, min_dep=min_dep, max_dep=max_dep)  # (3, H, W)
        sampling_std, std_array, depth_std = train_utils.generate_std_img(results, typ, H, W, norm_type='minmax') #.permute(2, 0, 1).cpu()
        sigmas = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'sigmas', idx=idx, norm_type='minmax')
        alphas = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'alphas', idx=idx, bUnsqz=True, norm_type='none') 
        transparency = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'transparency', idx=idx, bUnsqz=True, norm_type='none')
        weights = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'weights', idx=idx, bUnsqz=True, norm_type='none')
        nr_from_depth, _ = self.val_dataset[0].calc_normal_from_depth_v2(rays.cpu(), results[f"depth_{typ}"].cpu(), H, W)
        nr_from_depth = train_utils.visu_normal(nr_from_depth.view(H, W, 3))
        stack = torch.stack([rgb, rgb_gt, depth, nr_from_depth])
        strStack = 'rgb, rgb_gt, depth, nr_from_depth'
        if 'depths' in batch and self.is_validation_image == False:
            depth_gt, min_dep, max_dep = train_utils.visualize_depth(batch["depths"][0,:,0].view(H, W), keyword='depth_gt') #, min_dep=min_dep, max_dep=max_dep)
            stack = torch.cat((stack, depth_gt.unsqueeze(0)), dim=0)
            strStack += ', depth_gt'
        if 'normals' in batch and self.is_validation_image == False:
            nr_gt = train_utils.visu_normal(batch["normals"][0,:,:].view(H, W, 3))
            stack = torch.cat((stack, nr_gt.unsqueeze(0)), dim=0)
        if f"albedo_{typ}" in results and brdf_type != 'Lambertian':
            albedo = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'albedo', Accum=True, norm_type='none')
            stack = torch.cat((stack, albedo.unsqueeze(0)), dim=0)
            strStack += ', albedo'
        if 'normal_norot' in batch:
            nr_gt = train_utils.visu_normal(batch["normal_norot"][0,:,:].view(H, W, 3))
            stack = torch.cat((stack, nr_gt.unsqueeze(0)), dim=0)
        if f"sun_{typ}" in results:
            sun_s = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'sun', Accum=False, norm_type='none', tile=True)
            stack = torch.cat((stack, sun_s.unsqueeze(0)), dim=0)
        if f"normal_lr_{typ}" in results:
            normal_lr = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'normal_lr', Accum=True, norm_type='-1et1') # (3, H, W)
            stack = torch.cat((stack, normal_lr.unsqueeze(0)), dim=0)
        if f"normal_an_{typ}" in results:
            perc_vec0 = train_utils.check_vec0(f'normal_an_{typ}', results[f'normal_an_{typ}'])
            strOut += ' bad_nr_an%: {:.3f} nr_an0%: {:.3f}'.format(perc_ng_nr, perc_vec0)
            self.log("val/nr_an0%", perc_vec0)
            normal_an = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'normal_an', Accum=True, norm_type='-1et1') # (3, H, W)
            nr_vw = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_vw', Accum=False, norm_type='-1et1') # (1, H, W)
            nr_vw_enh = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_vw', Accum=False, norm_type='enhance') # (1, H, W)
            nr_sun = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_sun', Accum=False, norm_type='-1et1') # (1, H, W)
            nr_sun_enh = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'nr_sun', Accum=False, norm_type='enhance') # (1, H, W)
            stack = torch.cat((stack, normal_an.unsqueeze(0)), dim=0)
        if f"roughness_{typ}" in results:
            roughness = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'roughness', Accum=True, norm_type='none')
            glossy_norm = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'glossy', Accum=True, norm_type='minmax')
            glossy = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'glossy', Accum=True, norm_type='none')
            brdf = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'brdf', Accum=True, norm_type='none')
            stack = torch.cat((stack, roughness.unsqueeze(0)), dim=0)   #, glossy_norm.unsqueeze(0), glossy.unsqueeze(0), brdf.unsqueeze(0)), dim=0)
            if self.args.MultiBRDF == True:
                roughness_s = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'roughness', idx=idx, norm_type='none')
                glossy_s = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'glossy', idx=idx, norm_type='minmax')
                brdf_s = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'brdf', idx=idx, norm_type='none')

            f = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'f', Accum=True, norm_type='none')
            g = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'g', Accum=True, norm_type='none')
            d = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'd', Accum=True, norm_type='minmax')
            l_dot_n = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'l_dot_n', Accum=True, norm_type='none')
            v_dot_n = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'v_dot_n', Accum=True, norm_type='none')
            h = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'halfvec', Accum=True, norm_type='-1et1')
            n_h = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'n_h', Accum=True, norm_type='none')
        elif f"rpv_k_{typ}" in results:
            rpv_k = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'rpv_k', Accum=True, norm_type='minmax')
            stack = torch.cat((stack, rpv_k.unsqueeze(0)), dim=0)
            if f"rpv_theta_{typ}" in results:
                rpv_theta = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'rpv_theta', Accum=True, norm_type='minmax')
                stack = torch.cat((stack, rpv_theta.unsqueeze(0)), dim=0)
            if f"rpv_rhoc_{typ}" in results:
                rpv_rhoc = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'rpv_rhoc', Accum=True, norm_type='minmax')
                stack = torch.cat((stack, rpv_rhoc.unsqueeze(0)), dim=0)
        else:
            if f"hpk_b_{typ}" in results:
                hpk_b = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'hpk_b', Accum=True, norm_type='enhance')
                stack = torch.cat((stack, hpk_b.unsqueeze(0)), dim=0)
            if f"hpk_c_{typ}" in results:
                hpk_c = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'hpk_c', Accum=True, norm_type='none')
                stack = torch.cat((stack, hpk_c.unsqueeze(0)), dim=0)
            if f"hpk_theta_{typ}" in results:
                hpk_theta = train_utils.visualize_accumulated_feature(results, typ, H, W, 1, 'hpk_theta', Accum=True, norm_type='none')
                stack = torch.cat((stack, hpk_theta.unsqueeze(0)), dim=0)
            if f"hpk_P_{typ}" in results:
                brdf = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'brdf', Accum=True, norm_type='none')
                hpk_P = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'hpk_P', Accum=True, norm_type='enhance')
                hpk_Hi = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'hpk_Hi', Accum=True, norm_type='enhance')
                hpk_Hv = train_utils.visualize_accumulated_feature(results, typ, H, W, 3, 'hpk_Hv', Accum=True, norm_type='enhance')
                if 0: #f"hpk_b_{typ}" in results:       #otherwise hpk_P, hpk_Hi, hpk_Hv would be one, no need to save image
                    stack_1 = torch.stack([hpk_P, hpk_Hi, hpk_Hv])
                    stack = torch.cat((stack, stack_1), dim=0)

        if self.args.toyBRDF == True and self.is_validation_image == False:
            stack = torch.stack([normal_an, roughness, brdf, albedo, glossy, f, g, d, l_dot_n, v_dot_n])

        split = 'val' if self.is_validation_image else 'train'

        if save_file_every_n_epochs < 0:
            save = True if (epoch >= max_epochs-1) else False
        else:
            save = not bool(epoch % save_file_every_n_epochs)

        if self.args.data == 'sat' and batch["save_cross"] == True and (epoch<=self.args.save_first_n_visu or save==True): #(epoch % (self.args.save_file_every_n_epochs) == 0):
            h_mid = int(H/2)
            start = W*h_mid
            end = W*(h_mid+1)
            sort_num = 0
            feature_num = 0
            if f"sort_idx_{typ}" in results:
                sort_idx = results[f'sort_idx_{typ}'][start:end,:]
                sort_num = 1
            z_vals = results[f'z_vals_{typ}'][start:end,:]
            sigma = results[f'sigmas_{typ}'][start:end,:]
            alphas = results[f'alphas_{typ}'][start:end,:]
            transparency = results[f'transparency_{typ}'][start:end,:]
            feature_num += 4
            depth_gt = batch["depths"][0,start:end,0]
            depth = results[f'depth_{typ}'][start:end]
            std = std_array[h_mid:h_mid+1,:,0]
            head = torch.tensor([W, self.args.n_samples, self.args.guided_samples, sort_num, feature_num], device=z_vals.device).flatten()
            if f"sort_idx_{typ}" in results:
                cross_sec = torch.cat((head, sort_idx.flatten()), dim=0)
            else:
                cross_sec = head
            cross_sec = torch.cat((cross_sec, z_vals.flatten(), sigma.flatten(), alphas.flatten(), transparency.flatten(), depth_gt.flatten(), depth.flatten(), std.flatten()), dim=0)
            print('self.args.n_samples, self.args.guided_samples: ', self.args.n_samples, self.args.guided_samples)
            print('W, sort_num, feature_num, cross_sec.shape: ', W, sort_num, feature_num, cross_sec.shape)
            dir = self.outdir
            outFile1 = dir + "/{}_E{}_cross_sec.txt".format(batch["src_id"][0], epoch)
            print('cross_sec are saved in ', outFile1)
            np.savetxt(outFile1, cross_sec.cpu(), fmt="%lf",delimiter=' ')

        if self.args.data == 'sat' and (save or self.args.eval > 0):
            # save some images to disk for a more detailed visualization
            print("save files for epoch {}, step {}, batch_nb {}".format(epoch, self.train_steps, batch_nb), batch["src_id"])
            out_dir = self.val_im_dir if self.is_validation_image else self.train_im_dir
            save_nerf_output_to_images(self.val_dataset[0], batch, results, out_dir, epoch, sun_s=True)

        scl = True if epoch == 2 else False
        psnr_all, psnr_all_scl = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        psnr_, psnr_scl = metrics.psnr(results[f"rgb_{typ}"], rgbs, valid_mask=torch.tile(mask.view(H*W, 1), (1,3)), scl=scl) #, Print=True)
        ssim_all, ssim_all_scl = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.view(1, 3, H, W))
        ssim_, ssim_scl = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W)*mask.view(1, 1, H, W), rgbs.view(1, 3, H, W)*mask.view(1, 1, H, W), scl=scl)
        lpips_all = metrics.lpips(results[f"rgb_{typ}"].view(3, H, W).cpu(), rgbs.view(3, H, W).cpu()).squeeze()
        lpips_ = metrics.lpips((results[f"rgb_{typ}"].view(3, H, W)*mask.view(1, H, W)).cpu(), (rgbs.view(3, H, W)*mask.view(1, H, W)).cpu()).squeeze()

        if True: #self.args.data == 'sat':
            # 1st image is from the training set, so it must not contribute to the validation metrics
            if 1: #self.is_validation_image == True: #batch["depths"] == None:
                if 1:
                    aoi_id = self.args.aoi_id
                    gt_roi_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.txt")
                    gt_dsm_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.tif")
                    assert os.path.exists(gt_roi_path), f"{gt_roi_path} not found"
                    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
                    depth = results[f"depth_{typ}"]
                    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    out_path = os.path.join(self.val_im_dir, "dsm/tmp_pred_dsm_{}.tif".format(unique_identifier))
                    _ = self.val_dataset[0].get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
                    fair_mae = False
                    if self.is_validation_image == False:
                        in_path = out_path
                        out_path = out_path[:-4] + '_Grid.tif'
                        Cloud2Grid(in_path, out_path, Print=False)
                        #print(batch["src_id"], 'DSM is transformed from Cloud to Grid')
                        os.remove(in_path)
                        fair_mae = True
                    mae_, mae_in, mae_out, diff, mae_nr, diff_nr = compute_mae_and_save_dsm_diff(out_path, batch["src_id"][0], aoi_id, self.args.gt_dir, self.val_im_dir, 0, save=False, calc_mae_nr=True)
                    min_dep, max_dep = -20, 20
                    diff_visu, _, _ = train_utils.visualize_diff(diff, H=H, W=W, keyword='diff_dsm', min_dep=min_dep, max_dep=max_dep)
                    min_dep, max_dep = 0, 90
                    diff_nr_visu, _, _ = train_utils.visualize_diff(diff_nr, H=H, W=W, keyword='diff_nr', min_dep=min_dep, max_dep=max_dep)
                    if fair_mae == True: #self.is_validation_image == False:
                        stack = torch.cat((stack, diff_visu.unsqueeze(0), diff_nr_visu.unsqueeze(0)), dim=0)
                    os.remove(out_path)
                
                if self.is_validation_image == True:
                    self.log("val_loss/total", loss)
                    self.log("val_sub/psnr_{}".format(batch_nb), psnr_)
                    self.log("val_sub/ssim_{}".format(batch_nb), ssim_)
                    self.log("val_sub/lpips_{}".format(batch_nb), lpips_)
                    self.log("val/psnr", psnr_)
                    self.log("val/ssim", ssim_)
                    self.log("val/lpips", lpips_)
                    if fair_mae == True:
                        self.log("val_sub/mae_{}".format(batch_nb), mae_)
                        self.log("val_sub/mae_nr_{}".format(batch_nb), mae_nr)
                        self.log("val_sub/depth_std_{}".format(batch_nb), depth_std)
                        self.log("val/mae", mae_)
                        self.log("val/mae_nr", mae_nr)
                        self.log("val/depth_std", depth_std)
                        if mae_in > 0 and mae_out > 0:
                            self.log("val_sub/mae_in_{}".format(batch_nb), mae_in)      
                            self.log("val_sub/mae_out_{}".format(batch_nb), mae_out)    
                            self.log("val/mae_in", mae_in)
                            self.log("val/mae_out", mae_out)
                else:
                    self.log("train_/psnr", psnr_)
                    self.log("train_/ssim", ssim_)
                    self.log("train_/lpips", lpips_)
                    if fair_mae == True:
                        self.log("train_/mae", mae_)
                        self.log("train_/mae_nr", mae_nr)
                        self.log("train_/depth_std", depth_std)
                        if mae_in > 0 and mae_out > 0:
                            self.log("train_/mae_in", mae_in)
                            self.log("train_/mae_out", mae_out)

                strOut += " psnr_all: {:.3f}, ssim_all: {:.3f}, lpips_all: {:.3f}".format(psnr_all, ssim_all, lpips_all.numpy())
                if scl == True: #torch.abs(psnr_-psnr_scl) > 1e-5:
                    strOut += "\n psnr_scl: {:.3f}, ssim_scl: {:.3f} |".format(psnr_scl, ssim_scl)
                    strOut += " psnr_dif: {:.5f}".format(psnr_-psnr_scl)
                    strOut += " ssim_dif: {:.6f}".format(ssim_-ssim_scl)
                strOut += "\nloss: {:.3f}, psnr: {:.3f}, ssim: {:.3f}, lpips: {:.3f}\n".format(loss, psnr_, ssim_, lpips_.numpy())
                for k in loss_dict.keys():
                    self.log("val_loss/{}".format(k), loss_dict[k])
                    strOut += " {}: {:.6f}".format(k, loss_dict[k])
                if fair_mae == True:
                    strOut += "\nmae_nr: {:.3f}, mae: {:.3f}, mae_in: {:.3f}, mae_out: {:.3f}, depth_std: {:.3f}".format(mae_nr, mae_, mae_in, mae_out, depth_std)

        if add_images_in_logger == True:
            idx = batch_nb if "idx" not in batch else batch["idx"].item()
            tagg = '{}_{}'.format(split, idx)
            stack = stack.cpu()
            self.logger.experiment.add_images(tagg, stack, self.global_step)
            strOut += '\nvisualization saved in logger {}, image size: {}'.format(tagg, stack.shape)

        print(strOut)

        return {"loss": loss}
        
    def get_current_epoch(self, tstep):
        return train_utils.get_epoch_number_from_train_step(tstep, len(self.train_dataset[0]), self.args.batch_size)

def main():
    time = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    print("Launched time: ", time)

    torch.cuda.empty_cache()
    args = Train_parser()
    printArgs(args)
    system = NeRF_pl(args)

    if args.data == 'sat':
        shutil.copyfile(args.root_dir+"/train{}".format(args.infile_postfix), system.outdir+"/train{}".format(args.infile_postfix))
        shutil.copyfile(args.root_dir+"/test{}".format(args.infile_postfix), system.outdir+"/test{}".format(args.infile_postfix))

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir, default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath="{}".format(args.ckpts_dir),
                                                 filename="{epoch:d}",
                                                 save_top_k=-1,
                                                 every_n_val_epochs=args.save_ckpt_every_n_epochs)

    num_sanity_val_steps = 0 #1 #-1   #2
    if args.eval > 0:
        num_sanity_val_steps = -1
    print('num_sanity_val_steps: ', num_sanity_val_steps)
    max_steps = args.max_train_steps if args.gpu_id == 0 else int(args.max_train_steps/args.gpu_id)

    trainer = pl.Trainer(max_steps=max_steps,
                         logger=logger,
                         callbacks=[ckpt_callback],
                         resume_from_checkpoint=args.ckpt_path,
                         gpus=args.gpu_id,
                         auto_select_gpus=False,
                         deterministic=True,
                         benchmark=True,
                         weights_summary=None,
                         num_sanity_val_steps=num_sanity_val_steps, 
                         check_val_every_n_epoch=1,
                         profiler="simple")

    trainer.fit(system)

    time = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    print("Ended time: ", time)

if __name__ == "__main__":
    main()
