"""
This script defines the dataloader for a dataset of multi-view satellite images
"""

import numpy as np
import os
import sys

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

import rasterio
import rpcm
import glob
import sat_utils
import train_utils
from .cal_rmse_depth import cal_rmse_depth
from pytorch3d.transforms import axis_angle_to_matrix


def get_rays(cols, rows, rpc, min_alt, max_alt, bPrint=False, cs='ecef'):
    """
            Draw a set of rays from a satellite image
            Each ray is defined by an origin 3d point + a direction vector
            First the bounds of each ray are found by localizing each pixel at min and max altitude
            Then the corresponding direction vector is found by the difference between such bounds
            Args:
                cols: 1d array with image column coordinates
                rows: 1d array with image row coordinates
                rpc: RPC model with the localization function associated to the satellite image
                min_alt: float, the minimum altitude observed in the image
                max_alt: float, the maximum altitude observed in the image
            Returns:
                rays: (h*w, 8) tensor of floats encoding h*w rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
            """

    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    # assume the points of maximum altitude are those closest to the camera
    lons, lats = rpc.localization(cols, rows, max_alts)
    if cs == 'ecef':
        x_near, y_near, z_near = sat_utils.latlon_to_ecef_custom(lats, lons, max_alts)
    elif cs == 'utm':
        x_near, y_near = sat_utils.utm_from_latlon(lats, lons)
        z_near = max_alts
    xyz_near = np.vstack([x_near, y_near, z_near]).T

    # similarly, the points of minimum altitude are the furthest away from the camera
    lons, lats = rpc.localization(cols, rows, min_alts)
    if cs == 'ecef':
        x_far, y_far, z_far = sat_utils.latlon_to_ecef_custom(lats, lons, min_alts)
    elif cs == 'utm':
        x_far, y_far = sat_utils.utm_from_latlon(lats, lons)
        z_far = min_alts
    xyz_far = np.vstack([x_far, y_far, z_far]).T

    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)

    # create a stack with the rays origin, direction vector and near-far bounds
    rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
    rays = rays.type(torch.FloatTensor)
    return rays

def get_zone(cols, rows, rpc, min_alt):
    import utm
    lons, lats = rpc.localization(cols[0], rows[0], min_alt)
    n = utm.latlon_to_zone_number(lats, lons)
    l = utm.latitude_to_zone_letter(lats)
    return n, l

def ScaleImg(img, scalefacter=1., aoi_id='', min=0, max=1):
    if 1:
        train_utils.PrintMMM('rgb_gt after load ', img, dim=3)
        if np.abs(scalefacter) < 1e-5:
            print('min_, max_, scale: {:.3f} {:.3f} {:.3f}'.format(min, max, 1./(max - min)))
            img = (img - min)/(max - min)
        elif scalefacter < 0:
            for i in range(3):
                min_, max_ = np.min(img[..., i]), np.max(img[..., i])
                print('Channel, min_, max_, scale: {} {:.3f} {:.3f} {:.3f}'.format(i, min_, max_, 1./(max_ - min_)))
                img[..., i] = (img[..., i] - min_)/(max_ - min_)
        else:
            print('scalefacter: {:.3f}'.format(scalefacter))
            img = img * scalefacter
        train_utils.PrintMMM('rgb_gt after scale', img, dim=3)
        img = np.clip(img, 0., 1.)
        train_utils.PrintMMM('rgb_gt after clip ', img, dim=3)

    return img

def load_tensor_from_rgb_geotiff(img_path, downscale_factor, imethod=Image.BILINEAR, scalefacter=1/255., aoi_id='', min=0, max=1): #Image.BICUBIC leads to noisy pixel errors
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0))
        img = ScaleImg(img, scalefacter=scalefacter, aoi_id=aoi_id, min=min, max=max)
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = T.ToTensor()(img)  # (3, h, w)
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)
    return rgbs

class SatelliteRGBDEPDataset(Dataset):
    def __init__(self, args, split="train"):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
            cache_dir: string, directory containing precomputed rays
        """
        self.args = args
        self.corrscale = args.corrscale
        self.stdscale = args.stdscale
        self.margin = args.margin
        self.cs = args.cs
        self.gpu_id = args.gpu_id
        self.infile_postfix = args.infile_postfix
        self.json_dir = args.root_dir
        self.img_dir = args.img_dir if args.img_dir is not None else args.root_dir
        self.cache_dir = args.cache_dir
        self.gt_dir = args.gt_dir
        self.aoi_id = args.aoi_id
        self.train = split == "train"
        self.img_downscale = float(args.img_downscale)
        self.white_back = False
        self.depth_dir = self.json_dir+'/'+args.inputdds+'/'
        self.n = '17'       #only useful for get_dsm_from_nerf_prediction, will be recalculated when loading data
        self.l = 'R'

        print('Load SatelliteRGBDEPDataset with corrscale: ', self.corrscale)

        assert os.path.exists(args.root_dir), f"root_dir {root_dir} does not exist"
        assert os.path.exists(args.img_dir), f"img_dir {img_dir} does not exist"

        # load scaling params
        if not os.path.exists(f"{self.json_dir}/scene.loc"):
            self.init_scaling_params()
        else:
            print(f"{self.json_dir}/scene.loc already exist, hence skipped scaling")
        d = sat_utils.read_dict_from_json(os.path.join(self.json_dir, "scene.loc"), aoi_id=self.aoi_id, mod_alt_bound=self.args.mod_alt_bound)
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))
        if 1:
            print("X_scale, X_offset: ", d["X_scale"], d["X_offset"])
            print("Y_scale, Y_offset: ", d["Y_scale"], d["Y_offset"])
            print("Z_scale, Z_offset: ", d["Z_scale"], d["Z_offset"])

        # load dataset split
        if self.train:
            self.load_train_split()
        else:
            self.load_val_split()

    def load_train_split(self):
        with open(os.path.join(self.json_dir, "train{}".format(self.infile_postfix)), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        n_train_ims = len(json_files)
        self.json_files_train = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        self.all_rays, self.all_rgbs, self.all_ids, self.all_rows, self.all_cols = self.load_data(self.json_files_train, verbose=True, sType='train')

        self.all_deprays, self.all_depths, self.all_valid_depth, self.all_depth_stds, self.all_normals, self.all_valid_normal = self.load_depth_data(self.json_files_train, self.depth_dir, verbose=True)

    def load_val_split(self):
        with open(os.path.join(self.json_dir, "test{}".format(self.infile_postfix)), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        self.json_files_all = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        # add an extra image from the training set to the validation set (for debugging purposes)
        with open(os.path.join(self.json_dir, "train{}".format(self.infile_postfix)), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        self.json_files_all = [os.path.join(self.json_dir, json_p) for json_p in json_files] + self.json_files_all
        n_train_ims = len(json_files)
        self.all_ids = [i + n_train_ims for i, j in enumerate(self.json_files)]
        add_train = 1
        if len(self.json_files) < self.gpu_id+1:
            add_train = np.min([len(json_files), self.gpu_id+1-len(self.json_files)])
        print('self.gpu_id, add_train: ', self.gpu_id, add_train)
        self.add_train = add_train
        for i in range(add_train-1, -1, -1):
            self.json_files = [os.path.join(self.json_dir, json_files[i])] + self.json_files
            self.all_ids = [i] + self.all_ids
        print('load_val_split: self.all_ids, self.json_files: ', self.all_ids, self.json_files)
        print('self.json_files_all (train + test):', self.json_files_all)

        self.samples = {}
        for idx, json_p in enumerate(self.json_files):
            rays, rgbs, _, rows, cols = self.load_data([self.json_files[idx]], sType='val')
            ts = self.all_ids[idx] * torch.ones(rays.shape[0], 1)
            d = sat_utils.read_dict_from_json(self.json_files[idx], aoi_id=self.aoi_id, mod_alt_bound=self.args.mod_alt_bound)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            mask = self.load_mask([self.json_files[idx]], h, w, sType='val')
            img_id = sat_utils.get_file_id(d["img"])
            sample = {"rays": rays, "rgbs": rgbs, "ts": ts.long(), "src_id": img_id, "h": h, "w": w, "rows": rows, "cols": cols, "idx": idx, "mask": mask}
            sample['save_cross'] = False
            sample['range'] = self.range
            try:
                _, depths, _, _, normals, valid_normal = self.load_depth_data([self.json_files[idx]], self.depth_dir)
                sample["depths"] = depths
                sample["normals"] = normals
                sample["valid_normal"] = valid_normal
            except:
                print('Validation depth file for {} does not exist, hence not loaded'.format(self.json_files[idx]))
            if idx < self.add_train:
                sample['is_val'] = False
                if idx == 0:
                    sample['save_cross'] = True
            else:
                sample['is_val'] = True

            self.samples['{}'.format(idx)] = sample

    def init_scaling_params(self):
        print("Could not find a scene.loc file in the root directory, creating one...")
        print("Warning: this can take some minutes")
        all_json = glob.glob("{}/*.json".format(self.json_dir))
        all_rays = []
        for json_p in all_json:
            d = sat_utils.read_dict_from_json(json_p, aoi_id=self.aoi_id, mod_alt_bound=self.args.mod_alt_bound)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt, cs=self.cs)
            all_rays += [rays]
        all_rays = torch.cat(all_rays, 0)
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_params(all_points[:, 2])
        sat_utils.write_dict_to_json(d, f"{self.json_dir}/scene.loc")
        print("... done !")

    def load_mask(self, json_files, h, w, verbose=False, sType='none'):
        all_masks = []
        for t, json_p in enumerate(json_files):
            mask_p = json_p[:-5] + '_mask.tif'
            # read json, image path and id
            if os.path.exists(mask_p) == False or os.path.isfile(mask_p) == False:
                print(mask_p, 'not exist or is not a file, hence is set to all one')
                mask = np.ones(h*w)
            else:
                mask_img = np.asarray(Image.open(mask_p)).flatten()
                train_utils.PrintMMM('mask_img', mask_img)
                mask = np.zeros_like(mask_img)
                print(np.where(mask_img>0)[0])
                mask[np.where(mask_img>0)[0]] = 1
                train_utils.PrintMMM('mask', mask)

            mask = torch.from_numpy(mask)
            all_masks += [mask]

        all_masks = torch.cat(all_masks, 0)  
        all_masks = all_masks.type(torch.bool)

        return all_masks

    def get_pixelval_bound(self):
        with open(os.path.join(self.json_dir, "test{}".format(self.infile_postfix)), "r") as f:
            json_files = f.read().split("\n")
        json_files_all = [os.path.join(self.json_dir, json_p) for json_p in json_files[:-1]]
        with open(os.path.join(self.json_dir, "train{}".format(self.infile_postfix)), "r") as f:
            json_files = f.read().split("\n")
        json_files_all = [os.path.join(self.json_dir, json_p) for json_p in json_files[:-1]] + json_files_all
        
        min_, max_ = sys.float_info.max, -sys.float_info.max
        print('min_, max_', min_, max_)
        for t, json_p in enumerate(json_files_all):        
            if os.path.exists(json_p) == False or os.path.isfile(json_p) == False:
                continue
            d = sat_utils.read_dict_from_json(json_p, aoi_id=self.aoi_id, mod_alt_bound=self.args.mod_alt_bound)
            img_p = os.path.join(self.img_dir, d["img"])
            with rasterio.open(img_p, 'r') as f:
                img = f.read()
                min, max = np.min(img), np.max(img)
                print(img_p, min, max)
                min_ = min if min < min_ else min_
                max_ = max if max > max_ else max_
        print('min_, max_', min_, max_)
        return min_, max_

    def load_data(self, json_files, verbose=False, sType='none'):
        """
        Load all relevant information from a set of json files
        Args:
            json_files: list containing the path to the input json files
        Returns:
            all_rays: (N, 11) tensor of floats encoding all ray-related parameters corresponding to N rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
                      columns 8,9,10 correspond to the sun direction vectors
            all_rgbs: (N, 3) tensor of floats encoding all the rgb colors corresponding to N rays
        """
        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        all_rows, all_cols = [], []
        if np.abs(self.args.scale) < 1e-5:
            min_, max_ = self.get_pixelval_bound()
        else:
            min_, max_ = 0, 1
        for t, json_p in enumerate(json_files):

            # read json, image path and id
            if os.path.exists(json_p) == False or os.path.isfile(json_p) == False:
                print(json_p, 'not exist or is not a file, hence skipped')
                continue

            d = sat_utils.read_dict_from_json(json_p, aoi_id=self.aoi_id, mod_alt_bound=self.args.mod_alt_bound)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = sat_utils.get_file_id(d["img"])

            # get rgb colors
            rgbs = load_tensor_from_rgb_geotiff(img_p, self.img_downscale, scalefacter=self.args.scale, aoi_id=self.args.aoi_id, min=min_, max=max_)
            print(t, img_id, ' rgbs: range: [{:.3f}, {:.3f}], me: {:.3f}'.format(torch.min(rgbs), torch.max(rgbs), torch.mean(rgbs)))

            # get rays
            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if 0: #self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                #numpy flatten, default 'C', means in row-major (C-style) order.
                rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt, cs=self.cs)
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)
            rays = self.normalize_rays(rays)

            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])
            print('load__data: {}/{}'.format(t, len(json_files)), 'sType, img_id: ', sType, img_id)
            if 1:
                showray = rays[[0, int(rays.shape[0]/2), -1], :]
                showsun = sun_dirs[[0, int(rays.shape[0]/2), -1], :]
                cos_l_v = torch.einsum('ij,ij->i', -showray[:, 3:6], showsun)
                ray_d_norm = np.linalg.norm(showray[:, 3:6].cpu(), axis=1)
                sun_d_norm = np.linalg.norm(showsun.cpu(), axis=1)
                g = torch.acos(cos_l_v)
                #print("cos_l_v, ray_d_norm, sun_d_norm, g: ", cos_l_v, ray_d_norm, sun_d_norm, g)
                #print("1st, middle, last rays: ")
                for i in range(1):
                    print(" rays_o: {}; rays_d: {}; near: {}; far: {}; sun_d: {}; cos_l_v: {:.3f}; g: {:.3f} ({:.3f})".format(showray[i, 0:3], showray[i, 3:6], showray[i, 6:7], showray[i, 7:8], showsun[i], cos_l_v[i], g[i], g[i]/3.14*180))
                    #print("  cos_l_v: {}; g: {} ({})".format(cos_l_v[i], g[i], g[i]/3.14*180))            

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rows += [torch.from_numpy(rows).reshape(rays.shape[0], 1)]
            all_cols += [torch.from_numpy(cols).reshape(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]

        all_ids = torch.cat(all_ids, 0)
        all_rows = torch.cat(all_rows, 0)
        all_cols = torch.cat(all_cols, 0)
        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_rgbs = all_rgbs.type(torch.FloatTensor)

        return all_rays, all_rgbs, all_ids, all_rows, all_cols

    def scale_depth(self, feature, height, width, depth=1):
        new_height, new_width = int(height/self.img_downscale), int(width/self.img_downscale)
        new_feature = torch.nn.functional.interpolate(feature.reshape(1, 1, height, width, depth), size=(new_height, new_width, depth))     #, mode='bilinear')
        return new_feature.squeeze().reshape(new_height*new_width, depth).squeeze()

    def load_depth_data(self, json_files, depth_dir, verbose=False):
        all_deprays, all_depths, all_sun_dirs, all_weights = [], [], [], []
        all_depth_stds = []
        all_valid_depth = []
        all_normals = []
        all_valid_normal = []
        depth_min = 0
        depth_max = 0

        for t, json_p in enumerate(json_files):
            # read json
            d = sat_utils.read_dict_from_json(json_p, aoi_id=self.aoi_id, mod_alt_bound=self.args.mod_alt_bound)
            img_id = sat_utils.get_file_id(d["img"])

            height = d["height"]
            width = d["width"]
            pts2d = []
            idx_cur = 0
            pts2d = np.loadtxt(depth_dir+img_id+"_2DPts.txt", dtype='int') #order: width first, height latter, like building a wall
            pts2d = pts2d.reshape(-1,2)
            if self.cs == 'ecef':
                pts3d_file = depth_dir+img_id+"_3DPts_ecef.txt"
            elif self.cs == 'utm':
                pts3d_file = depth_dir+img_id+"_3DPts.txt"
            pts3d = np.loadtxt(pts3d_file, dtype='float')
            pts3d = pts3d.reshape(-1,3)
            current_weights = np.loadtxt(depth_dir+img_id+"_Correl.txt", dtype='float')

            valid_depth = torch.zeros(height, width)
            valid_depth[pts2d[:,1], pts2d[:,0]] = torch.ones(pts2d.shape[0])
            valid_depth = valid_depth.flatten()

            CorrelMin = current_weights.min()
            CorrelMax = current_weights.max()
            current_weights = (current_weights-CorrelMin)/(CorrelMax-CorrelMin)
            current_weights = self.corrscale*current_weights

            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            pts2d = pts2d / self.img_downscale

            cols, rows = pts2d.T
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            rays = get_rays(cols, rows, rpc, min_alt, max_alt, cs=self.cs)
            rays = self.normalize_rays(rays)

            min_alt_pts, max_alt_pts = np.min(pts3d[:, 2]), np.max(pts3d[:, 2])
            if self.cs == 'utm' and (min_alt_pts < min_alt or max_alt_pts > max_alt):
                print('Input 3D points range [{:.3f}, {:.3f}] is OUT of bounding range [{:.3f}, {:.3f}]'.format(min_alt_pts, max_alt_pts, min_alt, max_alt))
            else:
                print('Input 3D points range [{:.3f}, {:.3f}] is WITHIN of bounding range [{:.3f}, {:.3f}]'.format(min_alt_pts, max_alt_pts, min_alt, max_alt))

            if t == 0:
                self.n, self.l = get_zone(cols, rows, rpc, min_alt)

            # normalize the 3d coordinates of the tie points observed in the current view
            pts3d = torch.from_numpy(pts3d).type(torch.FloatTensor)
            pts3d[:, 0] -= self.center[0]
            pts3d[:, 1] -= self.center[1]
            pts3d[:, 2] -= self.center[2]
            train_utils.PrintMMM('pts3ds z after offset', pts3d[..., 2], show_val=False)
            pts3d[:, 0] /= self.range
            pts3d[:, 1] /= self.range
            pts3d[:, 2] /= self.range
            train_utils.PrintMMM('pts3ds z after scaling', pts3d[..., 2], show_val=False)

            # compute depths
            tmmmp = pts3d - rays[:, :3]
            train_utils.PrintMMM('pts3ds z after -rays_o', tmmmp[..., 2], show_val=False)
            depths = torch.linalg.norm(pts3d - rays[:, :3], axis=1)
            train_utils.PrintMMM('depths', depths)
            current_weights = torch.from_numpy(current_weights).type(torch.FloatTensor)
            
            depths_padded = torch.ones(height*width) * torch.mean(depths)
            depths_padded[np.where(valid_depth>0)[0]] = depths
            depths_padded = self.scale_depth(depths_padded, height, width)

            train_utils.PrintMMM('rays', rays)
            if 1:
                rays_d = rays[:,3:6]
                nadir = torch.zeros_like(rays_d)
                nadir[:,-1] = -1
                angle_weights = (rays_d * nadir).sum(dim=-1)
                show_val = False
                train_utils.PrintMMM('rays_d x', rays_d[..., 0], show_val=show_val)
                train_utils.PrintMMM('rays_d y', rays_d[..., 1], show_val=show_val)
                train_utils.PrintMMM('rays_d z', rays_d[..., 2], show_val=show_val)
                train_utils.PrintMMM('nadir x', nadir[..., 0], show_val=show_val)
                train_utils.PrintMMM('nadir y', nadir[..., 1], show_val=show_val)
                train_utils.PrintMMM('nadir z', nadir[..., 2], show_val=show_val)
                train_utils.PrintMMM('angle_weights', angle_weights, show_val=show_val)
                train_utils.PrintMMM('current_weights', current_weights, show_val=show_val)
                current_weights = current_weights * angle_weights
                train_utils.PrintMMM('current_weights', current_weights, show_val=show_val)
            weights_padded = torch.zeros(height*width)
            weights_padded[np.where(valid_depth>0)[0]] = current_weights
            weights_padded = self.scale_depth(weights_padded, height, width)

            current_depth_std = self.stdscale*(torch.ones_like(current_weights) - current_weights) + torch.ones_like(current_weights)*self.margin
            depth_std_padded = torch.zeros(height*width)
            depth_std_padded[np.where(valid_depth>0)[0]] = current_depth_std
            depth_std_padded = self.scale_depth(depth_std_padded, height, width)

            rays_padded = torch.zeros(height*width, 8)
            rays_padded[np.where(valid_depth>0)[0],:] = rays
            rays_padded = self.scale_depth(rays_padded, height, width, 8)
            valid_depth = self.scale_depth(valid_depth, height, width)

            if 1:
                pts3d_padded = torch.zeros(height*width, 3)
                pts3d_padded[np.where(valid_depth>0)[0]] = pts3d
                normal_padded = torch.zeros_like(pts3d_padded)
                normal_padded[:, 2] = 1.
                normal_padded_, valid_normal = sat_utils.calc_normal_from_pts3d(pts3d_padded.reshape(height, width, 3), valid_depth.reshape(height, width))
                normal_padded[np.where(valid_normal>0)[0]] = normal_padded_[np.where(valid_normal>0)[0]]
                print('------', json_p)
                show_val=False
                places = 3
                pts3ds = pts3d_padded[np.where(valid_depth>0)[0]]
                train_utils.PrintMMM('depths', depths, show_val=show_val, places=places)
                normals = normal_padded[np.where(valid_normal>0)[0]]
                train_utils.PrintMMM('normals', normals, show_val=show_val, places=places, dim=3)
                train_utils.check_vec0('normals', normals, Print=False)
                train_utils.PrintMMM('ray_o x', rays[..., 0:3], show_val=show_val, places=places, dim=3)
                train_utils.PrintMMM('ray_d x', rays[..., 3:6], show_val=show_val, places=places, dim=3)

            all_valid_depth += [valid_depth]
            all_depths += [depths_padded[:, np.newaxis]]
            all_weights += [weights_padded[:, np.newaxis]]
            all_depth_stds += [depth_std_padded]
            all_deprays += [rays_padded]
            all_normals += [normal_padded]
            all_valid_normal += [valid_normal]

        all_valid_depth = torch.cat(all_valid_depth, 0)
        all_deprays = torch.cat(all_deprays, 0)  # (len(json_files)*h*w, 8)
        all_depths = torch.cat(all_depths, 0)  # (len(json_files)*h*w, 1)
        all_weights = torch.cat(all_weights, 0)
        all_depth_stds = torch.cat(all_depth_stds, 0)
        all_depth_stds = all_depth_stds*(depth_max-depth_min)
        all_normals = torch.cat(all_normals, 0)
        all_valid_normal = torch.cat(all_valid_normal, 0)
        all_depths = torch.hstack([all_depths, all_weights])  # (len(json_files)*h*w, 11)
        all_deprays = all_deprays.type(torch.FloatTensor)
        all_depths = all_depths.type(torch.FloatTensor)
        all_normals = all_normals.type(torch.FloatTensor)
        all_valid_normal = all_valid_normal.type(torch.FloatTensor)

        return all_deprays, all_depths, all_valid_depth, all_depth_stds, all_normals, all_valid_normal

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range    #near of the ray
        rays[:, 7] /= self.range    #far of the ray
        return rays

    def get_sun_dirs(self, sun_elevation_deg, sun_azimuth_deg, n_rays):
        """
        Get sun direction vectors
        Args:
            sun_elevation_deg: float, sun elevation in  degrees
            sun_azimuth_deg: float, sun azimuth in degrees
            n_rays: number of rays affected by the same sun direction
        Returns:
            sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
        """
        sun_el = np.radians(sun_elevation_deg)
        sun_az = np.radians(sun_azimuth_deg)
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
        sun_dirs = sun_dirs.type(torch.FloatTensor)
        return sun_dirs

    def calc_normal_from_depth_v2(self, rays, depth, height, width, valid_depth=None):
        easts, norths, alts = self.get_latlonalt_from_nerf_prediction(rays, depth)
        pts3d = np.vstack([easts, norths, alts]).T
        pts3d = torch.from_numpy(pts3d).type(torch.FloatTensor)
        
        normals, valid_normal = sat_utils.calc_normal_from_pts3d(pts3d.reshape(height, width, 3))
        return normals, valid_normal

    def calc_nr_mae(self, nr1, nr2):
        return torch.sum(torch.abs(nr1 - nr2))/nr1.shape[0]/nr1.shape[1]/3.*100

    def calc_normal_from_depth(self, depth, alt_scl=1):
        normals = torch.zeros((depth.shape[0], depth.shape[1], 3), device=depth.device)
        dzdx = self.range * (depth[2:, :] - depth[:-2, :]) / 2.0
        dzdy = self.range * (depth[:, 2:] - depth[:, :-2]) / 2.0
        normals[1:-1, :, 0] = dzdx
        normals[:, 1:-1, 1] = dzdy  #delete the negative because the depth is reversed to elevation
        normals[1:-1, :, 2] = torch.ones_like(dzdx) * alt_scl

        normals = train_utils.l2_normalize(normals)

        return normals

    def get_latlonalt_from_nerf_prediction(self, rays, depth, bPrint=False):
        """
        Compute an image of altitudes from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
        Returns:
            easts: numpy vector of length h*w with the utm easts of the predicted points
            norths: numpy vector of length h*w with the utm norths of the predicted points
            alts: numpy vector of length h*w with the altitudes of the predicted points
        """

        # convert inputs to double (avoids loss of resolution later when the tensors are converted to numpy)
        rays = rays.double()
        depth = depth.double()

        # use input rays + predicted sigma to construct a point cloud
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        xyz_n = rays_o + rays_d * depth.view(-1, 1)
       
        # denormalize prediction to obtain ECEF coordinates
        xyz = xyz_n * self.range
        xyz[:, 0] += self.center[0]
        xyz[:, 1] += self.center[1]
        xyz[:, 2] += self.center[2]

        # convert to lat-lon-alt
        xyz = xyz.data.numpy()
        if self.cs == 'ecef':
            lats, lons, alts = sat_utils.ecef_to_latlon_custom(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            easts, norths = sat_utils.utm_from_latlon(lats, lons)
        elif self.cs == 'utm':
            easts, norths, alts = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return easts, norths, alts

    def get_dsm_from_nerf_prediction(self, rays, depth, dsm_path=None, roi_txt=None):
        """
        Compute a DSM from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
            dsm_path (optional): string, path to output DSM, in case you want to write it to disk
            roi_txt (optional): compute the DSM only within the bounds of the region of interest of the txt
        Returns:
            dsm: (h, w) numpy array with the output dsm
        """

        # get point cloud from nerf depth prediction
        if self.cs == 'ecef':
            lats, lons, alts = self.get_latlonalt_from_nerf_prediction(rays, depth)
            easts, norths = sat_utils.utm_from_latlon(lats, lons)
        elif self.cs == 'utm':
            easts, norths, alts = self.get_latlonalt_from_nerf_prediction(rays, depth)
        print('get_dsm_from_nerf_prediction, alt range: [{:.2f}, {:.2f}]'.format(np.min(alts), np.max(alts)))
        cloud = np.vstack([easts, norths, alts]).T

        # (optional) read region of interest, where lidar GT is available
        if roi_txt is not None:
            gt_roi_metadata = np.loadtxt(roi_txt)
            xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
            xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
            resolution = gt_roi_metadata[3]
            yoff += ysize * resolution  # weird but seems necessary ?
        else:
            resolution = 0.5
            xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
            ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
            xoff = np.floor(xmin / resolution) * resolution
            xsize = int(1 + np.floor((xmax - xoff) / resolution))
            yoff = np.ceil(ymax / resolution) * resolution
            ysize = int(1 - np.floor((ymin - yoff) / resolution))

        from plyflatten import plyflatten
        from plyflatten.utils import rasterio_crs, crs_proj
        import utm
        import affine
        import rasterio

        # run plyflatten
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))

        crs_proj = rasterio_crs(crs_proj("{}{}".format(self.n, self.l), crs_type="UTM"))

        # (optional) write dsm to disk
        if dsm_path is not None:
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
            profile = {}
            profile["dtype"] = dsm.dtype
            profile["height"] = dsm.shape[0]
            profile["width"] = dsm.shape[1]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["nodata"] = float("nan")
            profile["crs"] = crs_proj
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
            with rasterio.open(dsm_path, "w", **profile) as f:
                f.write(dsm[:, :, 0], 1)

        return dsm

    def __len__(self):
        # compute length of dataset
        if self.train:
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):
        # take a batch from the dataset
        if self.train:
            #rays_ref: depth rays padded to the same size of rgb rays, for debug only (to verify the correspondence)
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx], "ts": self.all_ids[idx].long(), "valid_depth": self.all_valid_depth[idx].long(), "depths": self.all_depths[idx], "rays_ref": self.all_deprays[idx], "depth_std": self.all_depth_stds[idx], "normals": self.all_normals[idx], "rows": self.all_rows[idx], "cols": self.all_cols[idx], "valid_normal": self.all_valid_normal[idx]}
        else:
            sample = self.samples['{}'.format(idx)]

        return sample
