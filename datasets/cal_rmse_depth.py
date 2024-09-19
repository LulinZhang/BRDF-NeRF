import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sat_utils import dsm_pointwise_diff, Cloud2Grid, compute_mae_and_save_dsm_diff
from plyflatten.utils import rasterio_crs, crs_proj
from rasterio.enums import Resampling
from rasterio.transform import Affine
import glob
import datetime
import shutil


def get_dsm_from_dense_depth(densedepth_file, zonestring, dsm_out_path, resolution, roi_txt=None, dst_transform=None):
    pts3d = np.loadtxt(densedepth_file, dtype='float')
    easts = pts3d[:,0]
    norths = pts3d[:,1]
    alts = pts3d[:,2]    
    cloud = np.vstack([easts, norths, alts]).T

    # (optional) read region of interest, where lidar GT is available
    if roi_txt is not None:
        gt_roi_metadata = np.loadtxt(roi_txt)
        xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
        xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
        resolution = gt_roi_metadata[3]
        yoff += ysize * resolution  # weird but seems necessary ?
    else:
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

    crs_proj = rasterio_crs(crs_proj(zonestring, crs_type="UTM"))

    # (optional) write dsm to disk
    if dsm_out_path is not None:
        os.makedirs(os.path.dirname(dsm_out_path), exist_ok=True)
        profile = {}
        profile["dtype"] = dsm.dtype
        profile["height"] = dsm.shape[0]
        profile["width"] = dsm.shape[1]
        profile["count"] = 1
        profile["driver"] = "GTiff"
        profile["nodata"] = float("nan")
        profile["crs"] = crs_proj
        if dst_transform != None:
            profile["transform"] = dst_transform
        else:
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
        with rasterio.open(dsm_out_path, "w", **profile) as f:
            f.write(dsm[:, :, 0], 1)


def get_info(aoi_id):
    if aoi_id[0:3] == 'JAX':
        zonestring = "17R"
    elif aoi_id[1:3] == 'ji':
        zonestring = "38N"
    elif aoi_id[0:3] == 'Itl':
        zonestring = "33N"
    elif aoi_id[:3] == 'Lzh':
        zonestring = "48N"

    resolution = 0.5

    return zonestring, resolution #, upscale_factor

def CalcMAE(in_dsm_path, gt_dsm_path, gt_roi_path, gt_seg_path, rdsm_path, rdsm_diff_path, mask_out_path, outputMask=False):
    gt_roi_metadata = np.loadtxt(gt_roi_path)

    diff, _ = dsm_pointwise_diff(in_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path, out_rdsm_path=rdsm_path, out_err_path=rdsm_diff_path)

    isnan = np.isnan(diff)
    if outputMask:
        mask=(isnan==False)
        Image.fromarray(mask).save(mask_out_path)
    nanNb = np.sum(isnan == True)
    totalNb = diff.shape[0]*diff.shape[1]

    return diff, nanNb, totalNb

def cal_rmse_depth(densedepth_file, gt_dir, aoi_id, img_id=''):
    ###########################prepare dsm from dense depth###########################
    depth_img_path = os.path.join(os.path.dirname(densedepth_file), 'depth_img')
    os.makedirs(os.path.dirname(depth_img_path), exist_ok=True)
    output_depth_imgs = os.path.join(depth_img_path, os.path.basename(densedepth_file))
    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    dsm_out_path = output_depth_imgs[:-4] +"_{}_tmp.tif".format(unique_identifier)
    roi_txt = None
    dst_transform = None

    #zonestring, resolution, upscale_factor = get_info(aoi_id)
    zonestring, resolution = get_info(aoi_id)

    get_dsm_from_dense_depth(densedepth_file, zonestring, dsm_out_path, resolution, roi_txt=roi_txt, dst_transform=dst_transform)

    dsm_in_path = dsm_out_path
    dsm_out_path = output_depth_imgs[:-4] +".tif"
    shutil.copyfile(dsm_in_path, dsm_out_path)
    os.remove(dsm_in_path)

    if 1:
        dsm_in_path = dsm_out_path
        dsm_out_path = dsm_out_path[:-4] +"_Grid.tif"
        if os.path.exists(dsm_out_path) == False:
            Cloud2Grid(dsm_in_path, dsm_out_path, Print=True) #, interp=False)
        else:
            print('{} already existed, hence skipped'.format(dsm_out_path))

    ###########################CalcMAE###########################
    gt_roi_path = gt_dir + aoi_id + '_DSM.txt'
    gt_seg_path = None
    gt_dsm_path = gt_dir + aoi_id + '_DSM.tif'

    rdsm_path = dsm_out_path[:-4] + '_aligned.tif'
    rdsm_diff_path = dsm_out_path[:-4] + '_dod.tif'
    mask_out_path = dsm_out_path[:-4] + 'mask.tif'
    outputMask = True

    mae_, mae_in, mae_out, diff, mae_nr, diff_nr = compute_mae_and_save_dsm_diff(dsm_out_path, img_id, aoi_id, gt_dir, os.path.dirname(dsm_out_path), -1, save=True, calc_mae_nr=True)
    print('{} depth supervision, mae_: {:.3f}, mae_in: {:.3f}, mae_out: {:.3f}, mae_nr: {:.3f}'.format(img_id, mae_, mae_in, mae_out, mae_nr))

    return mae_, mae_in, mae_out, diff, mae_nr, diff_nr
