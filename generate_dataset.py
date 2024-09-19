#/home/LZhang/anaconda3/envs/ba/lib/python3.8/site-packages/rpcm
#/home/LZhang/anaconda3/envs/ba/lib/python3.8/site-packages/bundle_adjust
import rpcm
import glob
import os
import numpy as np
import srtm4
import shutil
import sys
import json
from sat_utils import get_file_id, read_dict_from_json
import rasterio
import os
import re
import cv2
from PIL import Image

def rio_open(*args,**kwargs):
    import rasterio
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return rasterio.open(*args,**kwargs)

def get_image_lonlat_aoi(rpc, h, w):
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    cols, rows, alts = [0,w,w,0], [0,0,h,h], [z]*4
    lons, lats = rpc.localization(cols, rows, alts)
    lonlat_coords = np.vstack((lons, lats)).T
    geojson_polygon = {"coordinates": [lonlat_coords.tolist()], "type": "Polygon"}
    x_c = lons.min() + (lons.max() - lons.min())/2
    y_c = lats.min() + (lats.max() - lats.min())/2
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon

#Input: cropped images (tif files only)
#Output: 
def run_ba(img_dir, output_dir):

    from bundle_adjust.cam_utils import SatelliteImage
    from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
    from bundle_adjust import loader

    # load input data
    os.makedirs(output_dir, exist_ok=True)
    myimages = sorted(glob.glob(img_dir + "/*.tif"))
    myrpcs = [rpcm.rpc_from_geotiff(p) for p in myimages]
    input_images = [SatelliteImage(fn, rpc) for fn, rpc in zip(myimages, myrpcs)]
    ba_input_data = {}
    ba_input_data['in_dir'] = img_dir
    ba_input_data['out_dir'] = os.path.join(output_dir, "ba_files")
    ba_input_data['images'] = input_images
    print('Input data set!\n')

    # redirect all prints to a bundle adjustment logfile inside the output directory
    os.makedirs(ba_input_data['out_dir'], exist_ok=True)
    path_to_log_file = "{}/bundle_adjust.log".format(ba_input_data['out_dir'])
    print("Running bundle adjustment for RPC model refinement ...")
    print("Path to log file: {}".format(path_to_log_file))
    log_file = open(path_to_log_file, "w+")
    sys.stdout = log_file
    sys.stderr = log_file
    # run bundle adjustment
    #tracks_config = {'FT_reset': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_K": 300}
    tracks_config = {'FT_reset': False, 'FT_save': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based'}
    ba_extra = {"cam_model": "rpc"}
    ba_pipeline = BundleAdjustmentPipeline(ba_input_data, tracks_config=tracks_config, extra_ba_config=ba_extra)
    ba_pipeline.run()
    # close logfile
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()
    print("... done !")
    print("Path to output files: {}".format(ba_input_data['out_dir']))

    # save all bundle adjustment parameters in a temporary directory
    ba_params_dir = os.path.join(ba_pipeline.out_dir, "ba_params")
    os.makedirs(ba_params_dir, exist_ok=True)
    np.save(os.path.join(ba_params_dir, "pts_ind.npy"), ba_pipeline.ba_params.pts_ind)
    np.save(os.path.join(ba_params_dir, "cam_ind.npy"), ba_pipeline.ba_params.cam_ind)
    np.save(os.path.join(ba_params_dir, "pts3d.npy"), ba_pipeline.ba_params.pts3d_ba - ba_pipeline.global_transform)
    np.save(os.path.join(ba_params_dir, "pts2d.npy"), ba_pipeline.ba_params.pts2d)
    fnames_in_use = [ba_pipeline.images[idx].geotiff_path for idx in ba_pipeline.ba_params.cam_prev_indices]
    loader.save_list_of_paths(os.path.join(ba_params_dir, "geotiff_paths.txt"), fnames_in_use)

#Input：NTF files，ba results，images，dsms
#Output：json file （"img"，  "height"，  "width"，  "sun_elevation"，  "sun_azimuth"，  "acquisition_date"，  "geojson"，  "min_alt"，  "max_alt"，  "rpc"）
def create_dataset_from_DFC2019_data_zll(nerf_dir, aoi_id, img_dir, dfc_dir, output_dir, path_to_dsm, use_ba=False, min_alt=None, max_alt=None):    #, aoi_id = 'JAX_068'

    # create a json file of metadata for each input image
    # contains: h, w, rpc, sun elevation, sun azimuth, acquisition date
    #           + geojson polygon with the aoi of the image
    os.makedirs(output_dir, exist_ok=True)
    if use_ba:
        from bundle_adjust import loader
        geotiff_paths = loader.load_list_of_paths(os.path.join(output_dir, "ba_files/ba_params/geotiff_paths.txt"))
        geotiff_paths = [p.replace("/pan_crops/", "/crops/") for p in geotiff_paths]
        geotiff_paths = [p.replace("PAN.tif", "RGB.tif") for p in geotiff_paths]
        ba_geotiff_basenames = [os.path.basename(x) for x in geotiff_paths]
        ba_kps_pts3d_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/pts_ind.npy"))
        ba_kps_cam_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/cam_ind.npy"))
        ba_kps_pts2d = np.load(os.path.join(output_dir, "ba_files/ba_params/pts2d.npy"))
    else:
        geotiff_paths = sorted(glob.glob(img_dir + "/*.tif"))

    print('nerf_dir', nerf_dir)
    newoutput_dir = nerf_dir + '/root_dir/'
    os.makedirs(newoutput_dir, exist_ok=True)
    newoutput_dir += '/crops_rpcs_ba_v2/'
    os.makedirs(newoutput_dir, exist_ok=True)
    newoutput_dir += '/'+aoi_id+'/'
    os.makedirs(newoutput_dir, exist_ok=True)

    msi_p = dfc_dir + '/pleiades_sun_angles.txt'
    print('sun_angle_file', msi_p)
    sun_angles = np.loadtxt(msi_p, dtype='str') #, dtype='float')
    for rgb_p in geotiff_paths:
        d = {}
        d["img"] = os.path.basename(rgb_p)

        idx = -1
        for i, item in enumerate(sun_angles):
            if sun_angles[i][0] in d["img"]:
                idx = i
                break
        print(idx, d["img"], sun_angles[idx])

        src = rio_open(rgb_p)
        d["height"] = int(src.meta["height"])
        d["width"] = int(src.meta["width"])
        original_rpc = rpcm.RPCModel(src.tags(ns='RPC'), dict_format="geotiff")

        if(os.path.exists(msi_p) == False):
            print(msi_p, 'not exist')
            continue
        
        d["sun_elevation"] = sun_angles[idx][1].astype(float)
        d["sun_azimuth"] = sun_angles[idx][2].astype(float)
        d["acquisition_date"] = '20130126'
        d["geojson"] = get_image_lonlat_aoi(original_rpc, d["height"], d["width"])

        src = rio_open(path_to_dsm)
        dsm = src.read()[0, :, :]
        d["min_alt"] = int(np.round(dsm.min() - 1)) if min_alt == None else min_alt
        d["max_alt"] = int(np.round(dsm.max() + 1)) if max_alt == None else max_alt

        if use_ba:
            # use corrected rpc
            rpc_path = os.path.join(output_dir, "ba_files/rpcs_adj/{}.rpc_adj".format(get_file_id(rgb_p)))
            d["rpc"] = rpcm.rpc_from_rpc_file(rpc_path).__dict__
            #d_out["rpc"] = rpc_rpcm_to_geotiff_format(rpc.__dict__)

            # additional fields for depth supervision
            ba_kps_pts3d_path = os.path.join(output_dir, "ba_files/ba_params/pts3d.npy")
            shutil.copyfile(ba_kps_pts3d_path, os.path.join(newoutput_dir, "pts3d.npy"))
            cam_idx = ba_geotiff_basenames.index(d["img"])
            d["keypoints"] = {"2d_coordinates": ba_kps_pts2d[ba_kps_cam_ind == cam_idx, :].tolist(),
                              "pts3d_indices": ba_kps_pts3d_ind[ba_kps_cam_ind == cam_idx].tolist()}
        else:
            # use original rpc
            d["rpc"] = original_rpc.__dict__

        jsonFile = os.path.join(newoutput_dir, "{}.json".format(get_file_id(rgb_p)))
        with open(jsonFile, "w") as f:
            json.dump(d, f, indent=2)
        print('Finished dumping ', idx, 'th file: ', jsonFile)

    return newoutput_dir

#this is for generating mask to define the trained area, in case test image has larger area.
def generate_img_mask(json_dir, aoi_id='Dji_xxx'):
    all_json = glob.glob("{}/*.json".format(json_dir))
    #ref_img is the most nadir image json file
    if aoi_id[:3] == 'Dji':
        ref_img = 'IMG13_TOC_Converted.json'
    elif aoi_id[:3] == 'ItB' or aoi_id[:3] == 'Itl':
        ref_img = 'IMG_0005_RGB.json'
    elif aoi_id[:3] == 'ItA':
        ref_img = 'IMG_1005_RGB.json'
    else:
        ref_img = None
    if ref_img != None:
        json_p = json_dir + '/' + ref_img
        d = read_dict_from_json(json_p)
        alt = (d["min_alt"] + d["max_alt"])/2.
        tmp = np.array(d["geojson"]["coordinates"][0]) #[0][0]
        lon, lat = tmp[:, 0], tmp[:, 1]
        print('lon, lat, alt', lon, lat, alt)
    for json_p in all_json:
        d = read_dict_from_json(json_p)
        h, w = int(d['height']), int(d['width'])
        mask = np.ones((h, w))*255
        if ref_img != None:
            rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
            col, row = rpc.projection(lon, lat, alt)
            print(json_p, col, row)
            poly = np.array([row, col], dtype=np.int32).T
            #poly = np.clip(poly, 0, None)
            print(poly.shape, poly)
            for i in range(int(d['width'])):
                for j in range(int(d['height'])):
                    inpoly = cv2.pointPolygonTest(poly, (j, i), measureDist=False)
                    if inpoly < 0:
                        mask[j, i] = 0
        mask = Image.fromarray(np.uint8(mask))
        mask.save(json_p[:-5] + '_mask.tif')

def create_train_test_splits(input_sample_ids, test_percent=0.15, min_test_samples=2):

    def shuffle_array(array):
        import random
        v = array.copy()
        random.shuffle(v)
        return v

    n_samples = len(input_sample_ids)
    input_sample_ids = np.array(input_sample_ids)
    all_indices = shuffle_array(np.arange(n_samples))
    n_test = max(min_test_samples, int(test_percent * n_samples))
    n_train = n_samples - n_test

    train_indices = all_indices[:n_train]
    test_indices = all_indices[-n_test:]

    train_samples = input_sample_ids[train_indices].tolist()
    test_samples = input_sample_ids[test_indices].tolist()

    return train_samples, test_samples

#Input: ../Track3-Truth/JAX_068_DSM.txt (there contains origins, size and resolutions in UTM)
#Ouput: Bounding box in longitude/latitude
def read_DFC2019_lonlat_aoi(aoi_id, dfc_dir, DSMFile=None):
    from bundle_adjust import geo_utils
    if aoi_id[:3] == "JAX":
        zonestring = "17" #"17R"
    elif aoi_id[:3] == "Dji":
        zonestring = "38" #"38N"
    elif aoi_id[:3] == "ItA" or aoi_id[:3] == "ItB" or aoi_id[:3] == "Itl":
        zonestring = "33" 
    else:
        raise ValueError("AOI not valid. Expected JAX_(3digits), Dji_(3digits), ItA_(3digits), ItB_(3digits) but received {}".format(aoi_id))

    if DSMFile == None:
        DSMFile = os.path.join(dfc_dir, "Track3-Truth/" + aoi_id + "_DSM.txt")
    roi = np.loadtxt(DSMFile)
    print(DSMFile, roi)
    xoff, yoff, xsize, ysize, resolution = roi[0], roi[1], int(roi[2]), int(roi[2]), roi[3]
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff
    #bounding box
    xmin, xmax, ymin, ymax = ulx, lrx, uly, lry
    easts = [xmin, xmin, xmax, xmax, xmin]
    norths = [ymin, ymax, ymax, ymin, ymin]
    lons, lats = geo_utils.lonlat_from_utm(easts, norths, zonestring)
    lonlat_bbx = geo_utils.geojson_polygon(np.vstack((lons, lats)).T)
    return lonlat_bbx

#Input: Bounding box in longitude/latitude, images and rpcs before cropping
#Output: images and rpcs after cropping
def crop_geotiff_lonlat_aoi(geotiff_path, output_path, lonlat_aoi):
    with rasterio.open(geotiff_path, 'r') as src:
        profile = src.profile
        tags = src.tags()
    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, lonlat_aoi)
    rpc = rpcm.rpc_from_geotiff(geotiff_path)
    rpc.row_offset -= y
    rpc.col_offset -= x
    not_pan = len(crop.shape) > 2
    if not_pan:
        profile["height"] = crop.shape[1]
        profile["width"] = crop.shape[2]
    else:
        profile["height"] = crop.shape[0]
        profile["width"] = crop.shape[1]
        profile["count"] = 1
    with rasterio.open(output_path, 'w', **profile) as dst:
        if not_pan:
            dst.write(crop)
        else:
            dst.write(crop, 1)
        dst.update_tags(**tags)
        dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())

def CropDSM(aoi_id):
    elif aoi_id == 'Dji_005':
        #x, y, w, h = 22500, 9000, 2500, 2500      #when flight line is not parallel with coordiante direction, we need a small DSM zone to crop the image patches, and a big DSM zone to crop GT DSM
        x, y, w, h = 21500, 8000, 4500, 4500       #big DSM
    elif aoi_id == 'Dji_012':
        x, y, w, h = 25000, 9000, 2500, 2500       
        #x, y, w, h = 24000, 8000, 4500, 4500
    elif aoi_id[:2] == 'It' and aoi_id[-3:] == '001':
        x, y, w, h = 5800, 26000, 2500, 2500  
    elif aoi_id[:2] == 'It' and aoi_id[-3:] == '002':
        x, y, w, h = 2800, 21500, 2500, 2500  

    #aoi_id = 'Djibouti'
    #DSMFile = '/home/LZhang/Documents/CNESPostDoc/Djibouti/4DSMsFromArthur/'+aoi_id+'_DSM.txt'
    #aoi_lonlat = read_DFC2019_lonlat_aoi(aoi_id, None, DSMFile)
    #print(aoi_lonlat)
    aoi_lonlat = None

    #crop_geotiff_lonlat_aoi(geotiff_path, output_path, lonlat_aoi):

    if aoi_id[:3] == 'Dji':
        root_dir = '/home/LZhang/Documents/CNESPostDoc/Djibouti/4DSMsFromArthur/'
        geotiff_path = root_dir + 'mns_pleiades_djibouti_50cm_utm.tif'
        left_whole, upper_whole, resolution = 214430.250, 1286916.750, 0.5
    elif aoi_id[:2] == 'It':
        root_dir = '/home/LZhang/Documents/CNESPostDoc/Italy/DSM_0.5m/'
        geotiff_path = root_dir + 'dsm_ama2_nord_subset.tif'
        left_whole, upper_whole, resolution = 350697.75, 4754094.75, 0.5
    output_tif = root_dir+aoi_id+'_DSM.tif'
    output_txt = root_dir+aoi_id+'_DSM.txt'
    #act as we are inputting rpc to avoid loading rpc from geotiff_path
    print(aoi_id)  
    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, aoi_lonlat, InputRPC=1, CropDSM = True, box = [x, y, w, h])
    with rasterio.open(geotiff_path, 'r') as src:
        profile = src.profile
        tags = src.tags()

    not_pan = len(crop.shape) > 2
    if not_pan:
        profile["height"] = crop.shape[1]
        profile["width"] = crop.shape[2]
    else:
        profile["height"] = crop.shape[0]
        profile["width"] = crop.shape[1]
        profile["count"] = 1
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(crop, 1)

    left = left_whole + x * resolution
    lower = upper_whole - (y+h) * resolution
    data = [left, lower, str(h), resolution]
    print(data)
    np.savetxt(output_txt, data, delimiter="\n", fmt="%s")


def ScaleImg(img):
    #print('zll-img before scaling: ', img.shape, img)
    #print(img)
    for i in range(3):
        channel = img[i]
        #print(i, 'th channel before scaling', channel)
        max = np.max(channel)
        min = np.min(channel)
        scale = 255/(max - min)
        print('channel, scale', i, scale)
        channel = (channel - min)*scale
        #print(i, 'th channel after scaling', channel)
        img[i] = channel.astype(int)

    #print('zll-img after scaling: ', img.shape, img)

    return img

def ConvertTif2GeoTiff(aoi_id, splits = True, min_alt=None, max_alt=None, alt_me=0):
    if aoi_id[:3] == 'Dji':
        #idxs = ['4', '7', '13']
        #idxs = ['8', '12', '13']
        #idxs = ['6', '13', '14']
        #idxs = ['2', '13', '15']
        #idxs = ['2', '13']
        #idxs = ['4', '13']
        #idxs = ['8', '12']

        idxs = ['10', '13']
        idxs = ['10', '14']
        idxs = ['11', '13']
        #idxs = ['8', '12']
        idxs = ['4', '13']
        #idxs = ['2', '13', '15', '6', '10', '9']
        idxs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        idxs = ['2', '13', '15']
        #idxs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
        #idxs = ['2', '13']
        idxs = ['2', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    elif aoi_id[:3] == 'ItA':
        idxs = ['1004', '1005', '1006']
    elif aoi_id[:3] == 'ItB':
        idxs = ['0004', '0005', '0006']
    elif aoi_id[:3] == 'Itl':
        idxs = ['0004', '0005', '0006', '1004', '1005', '1006']

    ImgNb = len(idxs)

    #dfc_dir = '/home/LZhang/Documents/CNESPostDoc/Djibouti/Pleiades/TOC/params11/Converted/'
    if aoi_id[:3] == 'Dji':
        dfc_dir = '/home/LZhang/Documents/CNESPostDoc/Djibouti/Pleiades/TOC/params10/Converted/'
        dsmDir = '/home/LZhang/Documents/CNESPostDoc/Djibouti/4DSMsFromArthur/CroppedDSM/'
        prefix, postfix = 'IMG', '_TOC_Converted'
    elif aoi_id[:3] == 'ItA' or aoi_id[:3] == 'ItB' or aoi_id[:3] == 'Itl':
        dfc_dir = '/home/LZhang/Documents/CNESPostDoc/Italy/Image_RGB/'
        dsmDir = '/home/LZhang/Documents/CNESPostDoc/Italy/DSM_0.5m/'
        prefix, postfix = 'IMG_', '_RGB'
    subdir = aoi_id + '_' + str(ImgNb) + '_imgs'
    output_dir = dfc_dir + subdir + '/'
    print('output_dir: ', output_dir)
    #crops_dir = os.path.join(output_dir, "crops/")
    #os.makedirs(crops_dir, exist_ok=True)
    
    DSMTxtFile = dsmDir + aoi_id + '_DSM.txt'
    DSMFile = dsmDir + aoi_id + '_DSM.tif'
    aoi_lonlat = read_DFC2019_lonlat_aoi(aoi_id, None, DSMTxtFile)

    crops_dir = os.path.join(output_dir, "dataset"+aoi_id+"/")     #subdir+"_NeRF/")
    nerf_dir = crops_dir
    os.makedirs(crops_dir, exist_ok=True)
    crops_dir = os.path.join(crops_dir, aoi_id+"/")
    os.makedirs(crops_dir, exist_ok=True)
    truth_dir = os.path.join(crops_dir, "Truth/")
    os.makedirs(truth_dir, exist_ok=True)
    shutil.copyfile(DSMTxtFile, truth_dir + os.path.basename(DSMTxtFile))
    shutil.copyfile(DSMFile, truth_dir + os.path.basename(DSMFile))

    crops_dir = os.path.join(crops_dir, "RGB-crops/")
    os.makedirs(crops_dir, exist_ok=True)
    crops_dir = os.path.join(crops_dir, aoi_id+"/")
    os.makedirs(crops_dir, exist_ok=True)

    for idx in idxs:
        geotiff_path = dfc_dir + prefix + idx + postfix + '.tif'
        output_path = crops_dir + os.path.basename(geotiff_path)

        if aoi_id[:3] == 'Dji':
            rpc_path = dfc_dir + prefix + idx + postfix + '.xml'
            rpc = rpcm.rpc_from_rpc_file(rpc_path)
        else:
            rpc = rpcm.rpc_from_geotiff(geotiff_path)

        #print('aoi_lonlat', aoi_lonlat)
        #print('geotiff_path, rpc', geotiff_path, rpc)

        #rpcm.utils.crop_aoi will project the origin of bounding box (aoi_lonlat) to image coordinates to get the offset of x, y
        #the new rpc is the same as input with only the offset changed
        crop, x, y = rpcm.utils.crop_aoi(geotiff_path, aoi_lonlat, z=alt_me, rpc=rpc, InputRPC = 1)
        ScaleImg(crop)         #this will change the color consistency, but it is necessary for Djibouti and Italy data
        print('x, y after cropping ', x, y)
        rpc.row_offset -= y     #LINE_OFF
        rpc.col_offset -= x     #SAMP_OFF

        with rasterio.open(geotiff_path, 'r') as src:
            profile = src.profile
            tags = src.tags()

        print('ImgSz before cropping: ', profile["height"], profile["width"])

        not_pan = len(crop.shape) > 2
        if not_pan:
            profile["height"] = crop.shape[1]
            profile["width"] = crop.shape[2]
        else:
            profile["height"] = crop.shape[0]
            profile["width"] = crop.shape[1]
            profile["count"] = 1

        print('ImgSz after cropping: ', profile["height"], profile["width"])
        #print('crop', crop)

        profile['dtype'] = rasterio.uint8

        #save cropped images
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(crop)
            dst.update_tags(**tags)
            dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())

        print('Cropped image saved in ', output_path)

    img_dir = crops_dir

    ba = True
    if ba:
        run_ba(img_dir, output_dir)

    newoutput_dir = create_dataset_from_DFC2019_data_zll(nerf_dir, aoi_id, img_dir, dfc_dir, output_dir, DSMFile, use_ba=ba, min_alt=min_alt, max_alt=max_alt)  #, aoi_id=aoi_id
    generate_img_mask(newoutput_dir, aoi_id)

    # create train and test splits
    if splits:
        json_files = [os.path.basename(p) for p in glob.glob(os.path.join(newoutput_dir, "*.json"))]
        train_samples, test_samples = create_train_test_splits(json_files)
        with open(os.path.join(newoutput_dir, "train.txt"), "w+") as f:
            f.write("\n".join(train_samples))
        with open(os.path.join(newoutput_dir, "test.txt"), "w+") as f:
            f.write("\n".join(test_samples))

    print("done")


def CheckAltBound(path_to_dsm):
    src = rio_open(path_to_dsm)
    dsm = src.read()[0, :, :]
    min_alt = int(np.round(dsm.min() - 1))
    max_alt = int(np.round(dsm.max() + 1))

    print('min_alt, max_alt', min_alt, max_alt)

if __name__ == '__main__':
    if 0:
        path_to_dsm = '/home/LZhang/Documents/CNESPostDoc/Djibouti/Pleiades/TOC/params10/Converted/Dji_012_3_imgs/DenseDepth-SatNeRFBA/MEC-Malt-ObjGeo/Z_Num8_DeZoom1_STD-MALT_Masked.tif'
        path_to_dsm = '/home/LZhang/Documents/CNESPostDoc/Djibouti/Pleiades/TOC/params10/Converted/Dji_012_3_imgs/datasetDji_012/Dji2013/Truth/Dji_012_DSM.tif'
        CheckAltBound(path_to_dsm)

    #aoi_id = 'Dji_005'
    aoi_id = 'Dji_001'

    aoi_id = 'ItA_001'
    aoi_id = 'ItB_001'
    aoi_id = 'Itl_001'

    aoi_id = 'ItA_002'
    #aoi_id = 'ItB_002'
    aoi_id = 'Itl_002'
    aoi_id = 'Dji_021'
    min_alt = None
    max_alt = None
    alt_me = 0
    
    #CropDSM(aoi_id)

    ConvertTif2GeoTiff(aoi_id, min_alt=min_alt, max_alt=max_alt, alt_me=alt_me)

    #newoutput_dir = '/home/LZhang/Documents/CNESPostDoc/Djibouti/Pleiades/TOC/params10/Converted/Dji_012_3_imgs/datasetDji_012/root_dir/crops_rpcs_ba_v2/Dji_012_mask/'
    newoutput_dir = '/home/LZhang/Documents/CNESPostDoc/Djibouti/Pleiades/TOC/params10/Converted/Dji_005_19_imgs/datasetDji_005/root_dir/crops_rpcs_ba_v2/Dji_005_mask/'


    #GenerateImgMask()

'''
cd /home/LZhang/Documents/CNESPostDoc/SatNeRFProj/code/satnerf-master-zll/
conda activate ba
python3 zll-create_satellite_dataset.py

'''
