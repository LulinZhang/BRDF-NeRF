from .satellite import SatelliteDataset
from .satellite_depth import SatelliteDataset_depth
from .blender import BlenderDataset
from .satellite_rgb_dep import SatelliteRGBDEPDataset

def load_dataset(args, split):

    outputs = []
    if args.data == 'sat':
        if args.model == 'sps-nerf' or args.model == 'spsbrdf-nerf':        #for sps-nerf, load rgb and depth at the same time, depth is padded to the same size as rgb
            d1 = SatelliteRGBDEPDataset(args=args, split=split)
            outputs.append(d1)
        elif args.model == 'nerf' or args.model == 's-nerf' or args.model == 'sat-nerf':
            d1 = SatelliteDataset(root_dir=args.root_dir,
                         img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                         split=split,
                         cache_dir=args.cache_dir,
                         img_downscale=args.img_downscale)
            outputs.append(d1)
            if args.ds_lambda > 0 and split == 'train':
                d2 = SatelliteDataset_depth(root_dir=args.root_dir,
                             img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                             split=split,
                             cache_dir=args.cache_dir,
                             img_downscale=args.img_downscale)
                outputs.append(d2)            
    else:
        img_wh=(400, 400)
        outputs.append(BlenderDataset(root_dir=args.root_dir, split=split, img_wh=img_wh))

    return outputs
