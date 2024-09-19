import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T

from kornia import create_meshgrid

def get_sun_dirs(c2w, w, h, sun_d=torch.FloatTensor([0,0,1])):
    sun_dirs = torch.from_numpy(np.tile(sun_d, (w*h, 1))).reshape(w, h, 3)
    sun_dirs = sun_dirs @ c2w[:, :3].T # (H, W, 3)
    sun_dirs = sun_dirs / torch.norm(sun_dirs, dim=-1, keepdim=True)

    sun_dirs = sun_dirs.view(-1, 3)
    sun_dirs = sun_dirs.type(torch.FloatTensor)
    return sun_dirs

def get_rows_cols(w, h, rays):
    all_rows, all_cols = [], []

    cols, rows = np.meshgrid(np.arange(w), np.arange(h))

    all_rows += [torch.from_numpy(rows).reshape(rays.shape[0], 1)]
    all_cols += [torch.from_numpy(cols).reshape(rays.shape[0], 1)]

    all_rows = torch.cat(all_rows, 0)
    all_cols = torch.cat(all_cols, 0)

    return all_rows, all_cols

def get_ray_directions(H, W, K):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    print("get_ray_directions, H, W, K: ", H, W, K)
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def add_perturbation(img, perturbation, seed):
    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img) / 255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s * img_np[..., :3] + b, 0, 1)
        img = Image.fromarray((255 * img_np).astype(np.uint8))
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10 * seed + i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left + 20 * i, top), (left + 20 * (i + 1), top + 200)),
                           fill=random_color)
    return img


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(400, 400),
                 perturbation=[]):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        assert set(perturbation).issubset({"color", "occ"}), \
            'Only "color" and "occ" perturbations are supported!'
        self.perturbation = perturbation
        if self.split == 'train':
            print(f'add {self.perturbation} perturbation!')
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        json_file = f"transforms_{self.split.split('_')[-1]}.json"
        with open(os.path.join(self.root_dir, json_file), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length

        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w / 2
        self.K[1, 2] = h / 2

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.K)  # (h, w, 3)

        max_trai_nimg = 100
        print('--->>>load data: ', self.split, json_file)
        for t, frame in enumerate(self.meta['frames']):
            print(t, f"{frame['file_path']}.png")
            if t > max_trai_nimg:
                print('training data after 10 images are ignored to speed up debugging')
                break

        if self.split == 'train':  # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            print('load data: ', self.split, json_file)
            for t, frame in enumerate(self.meta['frames']):
                if t > max_trai_nimg:
                    break
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(image_path)
                if t != 0:  # perturb everything except the first image.
                    img = add_perturbation(img, self.perturbation, t)

                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near * torch.ones_like(rays_o[:, :1]),
                                             self.far * torch.ones_like(rays_o[:, :1]),
                                             #sun_dirs,
                                             rays_t],
                                            1)]  # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately
            w, h = self.img_wh
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            t = 0  # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            if self.split == 'test_train' and idx != 0:
                t = idx
                img = add_perturbation(img, self.perturbation, idx)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, H, W)
            valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)
            sun_dirs = get_sun_dirs(c2w, w, h)

            rays = torch.cat([rays_o, rays_d,
                              self.near * torch.ones_like(rays_o[:, :1]),
                              self.far * torch.ones_like(rays_o[:, :1])],
                              1)  # (H*W, 8)
            
            all_rows, all_cols = get_rows_cols(self.img_wh[0], self.img_wh[1], rays)
            
            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'src_id': frame['file_path'],
                      'w': self.img_wh[0],
                      'h': self.img_wh[1],
                      'rows': all_rows,
                      'cols': all_cols,
                      'save_cross': False,
                      'idx': idx,
                      'valid_mask': valid_mask}

            if idx == 0:
                sample['save_cross'] = True

            if self.split == 'test_train' and self.perturbation:
                # append the original (unperturbed) image
                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, H, W)
                valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
                img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                sample['original_rgbs'] = img
                sample['original_valid_mask'] = valid_mask

        return sample

