"""
This script defines the NeRF architecture
"""

import numpy as np
import torch
import train_utils

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Siren(torch.nn.Module):
    """
    Siren layer
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


class Mapping(torch.nn.Module):
    def __init__(self, mapping_size, in_size, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = mapping_size
        self.in_channels = in_size
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels*(len(self.funcs)*self.N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(self.N_freqs-1), self.N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        #out = [x]
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                #print('freq, func, x', freq, func, x)
                out += [func(freq*x)]

        return torch.cat(out, -1)

def inference(model, args, rays_xyz, z_vals, rays_d=None):
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
    Returns:
        result: dictionary with the output magnitudes of interest
    """
    N_rays = rays_xyz.shape[0]
    N_samples = rays_xyz.shape[1]
    xyz_ = rays_xyz.view(-1, 3)  # (N_rays*N_samples, 3)

    # check if there are additional inputs, which are used or not depending on the nerf variant
    rays_d_ = None if rays_d is None else torch.repeat_interleave(rays_d, repeats=N_samples, dim=0)

    # the input batch is split in chunks to avoid possible problems with memory usage
    chunk = args.chunk
    batch_size = xyz_.shape[0]

    # run model
    out_chunks = []
    for i in range(0, batch_size, chunk):
        input_dir = None if rays_d_ is None else rays_d_[i:i + chunk]
        out_chunks += [model(xyz_[i:i+chunk], input_dir=input_dir)]
    out = torch.cat(out_chunks, 0)

    # retreive outputs
    out_channels = model.number_of_outputs
    nr_an_on = False
    if (model.normal == 'analystic_learned' or model.normal == 'analystic'): #and mode=='train':
        out_channels += 3 # + normal (3)
        nr_an_on = True    
    out = out.view(N_rays, N_samples, out_channels)
    rgbs = out[..., :3]  # (N_rays, N_samples, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples)

    # define deltas, i.e. the length between the points in which the ray is discretized
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples)

    # compute alpha as in the formula (3) of the nerf paper
    noise_std = args.noise_std
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples)
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
    transparency = torch.cumprod(alphas_shifted, -1)[:, :-1]  # T in the paper
    weights = alphas * transparency # (N_rays, N_samples)
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # return outputs
    depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
    result = {'rgb': rgb_final,
              'depth': depth_final,
              'weights': weights,
              'z_vals': z_vals,
              'sigmas': sigmas.unsqueeze(-1),
              'alphas': alphas,
              'transparency': transparency}

    if nr_an_on == True:
        idx = 4
        normal_an = out[..., idx:idx+3]
        result['normal_an'] = normal_an

    return result

class NeRF(torch.nn.Module):
    def print_parms(self, only_name=False):
        for name, parms in self.named_parameters():
            #print('name, parms, parms.requires_grad, parms.grad: ', name, parms, parms.requires_grad, parms.grad)
            #print(name, 'requires_grad: ', parms.requires_grad)
            str_print = '{} | gra {} | '.format(name, parms.requires_grad)
            if only_name == False:
                train_utils.PrintMMM(str_print, parms.data)
            else:
                print(str_print)

            if parms.grad != None:
                train_utils.PrintMMM('', parms.grad, Print=False)
                #_, _, _, details = train_utils.PrintMMM('', parms.grad, Print=False)
                #print('    gradient: ', details)

    def __init__(self, layers=8, feat=256, mapping=True, mapping_sizes=[10, 4], skips=[4], siren=False, normal='none'):
        super(NeRF, self).__init__()
        self.layers = layers
        self.skips = skips
        self.mapping = mapping
        self.input_sizes = [3, 3]
        self.rgb_padding = 0.001
        self.number_of_outputs = 4
        self.normal = normal

        print('NeRF: layers, feat: ', layers, feat)  #NeRF: layers, feat:  8 512

        # activation function
        nl = Siren() if siren else torch.nn.ReLU()

        # use positional encoding if specified
        in_size = self.input_sizes.copy()
        if mapping:
            self.mapping = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(mapping_sizes, self.input_sizes)]
            in_size = [2 * map_sz * in_sz for map_sz, in_sz in zip(mapping_sizes, self.input_sizes)]
        else:
            self.mapping = [torch.nn.Identity(), torch.nn.Identity()]

        # define the main network of fully connected layers, i.e. FC_NET
        fc_layers = []
        fc_layers.append(torch.nn.Linear(in_size[0], feat))
        fc_layers.append(Siren(w0=30.0) if siren else nl)
        for i in range(1, layers):
            if i in skips:
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
                                                   torch.nn.Linear(feat // 2, 3), torch.nn.Sigmoid())

        if siren:
            self.fc_net.apply(sine_init)
            self.fc_net[0].apply(first_layer_sine_init)

    def freeze(self, layer_name):
        for name, parms in self.named_parameters():
            if (layer_name in name) or layer_name=='all':
                parms.requires_grad = False
                print(name, ' freezed: ')

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

    def forward(self, input_xyz, input_dir=None, sigma_only=False):
        """
        Predicts the values rgb, sigma from a batch of input rays
        the input rays are represented as a set of 3d points xyz

        Args:
            input_xyz: (B, 3) input tensor, with the 3d spatial coordinates, B is batch size
            sigma_only: boolean, infer sigma only if True, otherwise infer both sigma and color

        Returns:
            if sigma_ony:
                sigma: (B, 1) volume density
            else:
                out: (B, 4) first 3 columns are rgb color, last column is volume density
        """

        # compute shared features
        input_xyz = self.mapping[0](input_xyz)
        xyz_ = input_xyz
        for i in range(self.layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = self.fc_net[2*i](xyz_)
            xyz_ = self.fc_net[2*i + 1](xyz_)
        shared_features = xyz_

        # compute volume density
        sigma = self.sigma_from_xyz(shared_features)
        if sigma_only:
            return sigma

        # compute color
        xyz_features = self.feats_from_xyz(shared_features)
        if self.input_sizes[1] > 0:
            input_xyzdir = torch.cat([xyz_features, self.mapping[1](input_dir)], -1)
        else:
            input_xyzdir = xyz_features
        rgb = self.rgb_from_xyzdir(input_xyzdir)
        # improvement suggested by Jon Barron to help stability (same paper as soft+ suggestion)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        out = torch.cat([rgb, sigma], 1) # (B, 4)

        if (self.normal == 'analystic_learned' or self.normal == 'analystic'):
            grad_an = self.calc_normals(input_xyz_)
            normal_an = -train_utils.l2_normalize(grad_an, keyword='nr_an_ori')
            out = torch.cat([out, normal_an], 1)           

        return out
