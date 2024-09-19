import numpy as np
import torch
import train_utils
import BRDF.basic_func as basic_func

def func_M1(ci, cv, k, eps=1e-5):
    tmp = ci*cv*(ci+cv)+eps
    if torch.min(tmp) < 0:
        train_utils.PrintMMM('ci*cv*(ci+cv) in func_M:', tmp)
    tmp1 = tmp
    y = torch.pow(tmp1, k-1)

    wei_total, wei_valid, wei_inval, weights = train_utils.check_badnr(tmp, 'bad nr in func_M1')
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='func_M1')

    return y_

def func_G(sza, vza, phi, eps=1e-5):
    ti = torch.tan(sza)
    tv = torch.tan(vza)
    cp = torch.cos(phi)
    tmp = ti**2 + tv**2 - 2*ti*tv*cp + eps
    y_test = torch.sqrt(tmp)
    y = y_test

    wei_total, wei_valid, wei_inval, weights = train_utils.check_badnr(tmp, 'bad nr in func_G')

    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='func_G')
    return y_

def func_H(rhoc, G, eps=1e-5):
    y = 1+(1-rhoc)/(1+G+eps)
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='func_H')
    return y_

class RPV(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def calc_rpv(self, pts2l, pts2c, normal, w, k, theta, rhoc, mode='train'):
        ci, sza, si, cv, vza, sv, cg, g, phi = basic_func.calc_angles(pts2l, pts2c, normal)

        if k != None:
            M1 = func_M1(ci.unsqueeze(-1), cv.unsqueeze(-1), k)
        else:
            M1 = torch.ones_like(ci.unsqueeze(-1))

        if theta != None:
            F = basic_func.Henyey_Greenstein(cg.unsqueeze(-1), theta)
        else:
            F = torch.ones_like(cg.unsqueeze(-1))
            
        if rhoc != None:
            G = func_G(sza, vza, phi).unsqueeze(-1)
            G_ =  G.detach()
            H = func_H(rhoc, G_)
        else:
            G = torch.ones_like(sza.unsqueeze(-1))
            H = torch.ones_like(sza.unsqueeze(-1))

        brdf = w * M1 * F * H

        return brdf, M1, G, H, ci, cv

    def forward(self, pts2l, pts2c, normal, w, k, theta, rhoc, mode='train'):
        brdf, M1, G, H, ci, cv= self.calc_rpv(pts2l, pts2c, normal, w, k, theta, rhoc, mode)

        return brdf, M1, G, H, ci, cv
