import numpy as np
import torch
import train_utils

def calc_angles(pts2l, pts2c, normal, eps=1e-5):
    if 1:
        Print = False
        train_utils.check_vec0('sun_dir ', pts2l.squeeze(), Print=Print)
        train_utils.check_vec0('view_dir', pts2c, Print=Print)
        train_utils.check_vec0('normal  ', normal, Print=Print)

    cos_min = eps
    ci = torch.einsum('ij,ij->i', pts2l.squeeze(), normal)
    ci = torch.clamp(ci, min=cos_min, max=1.0)
    sza = torch.acos(ci)
    si = torch.sin(sza)
    
    cv = torch.einsum('ij,ij->i', pts2c, normal)
    cv = torch.clamp(cv, min=cos_min, max=1.0)
    vza = torch.acos(cv)
    sv = torch.sin(vza)

    cg = torch.einsum('ij,ij->i', pts2c, pts2l.squeeze())
    cg = torch.clamp(cg, min=-1.0, max=1.0)
    g = torch.acos(cg)

    cp = (cg-ci*cv)/si/sv
    cp = torch.clamp(cp, min=-1.0, max=1.0)
    phi = torch.acos(cp)

    return ci, sza, si, cv, vza, sv, cg, g, phi

def Henyey_Greenstein(x, theta, eps=1e-6):
    # Henyey-Greenstein one phase function
    '''
    x: cos(phase_angle), Nx1
    theta: Nx3
    '''
    one = torch.ones_like(theta)
    theta_2 = torch.pow(theta, 2)

    y = (one - theta_2) / (torch.pow(one + 2 * theta * x + theta_2, 1.5) + eps)
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='Henyey_Greenstein')
    return y_
