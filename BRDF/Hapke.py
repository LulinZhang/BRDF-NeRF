import numpy as np
import torch
import train_utils 
import BRDF.basic_func as basic_func

def E1(x, theta, eps=1e-5):
    y = torch.exp(- (2. / np.pi) * 1. / torch.tan(theta + eps) * 1. / torch.tan(x + eps))
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='E1')
    return y_

def E2(x, theta, eps=1e-5):
    y = torch.exp(- (1. / np.pi) * (1. / torch.tan(theta + eps))**2 * (1. / torch.tan(x + eps))**2)
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='E2')
    return y_

def f(phi, eps=1e-5):
    y = torch.exp(-2. * torch.tan((phi + eps) / 2))
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='f')
    return y_

def chi(x, eps=1e-5):
    y = 1. / torch.sqrt(1. + np.pi * torch.tan(x + eps)**2)
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='chi')
    return y_
    
def eta(x, theta, eps=1e-5):
    chi_theta = chi(theta)
    y = chi_theta * (torch.cos(x) + torch.sin(x) * torch.tan(theta + eps) * (E2(x, theta) / (2 - E1(x, theta))))
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='eta')
    return y_

def mu0_eff(i, e, phi, theta):
    y = torch.zeros(e.shape, device=e.device)

    n1 = torch.where(i <= e)
    n2 = torch.where(i > e)

    y[n1] = torch.cos(phi[n1]) * E2(e[n1], theta[n1]) + torch.sin(phi[n1] / 2)**2 * E2(i[n1], theta[n1])
    y[n1] = y[n1] / (2 - E1(e[n1], theta[n1]) - phi[n1] / np.pi * E1(i[n1], theta[n1]))
    y[n1] = chi(theta[n1]) * (torch.cos(i[n1]) + torch.sin(i[n1]) * torch.tan(theta[n1]) * y[n1])

    y[n2] = E2(i[n2], theta[n2]) - torch.sin(phi[n2] / 2)**2 * E2(e[n2], theta[n2])
    y[n2] = y[n2] / (2 - E1(i[n2], theta[n2]) - phi[n2] / np.pi * E1(e[n2], theta[n2]))
    y[n2] = chi(theta[n2]) * (torch.cos(i[n2]) + torch.sin(i[n2]) * torch.tan(theta[n2]) * y[n2])

    y_, _ = train_utils.check_nan(y, val_rep=torch.cos(i), func='mu0_eff')

    return y_

def mu_eff(i, e, phi, theta):
    y = torch.zeros(e.shape, device=e.device)

    n1 = torch.where(i <= e)
    n2 = torch.where(i > e)

    y[n1] = E2(e[n1], theta[n1]) - (torch.sin(phi[n1] / 2))**2 * E2(i[n1], theta[n1])
    y[n1] = y[n1] / (2 - E1(e[n1], theta[n1]) - (phi[n1] / np.pi) * E1(i[n1], theta[n1]))
    y[n1] = chi(theta[n1]) * (torch.cos(e[n1]) + torch.sin(e[n1]) * torch.tan(theta[n1]) * y[n1])

    y[n2] = torch.cos(phi[n2]) * E2(i[n2], theta[n2]) + (torch.sin(phi[n2] / 2))**2 * E2(e[n2], theta[n2])
    y[n2] = y[n2] / (2 - E1(i[n2], theta[n2]) - (phi[n2] / np.pi) * E1(e[n2], theta[n2]))
    y[n2] = chi(theta[n2]) * (torch.cos(e[n2]) + torch.sin(e[n2]) * torch.tan(theta[n2]) * y[n2])

    y_, _ = train_utils.check_nan(y, val_rep=torch.cos(e), func='mu_eff')

    return y_

def S(i, e, phi, theta):
    theta_p = theta  # No transformation needed in Python

    ci = torch.cos(i)
    cv = torch.cos(e)

    mue = mu_eff(i, e, phi, theta_p)
    etai = eta(i, theta_p)
    etae = eta(e, theta_p)
    chit = chi(theta_p)

    f_func = f(phi)

    y = torch.zeros(e.shape, device=e.device)

    n1 = torch.where(i <= e)
    n2 = torch.where(i > e)

    temp = (mue / etae) * (ci / etai) * chit
    y[n1] = temp[n1] / (1 - f_func[n1] + f_func[n1] * chit[n1] * (ci[n1] / etai[n1]))
    y[n2] = temp[n2] / (1 - f_func[n2] + f_func[n2] * chit[n2] * (cv[n2] / etae[n2]))
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='S')

    return y_

def PF(x, b, c):
    # Henyey-Greenstein double phase function
    # Hapke 2017, p105; range [-1,1]
    # Schmitt et al. (2015), range [0,1]
    '''
    x: cos(phase_angle), Nx1
    b: Nx3
    c: Nx3
    '''
    one = torch.ones_like(b)
    b_2 = torch.pow(b, 2)
    bx = b * x
    
    numerator = c * (one - b_2)
    denominator = torch.pow(one - 2*bx + b_2, 1.5) + 1e-6
    y = numerator / denominator
   
    numerator = (one - c) * (one - b_2)
    denominator = torch.pow(one + 2*bx + b_2, 1.5) + 1e-6
    y += numerator / denominator
    y_, _ = train_utils.check_nan(y, val_rep=torch.zeros_like(y), func='PF')
    
    return y_

def HF(light, camera, normal, x, w, mode='train', func='view'):
    # Ambartsumian–Chandrasekhar H function (Hapke 2002)
    gamma = torch.sqrt(1 - w)
    ro = (1 - gamma) / (1 + gamma)
    tmp = (1 + x) / x
    log = torch.log(torch.abs(tmp))     #tobemodified zll: replace abs

    tmp1 = 1 - w * x * (ro + (1 - 2 * ro * x) / 2 * log)
    y_tmp = torch.pow(tmp1, -1)

    wei_total, wei_valid, wei_inval, _ = train_utils.check_badnr(tmp, 'bad nr in HF', Print=False)
    y = y_tmp

    y_, _ = train_utils.check_nan(y, val_rep=torch.ones_like(y), func='HF')
    return y_, wei_inval

class Hapke(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.hpk_scl = args.hpk_scl
        self.shell_hapke = args.shell_hapke

    def hapkeHG_6var(self, pts2l, pts2c, normal, w, b=None, c=None, theta=None, h=None, B0=None, mode='train'):
        '''
        pts2l: NxLx3  (L=1)
        pts2c: Nx3
        normal: Nx3
        w (albedo): Nx3
        b: Nx3
        '''
        ci, sza, si, cv, vza, sv, cg, g, phi = basic_func.calc_angles(pts2l, pts2c, normal)

        # Phase function P
        if b == None:
            P = torch.tile(torch.ones_like(cg).unsqueeze(-1), (1, 3))
        else:
            if c == None:
                P = basic_func.Henyey_Greenstein(cg.unsqueeze(-1), b)
            else:
                P = PF(cg.unsqueeze(-1), b, c)

        # Backscattering function B
        if B0 != None and h != None:
            B = B0 / (1 + 1 / h * torch.tan(g / 2)) + 1
        else:
            B = torch.ones_like(g).unsqueeze(-1)

        # Calculate effective cosines of incidence and viewing angles
        if theta != None:
            ci = mu0_eff(sza, vza, phi, theta)
            cv = mu_eff(sza, vza, phi, theta)
            show_val = False
            ShadFunc = S(sza, vza, phi, theta).unsqueeze(-1)
        else:
            ShadFunc = torch.ones_like(sza).unsqueeze(-1)

        # Chandrasekhar function H
        if 1:
            Hi, bad_nr_sun = HF(pts2l.squeeze(), pts2c, normal, ci.unsqueeze(-1), w, func='sun')     #ci: light & normal
            Hv, bad_nr_vw = HF(pts2l.squeeze(), pts2c, normal, cv.unsqueeze(-1), w, func='view')    #cv: view & normal
            if bad_nr_sun > 0 or bad_nr_vw > 0:
                print('--HF bad nr: {:.0f} | {:.0f} / {:.0f}'.format(bad_nr_sun, bad_nr_vw, torch.ones_like(ci.unsqueeze(-1)).sum()))
        else:
            Hi = torch.tile(torch.ones_like(ci).unsqueeze(-1), (1, 3))
            Hv = torch.tile(torch.ones_like(cv).unsqueeze(-1), (1, 3))            

        if b == None:       #shell_hapke
            if self.shell_hapke == 1:
                brdf = w / self.hpk_scl
            elif self.shell_hapke == 2:    #original
                scl = (ci + cv) * self.hpk_scl + 1e-6
                brdf = w / scl.unsqueeze(-1)
            elif self.shell_hapke == 3:
                scl = (ci + cv) * self.hpk_scl + 1e-6
                brdf = w * (Hi * Hv) / scl.unsqueeze(-1)
        else:
            # Bidirectional Reflectance Factor
            tmp1 = (ci / (ci + cv) / torch.cos(sza)).unsqueeze(-1)
            tmp2 = P * B + Hi * Hv - torch.ones_like(P)

            BRF = w / self.hpk_scl * tmp1 * tmp2 * ShadFunc
            brdf = BRF      #/np.pi

        return brdf, P, B, Hi, Hv, ShadFunc, ci, cv

    def forward(self, pts2l, pts2c, normal, w, b=None, c=None, theta=None, h=None, B0=None, mode='train'):
        brdf = self.hapkeHG_6var(pts2l, pts2c, normal, w, b, c, theta, h, B0, mode)

        return brdf

