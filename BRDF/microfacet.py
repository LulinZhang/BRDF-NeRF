import numpy as np
import torch

def safe_l2_normalize(x, axis=None, eps=1e-6):
    return torch.nn.functional.normalize(x, p=2, dim=axis, eps=eps)

class Microfacet(torch.nn.Module):
    """As described in:
        Microfacet Models for Refraction through Rough Surfaces [EGSR '07]
    """
    def __init__(self, default_rough=0.3, lambert_only=False, f0=0.04, lvis=True, glossy_scale=1., print_debuginfo=False):  #0.91):
        super().__init__()
        self.default_rough = default_rough
        self.lambert_only = lambert_only
        self.f0 = f0
        self.lvis = lvis  #if True, calculate light visibility in microfacet BRDF Geometry function
        self.glossy_scale = glossy_scale
        self.print_debuginfo = print_debuginfo

    def forward(self, pts2l, pts2c, normal, albedo=None, rough=None, mode='train'):
        """All in the world coordinates.

        Too low roughness is OK in the forward pass, but may be numerically
        unstable in the backward pass

        pts2l: NxLx3
        pts2c: Nx3
        normal: Nx3
        albedo: Nx3
        rough: Nx1
        """
        if albedo is None:
            albedo = torch.ones(pts2c.shape[0], 3).type(torch.FloatTensor)
        if rough is None:
            rough = self.default_rough * torch.ones(pts2c.shape[0], 1).type(torch.FloatTensor)
        # Normalize directions and normals
        pts2l = safe_l2_normalize(pts2l, axis=2)
        pts2c = safe_l2_normalize(pts2c, axis=1)
        normal = safe_l2_normalize(normal, axis=1)
        # Glossy
        h = pts2l + pts2c.unsqueeze(1) # NxLx3
        h = safe_l2_normalize(h, axis=2)
        f = self._get_f(pts2l, h) # NxL
        alpha = rough ** 2
        d, n_h = self._get_d(h, normal, alpha=alpha) # NxL
        g = self._get_g(pts2c, h, normal, alpha=alpha)
        if self.lvis == True:
            g =g*self._get_g(pts2l.squeeze(), h, normal, alpha=alpha) # NxL
            print('consider light visibility within microfacet BRDF.')
        l_dot_n = torch.einsum('ijk,ik->ij', pts2l, normal)  #i:327680; j:1; k:3
        v_dot_n = torch.einsum('ij,ij->i', pts2c, normal)  #i:327680; j:3
        min_angle = 0.001
        l_dot_n = torch.clamp(torch.abs(l_dot_n), min=min_angle)
        v_dot_n = torch.clamp(torch.abs(v_dot_n), min=min_angle)
        denom = 4 * l_dot_n * v_dot_n.unsqueeze(1)
        factor = 0.04
        microfacet = torch.nan_to_num(torch.div(factor * d, denom)) # NxL, replace nan with 0
        assert torch.isnan(microfacet).sum() == 0, print('microfacet contain nan: ', microfacet)
        brdf_glossy = torch.tile(microfacet.unsqueeze(2), (1, 1, 3)) # NxLx3
        ks = torch.tile(f.unsqueeze(2), (1, 1, 3)) # NxLx3
        kd = torch.ones_like(ks) - ks
        # Diffuse
        lambert = albedo #/ np.pi # Nx3
        brdf_diffuse = torch.broadcast_to(lambert.unsqueeze(1), brdf_glossy.shape) # NxLx3
        # Mix two shaders
        if self.lambert_only:
            brdf = brdf_diffuse
        else:
            brdf = brdf_diffuse + brdf_glossy

        # NxL and NxLx3
        return brdf_glossy[:,:,0], brdf, f, g, d, l_dot_n, v_dot_n, h, n_h
 
    @staticmethod
    def _get_g(v, m, n, alpha=0.1):
        """Geometric function (GGX).
        v: view direction
        m: microsurface normal
        n: macrosurface normal
        """
        cos_theta_v = torch.einsum('ij,ij->i', n, v)
        cos_theta = torch.einsum('ijk,ik->ij', m, v)
        denom = cos_theta_v.unsqueeze(1)
        div = torch.nan_to_num(torch.div(cos_theta, denom))
        assert torch.isnan(div).sum() == 0, print('div contain nan: ', div)
        chi = torch.where(div > 0, 1., 0.)
        cos_theta_v_sq = torch.square(cos_theta_v)
        cos_theta_v_sq = torch.clamp(cos_theta_v_sq, min=0., max=1.)
        denom = cos_theta_v_sq
        tan_theta_v_sq = torch.nan_to_num(torch.div(1 - cos_theta_v_sq, denom))
        assert torch.isnan(tan_theta_v_sq).sum() == 0, print('tan_theta_v_sq contain nan: ', tan_theta_v_sq)
        tan_theta_v_sq = torch.nan_to_num(torch.clamp(tan_theta_v_sq, min=0., max=np.inf))  #transform inf to number
        denom = 1 + torch.sqrt(1 + alpha ** 2 * tan_theta_v_sq.unsqueeze(1))
        g = torch.nan_to_num(torch.div(chi * 2, denom))
        assert torch.isnan(g).sum() == 0, print('g contain nan: ', g)
        return g # (n_pts, n_lights)

    @staticmethod
    def _get_d(m, n, alpha=0.1):
        """Microfacet distribution (GGX).
        """
        cos_theta_m = torch.einsum('ijk,ik->ij', m, n)
        chi = torch.where(cos_theta_m > 0, 1., 0.)
        cos_theta_m_sq = torch.square(cos_theta_m)
        denom = cos_theta_m_sq
        tan_theta_m_sq = torch.nan_to_num(torch.div(1 - cos_theta_m_sq, denom))
        assert torch.isnan(tan_theta_m_sq).sum() == 0, print('tan_theta_m_sq contain nan: ', tan_theta_m_sq)
        denom = np.pi * torch.square(cos_theta_m_sq) * torch.square(alpha ** 2 + tan_theta_m_sq)
        d = torch.nan_to_num(torch.div(alpha ** 2 * chi, denom))
        assert torch.isnan(d).sum() == 0, print('d contain nan: ', d)
        return d, cos_theta_m # (n_pts, n_lights)

    def _get_f(self, l, m):
        """Fresnel (Schlick's approximation).
        """
        cos_theta = torch.einsum('ijk,ijk->ij', l, m)
        f = self.f0 + (1 - self.f0) * (1 - cos_theta) ** 5
        return f # (n_pts, n_lights)

