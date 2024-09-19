from .nerf import *
from .satnerf import *
from .snerf import *
from .spsbrdfnerf import *

def load_model(args):
    if args.model == "nerf":
        model = NeRF(layers=args.fc_layers, feat=args.fc_feat, normal=args.normal)
    elif args.model == "s-nerf":
        model = ShadowNeRF(layers=args.fc_layers, mapping=args.mapping, feat=args.fc_feat, normal=args.normal)
    elif args.model == "sat-nerf" or args.model == "sps-nerf":
        model = SatNeRF(layers=args.fc_layers, mapping=args.mapping, feat=args.fc_feat, t_embedding_dims=args.t_embbeding_tau, beta=args.beta)
    elif args.model == "spsbrdf-nerf":
        model = SpSBRDFNeRF(args, layers=args.fc_layers, mapping=args.mapping, feat=args.fc_feat, t_embedding_dims=args.t_embbeding_tau, beta=args.beta, roughness=args.roughness, normal=args.normal, indirect_light=args.indirect_light, glossy_scale=args.glossy_scale, sun_v=args.sun_v, MultiBRDF=args.MultiBRDF, dim_RPV=args.dim_RPV, siren=args.siren)
    else:
        raise ValueError(f'model {args.model} is not valid')
    return model
