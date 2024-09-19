"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import datetime
import json
import os

def Test_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", type=str, default='',
                        help='exp_name when training SpS-NeRF')
    parser.add_argument("--logs_dir", type=str, default=None,
                        help='logs_dir when training SpS-NeRF')
    parser.add_argument("--output_dir", type=str, default=None,
                        help='directory to save the output')
    parser.add_argument("--epoch_number", type=int, default=28,
                        help='epoch_number when training SpS-NeRF')
    parser.add_argument("--split", type=str, default='val',
                        help='None')
    parser.add_argument('--infile_postfix', type=str, default="",
                        help='infile_postfix')

    args = parser.parse_args()
    args.infile_postfix += ".txt"

    return args

def printArgs(args):
    print('--------------------------Start printArgs--------------------------')
    print('--n_samples: ', args.n_samples)
    print('--guided_samples: ', args.guided_samples)
    print('--img_downscale: ', args.img_downscale)
    print('--scale: ', args.scale)
    print('--visu_scale: ', args.visu_scale)
    print('--cos_irra_on: ', args.cos_irra_on)
    print('--brdf_on: ', args.brdf_on)
    print('--gsam_only_on: ', args.gsam_only_on)
    print('--nrrg_on: ', args.nrrg_on)
    print('--TestNormal: ', args.TestNormal)
    print('--TestSun_v: ', args.TestSun_v)
    print('--lr: ', args.lr)
    print('--aoi_id: ', args.aoi_id)
    print('--beta: ', args.beta)
    print('--sc_lambda: ', args.sc_lambda)
    print('--lambda_rgb: ', args.lambda_rgb)
    print('--mapping: ', args.mapping)
    print('--inputdds: ', args.inputdds)
    print('--ds_lambda: ', args.ds_lambda)
    print('--ds_drop: ', args.ds_drop)
    print('--GNLL: ', args.GNLL)
    print('--usealldepth: ', args.usealldepth)
    print('--margin: ', args.margin)
    print('--stdscale: ', args.stdscale)
    print('--corrscale: ', args.corrscale)
    print('--model: ', args.model)
    print('--exp_name: ', args.exp_name)
    print('--n_importance: ', args.n_importance)
    print('--roughness: ', args.roughness)
    print('--normal: ', args.normal)
    print('--sun_v: ', args.sun_v)
    print('--nr_reg_an_lambda: ', args.nr_reg_an_lambda)
    print('--nr_reg_lr_lambda: ', args.nr_reg_lr_lambda)
    print('--nr_spv_lambda: ', args.nr_spv_lambda)
    print('--nr_spv_type: ', args.nr_spv_type)
    print('--hs_lambda: ', args.hs_lambda)
    print('--indirect_light: ', args.indirect_light)
    print('--glossy_scale: ', args.glossy_scale)
    print('--in_ckpts: ', args.in_ckpts)
    print('--print_debuginfo: ', args.print_debuginfo)
    print('--cs: ', args.cs)
    print('--pretrain_normal: ', args.pretrain_normal)
    print('--std_range: ', args.std_range)
    print('--toyBRDF: ', args.toyBRDF)
    print('--fresnel_f0: ', args.fresnel_f0)
    print('--infile_postfix: ', args.infile_postfix)
    print('--data: ', args.data)
    print('--MultiBRDF: ', args.MultiBRDF)
    print('--shell_hapke: ', args.shell_hapke)
    print('--hpk_scl: ', args.hpk_scl)
    print('--b: ', args.b)
    print('--c: ', args.c)
    print('--B0: ', args.B0)
    print('--h: ', args.h)
    print('--theta: ',args.theta) 
    print('--save_first_n_visu: ',args.save_first_n_visu)
    print('--funcM:', args.funcM)
    print('--funcF:', args.funcF)
    print('--funcH:', args.funcH)
    print('--input_viewdir: ', args.input_viewdir)
    print('--eval: ', args.eval)
    print('--mod_alt_bound: ', args.mod_alt_bound)
    print('------------------------------')
    print('--root_dir: ', args.root_dir)
    print('--img_dir: ', args.img_dir)
    print('--ckpts_dir: ', args.ckpts_dir)
    print('--logs_dir: ', args.logs_dir)
    print('--gt_dir: ', args.gt_dir)
    print('--cache_dir: ', args.cache_dir)
    print('--ckpt_path: ', args.ckpt_path)
    print('--gpu_id: ', args.gpu_id)
    print('--batch_size: ', args.batch_size)
    print('--max_train_steps: ', args.max_train_steps)
    print('--save_visu_every_n_epochs: ', args.save_visu_every_n_epochs)
    print('--save_file_every_n_epochs: ', args.save_file_every_n_epochs)
    print('--save_ckpt_every_n_epochs: ', args.save_ckpt_every_n_epochs)
    try:
        print('--eval_every_n_epochs: ', args.eval_every_n_epochs)
    except:
        pass
    print('--fc_feat: ', args.fc_feat)
    print('--fc_layers: ', args.fc_layers)
    print('--fc_feat_ref: ', args.fc_feat_ref)
    print('--fc_layers_ref: ', args.fc_layers_ref)
    print('--siren: ', args.siren)
    print('--noise_std: ', args.noise_std)
    print('--chunk: ', args.chunk)
    print('--ds_noweights: ', args.ds_noweights)
    print('--first_beta_epoch: ', args.first_beta_epoch)
    print('--t_embbeding_tau: ', args.t_embbeding_tau)
    print('--t_embbeding_vocab: ', args.t_embbeding_vocab)
    print('--------------------------End printArgs--------------------------')

def Train_parser():
    parser = argparse.ArgumentParser()

    # input paths
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of the input dataset')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory where the images are located (if different than root_dir)')
    parser.add_argument("--ckpts_dir", type=str, default="ckpts",
                        help="output directory to save trained models")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="output directory to save experiment logs")
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='directory where the ground truth DSM is located (if available)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='directory where cache for the current dataset is found')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="pretrained checkpoint path to load")

    # other basic stuff and dataset options
    parser.add_argument("--exp_name", type=str, default=None,
                        help="experiment name")
    parser.add_argument('--data', type=str, default='sat', choices=['sat', 'blender'],
                        help='type of dataset')
    parser.add_argument("--model", type=str, default='sps-nerf', choices=['nerf', 's-nerf', 'sat-nerf', 'sps-nerf', 'spsbrdf-nerf'],
                        help="which NeRF to use")
    parser.add_argument("--gpu_id", type=int, required=True,
                        help="GPU that will be used")

    # training and network configuration
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size (number of input rays per iteration)')
    parser.add_argument('--img_downscale', type=float, default=1.0,
                        help='downscale factor for the input images')
    parser.add_argument('--max_train_steps', type=int, default=300000,
                        help='number of training iterations')
    parser.add_argument('--save_visu_every_n_epochs', type=int, default=1,
                        help="save visualization images every n epochs")
    parser.add_argument('--save_file_every_n_epochs', type=int, default=-1,
                        help="save checkpoints and debug files every n epochs")
    parser.add_argument('--save_ckpt_every_n_epochs', type=int, default=5,
                        help="save checkpoints every n epochs")
    parser.add_argument('--eval_every_n_epochs', type=int, default=4,
                        help="evaluate model every n epochs")
    parser.add_argument('--fc_feat', type=int, default=512,
                        help='number of fully connected units in the main block of layers')
    parser.add_argument('--fc_layers', type=int, default=8,
                        help='number of fully connected layers in the main block of layers')
    parser.add_argument('--n_samples', type=int, default=64,
                        help='number of coarse scale discrete points per input ray')
    parser.add_argument('--n_importance', type=int, default=0,
                        help='number of fine scale discrete points per input ray')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='standard deviation of noise added to sigma to regularize')
    parser.add_argument('--chunk', type=int, default=1024*5,
                        help='maximum number of rays that can be processed at once without memory issues')

    # other sat-nerf specific stuff
    parser.add_argument('--lambda_rgb', type=float, default=1.,
                        help='')
    parser.add_argument('--sc_lambda', type=float, default=0.,
                        help='float that multiplies the solar correction auxiliary loss')
    parser.add_argument('--ds_lambda', type=float, default=0.,
                        help='float that multiplies the depth supervision auxiliary loss')
    #progress para
    parser.add_argument('--ds_drop', type=float, default=1.,
                        help='portion of training steps at which the depth supervision loss will be dropped, 0-1')
    parser.add_argument('--ds_noweights', action='store_true',
                        help='do not use reprojection errors to weight depth supervision loss')
    parser.add_argument('--first_beta_epoch', type=int, default=2,
                        help='')
    parser.add_argument('--t_embbeding_tau', type=int, default=4,
                        help='')
    parser.add_argument('--t_embbeding_vocab', type=int, default=30,
                        help='')

    #SpS-NeRF add-on
    parser.add_argument('--aoi_id', type=str, default="JAX_068",
                        help='aoi_id')
    parser.add_argument('--inputdds', type=str, default="DenseDepth_ZM4",
                        help='the folder to the dense depth files')
    parser.add_argument('--beta', action='store_true',  #Recommendation for SpS-NeRF: NOT present in the command-line argument
                        help='by default, do not use beta for transient uncertainty')
    parser.add_argument('--mapping', action='store_true',    #Recommendation for SpS-NeRF: present in the command-line argument
                        help='by default, do not use positional encoding')   
    parser.add_argument('--GNLL', action='store_true',    #Recommendation for SpS-NeRF: NOT present in the command-line argument
                        help='by default, use MSE depth loss instead of Gaussian negative log likelihood loss')    
    parser.add_argument('--usealldepth', action='store_true',    #Recommendation for SpS-NeRF: NOT present in the command-line argument
                        help='by default, use only a subset of depth which meets the condition of R_sub in equation 6 in SpS-NeRF article')    
    parser.add_argument('--guided_samples', type=int, default=64,
                        help='number of guided discrete points per input ray')
    parser.add_argument('--margin', type=float, default=0.0001,
                        help='so that the pts with correlation scores equal to 1 has the std value of margin, instead of 0. (m in equation 5 in SpS-NeRF article)')
    parser.add_argument('--stdscale', type=float, default=1,
                        help='so that the pts with correlation scores close to 0 has the std value of stdscale, instead of 1. (gama in equation 5 in SpS-NeRF article)')
    parser.add_argument('--corrscale', type=float, default=1,
                        help='scale the correlation for dense depth from different resolution (1 for ZM=4, 0.7 for ZM=8)')   #not used
    parser.add_argument('--siren', type=int, default=1,
                        help='sin activation function instead of ReLU')
    ################BRDF################
    parser.add_argument('--indirect_light', action='store_true',  #Recommendation for SpSBRDF-NeRF: not present in the command-line argument
                        help='by default, do not use indirect_light for BRDF estimation')
    parser.add_argument("--normal", type=str, default='none', choices=['none', 'analystic', 'learned', 'analystic_learned'])
    parser.add_argument("--sun_v", type=str, default='none', choices=['none', 'analystic', 'learned'])
    parser.add_argument('--nr_reg_an_lambda', type=float, default=0.,
                        help='float that multiplies the normal regularization (on normal_an) auxiliary loss')
    parser.add_argument('--nr_reg_lr_lambda', type=float, default=0.,
                        help='float that multiplies the normal regularization (on normal_lr) auxiliary loss')
    parser.add_argument('--nr_spv_lambda', type=float, default=0.,
                        help='float that multiplies the normal supervision auxiliary loss applied on the learned normal to match analystic normal')
    parser.add_argument('--nr_spv_type', type=int, default=0,
                        help='1: use nr_an to supervise nr_lr; 2: use nr_sgm to supervise nr_lr; 3: use nr_sgm to supervise nr_an')
    parser.add_argument('--hs_lambda', type=float, default=0.,
                        help='float that multiplies the hard surface regularization auxiliary loss')
    parser.add_argument('--brdf_on', type=float, default=0.,
                        help='portion of training steps at which the BRDF will be turned on, 0-1')
    parser.add_argument('--nrrg_on', type=float, default=0.,
                        help='portion of training steps at which the normal regularization will be turned on, 0-1')
    parser.add_argument('--TestNormal', type=int, default=0, choices=[0, 1])
    parser.add_argument('--TestSun_v', type=int, default=0, choices=[0, 1])
    parser.add_argument('--in_ckpts', type=str, default="none",
                        help='in_ckpts')
    parser.add_argument('--print_debuginfo', action='store_true', 
                        help='by default, do not print debug information')
    parser.add_argument("--cs", type=str, default='utm', choices=['ecef', 'utm'],
                        help='coordinate system')
    #progress para
    parser.add_argument('--gsam_only_on', type=float, default=1,
                        help='gsam_only_on, 0-1')
    parser.add_argument('--cos_irra_on', type=float, default=1.,
                        help='cos_irra_on, portion of training steps at which the cos(light, normal) will be multiplied to irradiance, 0-1')
    parser.add_argument('--std_range', type=float, default=3.0,
                        help='std_range')
    parser.add_argument('--MultiBRDF', type=int, default=0,
                        help='calculate one BRDF for each samples along the ray. If not, calculate one BRDF for each ray')
    parser.add_argument('--infile_postfix', type=str, default="",
                        help='infile_postfix')
    parser.add_argument('--scale', type=float, default=1/255.,
                        help='scale image pixel value')
    parser.add_argument('--visu_scale', type=float, default=1.,
                        help='visu_scale')

    ###################Microfacet BRDF model####################
    parser.add_argument('--roughness', action='store_true',                 #only valide for Microfacet model 
                        help='by default, do not use roughness for BRDF estimation')
    parser.add_argument('--glossy_scale', type=float, default=1,            #only valide for Microfacet model
                        help='scale the glossy part of the microfacet BRDF')
    parser.add_argument('--pretrain_normal', action='store_true',           #only valide for Microfacet model
                        help='pretrain normal network based on the analystic normal')
    parser.add_argument('--toyBRDF', action='store_true',                   #only valide for Microfacet model
                        help='manually set normal and roughness for test')
    parser.add_argument('--fresnel_f0', type=float, default=0.04,           #only valide for Microfacet model
                        help='fresnel_f0 factor in microfacet BRDF')

    ###################Hapke BRDF model####################
    parser.add_argument('--hpk_scl', type=float, default=4.0,           #only valide for Hapke model
                        help='denominator in Hapke model')
    parser.add_argument('--shell_hapke', type=int, default=0,
                        help='Hapke model without any subfunction')
    parser.add_argument('--b', type=int, default=0,     #action='store_true',                 #only valide for Hapke model 
                        help='Asymmetry parameter in Hapke')
    parser.add_argument('--c', type=int, default=0,     #action='store_true',                 #only valide for Hapke model 
                        help='Backscattering parameter in Hapke')
    parser.add_argument('--B0', type=int, default=0,    #action='store_true',                 #only valide for Hapke model 
                        help='Amplitude of opposition peak in Hapke')
    parser.add_argument('--h', type=int, default=0,     #action='store_true',                 #only valide for Hapke model 
                        help='Width of opposition peak in Hapke')
    parser.add_argument('--theta', type=int, default=0, #action='store_true',                 #only valide for Hapke model 
                        help='Roughness in Hapke')

    parser.add_argument('--save_first_n_visu', type=int, default=0,
                        help='')

    ###################RPV BRDF model####################
    parser.add_argument('--funcM', type=int, default=0,
                        help='Minnaert function in RPV')
    parser.add_argument('--funcF', type=int, default=0,
                        help='Henyey-Greenstein function in RPV')
    parser.add_argument('--funcH', type=int, default=0,
                        help='Backscatter function in RPV')
    parser.add_argument('--dim_RPV', type=int, default=1, choices=[1, 3],
                        help='dimension of RPV parameters')

    ###################encoder for reflectance####################
    parser.add_argument('--fc_feat_ref', type=int, default=0,
                        help='number of fully connected units in the main block of layers')
    parser.add_argument('--fc_layers_ref', type=int, default=0,
                        help='number of fully connected layers in the main block of layers')

    parser.add_argument('--input_viewdir', type=int, default=0,
                        help='input_viewdir')

    parser.add_argument('--eval', type=int, default=0,  #evaluate model, in_ckpts cannot be empty if eval=1
                        help='')

    parser.add_argument('--mod_alt_bound', type=int, default=1,  
                        help='')

    args = parser.parse_args()

    if args.nr_spv_type == 0:
        if args.normal == 'analystic_learned':
            args.nr_spv_type = 1
        if args.normal == 'learned':
            args.nr_spv_type = 2
        if args.normal == 'analystic':
            args.nr_spv_type = 3

    if args.fc_feat_ref == 0:
        args.fc_feat_ref = args.fc_feat

    #disable args.sc_lambda if args.sun_v != 'learned'
    if args.sun_v != 'learned':
        args.sc_lambda = 0.


    args.infile_postfix += ".txt"
    exp_id = args.config_name if args.exp_name is None else args.exp_name
    args.exp_name = exp_id
    print("\nRunning {} - Using gpu {}\n".format(args.exp_name, args.gpu_id))

    os.makedirs("{}".format(args.logs_dir), exist_ok=True)
    with open("{}/opts.json".format(args.logs_dir), "w") as f:
        json.dump(vars(args), f, indent=2)


    return args

